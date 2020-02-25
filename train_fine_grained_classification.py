import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
import torch.nn.functional as F
import datetime
from layers import GraphConvolution
import pickle
from scipy.sparse import csr_matrix
import torch.nn.init as init
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, fbeta_score


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)


def get_sentense_marix(x):
    one_matrix = np.zeros((140, 140), dtype=np.float32)
    for index, item in enumerate(x):
        one_matrix[index][index] = 1
        if not item:
            one_matrix[index, item-1] = 2
            one_matrix[item-1, index] = 3
    return torch.FloatTensor(one_matrix)




# h.p. define
torch.manual_seed(1)
EPOCH = 200
BATCH_SIZE = 32
LR = 0.001
HIDDEN_NUM = 64
HIDDEN_LAYER = 2
# process data
print("Loading data...")
max_document_length = 140

fr = open('data_train_noRen_noW2v.txt', 'rb')
x_train = pickle.load(fr)
y_train = pickle.load(fr)
length_train = pickle.load(fr)

fr = open('data_test.txt', 'rb')
x_dev = pickle.load(fr)
y_dev = pickle.load(fr)
length_dev = pickle.load(fr)
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y_train)))
print(shuffle_indices.shape)
print('x_train shape ', x_train.shape)

x_train = x_train[shuffle_indices]
y_train = y_train[shuffle_indices]

length_shuffled_train = length_train[shuffle_indices]
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).long()

length_dev = []
for item in x_dev:
    length_dev.append(list(item).index(0))
print(len(length_dev))

x_dev = torch.from_numpy(x_dev)
y_dev = torch.max(torch.from_numpy(y_dev).long(), dim=1)[1]

train_x = torch.LongTensor(x_train).cuda()
train_y = torch.LongTensor(y_train).cuda()

#   y = torch.LongTensor(y)
#test_x = torch.cat(test_x, dim=0)
#test_y = torch.LongTensor(test_y)
test_x = torch.LongTensor(x_dev).cuda()
test_y = torch.LongTensor(y_dev).cuda()

torch_dataset = Data.TensorDataset(train_x, train_y)
torch_testset = Data.TensorDataset(test_x, test_y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = Data.DataLoader(
    dataset=torch_testset,
    batch_size=128
)
print("data process finished")


class LSTM_GCN(nn.Module):
    def __init__(self):
        super(LSTM_GCN, self).__init__()
        self.embedding = nn.Embedding(76215, 300).cuda()
        self.lstm = nn.LSTM(
            input_size=300,  # dim of word vector
            hidden_size=180,  # dim of output of lstm nn`
            num_layers=2,  # num of hidden layers
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        ).cuda()
        self.batch1 = nn.BatchNorm1d(max_document_length).cuda()
        self.gc = GraphConvolution(360, 7)
        init.xavier_normal_(self.lstm.all_weights[0][0], gain=1)
        init.xavier_normal_(self.lstm.all_weights[0][1], gain=1)
        init.xavier_normal_(self.lstm.all_weights[1][0], gain=1)
        init.xavier_normal_(self.lstm.all_weights[1][1], gain=1)

    def forward(self, x_and_adj):
        x = x_and_adj[:, :max_document_length].cuda()
        adj = x_and_adj[:, -max_document_length:]
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x, None)
        out = self.batch1(lstm_out)
        out = F.relu(out)
        adj_Metrix = []
        for item in adj:
            adj_Metrix.append(torch.unsqueeze(get_sentense_marix(item), dim=0))
        adj_Metrix = torch.cat(adj_Metrix, dim=0)
        out_g1 = self.gc(out, adj_Metrix)
        out = torch.median(out_g1, 1)[0]
        return out


model = LSTM_GCN()
#model.cuda()
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)
loss_func = nn.CrossEntropyLoss()
print(model)
best = 0


def get_test():
    global best
    model.eval()
    print('start dev test')
    record = []
    for index, (batch_x, batch_y) in enumerate(test_loader):
        test_output = model(batch_x)
        test_output = list(torch.max(test_output, dim=1)[1].cpu().numpy())
        record.extend(test_output)
    label = list(test_y.cpu().numpy())
    y_true = label
    y_pred = record

    print("accuracy:", accuracy_score(y_true, y_pred))    # Return the number of correctly classified samples
    if accuracy_score(y_true, y_pred) > best:
        torch.save(model, "best_model.pth")
    print("macro_precision", precision_score(y_true, y_pred, average='macro'))
    print("micro_precision", precision_score(y_true, y_pred, average='micro'))

    # Calculate recall score
    print("macro_recall", recall_score(y_true, y_pred, average='macro'))
    print("micro_recall", recall_score(y_true, y_pred, average='micro'))

    # Calculate f1 score
    print("macro_f", f1_score(y_true, y_pred, average='macro'))
    print("micro_f", f1_score(y_true, y_pred, average='micro'))

    model.train()


f = open('accuracy_record.txt', 'w+')
f2 = open('loss_record.txt', 'w+')
loss_sum = 0
accuracy_sum = 0

for epoch in range(EPOCH):
    for index, (batch_x, batch_y) in enumerate(loader):
        right = 0
        if index == 0:
            get_test()
            loss_sum = 0
            accuracy_sum = 0
        #   one hot to scalar
        #batch_y = batch_y.cuda()
        output = model(batch_x)
        optimizer.zero_grad()
        #output = output.cuda()
        batch_y = torch.argmax(batch_y, dim=1)
        #print(batch_y)
        #print(output.size())
        #print(batch_y.size())
        loss = loss_func(output, batch_y)
        #   gcnloss = ((torch.matmul(model.gc.weight.t(), model.gc.weight) - i)**2).sum().cuda()
        #   loss += gcnloss * 0.000005
        lstmloss = 0
        for item in model.lstm.parameters():
            if len(item.shape) == 2:
                I = torch.eye(item.shape[1]).cuda()
                lstmloss += ((torch.matmul(item.t(), item)-I)**2).sum().cuda()
        loss += lstmloss * 0.00000005
        loss.backward()
        predict = torch.argmax(output, dim=1).cpu().numpy().tolist()
        label = batch_y.cpu().numpy().tolist()

        for i in range(0, batch_y.size(0)):
            if predict[i] == label[i]:
                right += 1
        optimizer.step()
        accuracy_sum += right/batch_y.size(0)
        loss_sum += float(loss)
        if index % 50 == 0:
            print("batch", index, "/ "+str(len(loader))+": ",  "\tloss: ", float(loss), "\taccuracy: ", right/batch_y.size(0))
    print('epoch: ', epoch, 'has been finish')
