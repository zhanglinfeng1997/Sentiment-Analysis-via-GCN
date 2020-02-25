import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.init as init

class SparseMM(torch.autograd.Function):
    def __init__(self, sparse):
        super(SparseMM, self).__init__()
        self.sparse = sparse.cuda()

    def forward(self, dense):
        return torch.bmm(self.sparse, dense)

    def backward(self, grad_output):
        grad_input = None
        sparse_t = []
        for item in self.sparse:
            sparse_t.append(torch.unsqueeze(item.t(), dim=0))
        sparse_t = torch.cat(sparse_t, dim=0)
        if self.needs_input_grad[0]:
            grad_input = torch.bmm(sparse_t, grad_output)
        return grad_input


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=None):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features)).cuda()
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        init.xavier_uniform(self.weight.data, gain=1)
        self.weight.data.uniform_(-stdv, stdv)#随机
        if self.bias is not None:
            init.xavier_uniform(self.bias.data, gain=1)
            self.bias.data.uniform_(-stdv, stdv)#随机

    def forward(self, input, adj):
        #   []
        weight_matrix = self.weight.repeat(input.shape[0], 1, 1)
        support = torch.bmm(input, weight_matrix)
        #print(adj.shape)
        #print(type(adj))
        output = SparseMM(adj)(support)
        if self.bias is not None:
            return output + self.bias.repeat(output.size(0))
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
