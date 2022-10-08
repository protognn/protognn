import torch.nn as nn

from layers import  MLP

import torch.nn.functional as F

from torch_geometric.nn import  GATConv
import torch
from torch_sparse import SparseTensor, matmul
#from proto_attention import ProtoAttention

from torch.nn import init


class ProtoAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_proto = config['proto']['num']
        self.iters = config['proto']['n_iter']
        self.eps = config['proto']['eps']
        dim = config['arch']['num_hiddens']
        self.scale = dim ** -0.5
        self.config = config


        self.proto_mu = nn.Parameter(torch.randn(config['data']['num_classes'],1, dim))

        self.proto_logsigma = nn.Parameter(torch.zeros(config['data']['num_classes'],1, dim))
        init.xavier_uniform_(self.proto_logsigma)

        self.proto = nn.Parameter(torch.randn(config['data']['num_classes'],self.num_proto, dim))
        init.xavier_uniform_(self.proto)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, config['proto']['hidden_dim'])

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def proto_conv(self, prototypes, x):
        for i in range(self.config['proto']['conv_iter']):
            x = self.norm_input(x)
            k, v = prototypes, prototypes
            q = x
            dots = torch.einsum('id,jd->ij', q, k) * self.scale # num_datapoints * num_proto
            attn = dots.softmax(dim=1) + self.eps  # num_datapoints * num_proto
            x = torch.einsum('jd,ij->id', v, attn) # num_datapoints * num_hidden
        return x


    def forward(self, inputs, all_proto=None, mu=None):
        n_class = len(inputs)
        device = inputs[0].device
        _, d = inputs[0].size()

        if all_proto is None:
            if self.config['proto']['init_strategy'] == 'random':
                all_proto = self.proto
            elif self.config['proto']['init_strategy'] == 'sample':
                mu = self.proto_mu.expand(-1, self.num_proto, -1)
                sigma = self.proto_logsigma.exp().expand(-1, self.num_proto, -1)
                all_proto = mu + sigma * torch.randn(mu.shape, device=device)
            elif self.config['proto']['init_strategy'] == 'fix_mu':
                mu = mu.view(n_class,1,-1)
                mu = mu.expand(-1, self.num_proto, -1)
                sigma = self.proto_logsigma.exp().expand(-1, self.num_proto, -1)
                all_proto = mu + sigma * torch.randn(mu.shape, device=device)

        output = []

        for i in range(n_class):
            x = inputs[i]
            k, v = self.to_k(x), self.to_v(x)
            proto = all_proto[i,:,:]

            for _ in range(self.iters):
                proto_prev = proto
                q = self.to_q(proto)
                dots = torch.einsum('id,jd->ij', q, k) * self.scale
                attn = dots.softmax(dim=0) + self.eps
                attn = attn / attn.sum(dim=-1, keepdim=True)
                updates = torch.einsum('jd,ij->id', v, attn)
                proto = self.gru(
                    updates.reshape(-1, d),
                    proto_prev.reshape(-1, d)
                )
                proto = self.norm_pre_ff(proto)
                proto = proto + self.mlp(proto)
            output.append(proto)


        return torch.stack(output)



class ProtoGNN(torch.nn.Module):
    def __init__(self, config, hidden=None):
        super().__init__()
        self.config = config
        if hidden:
            self.hidden_dim = hidden
        else:
            self.hidden_dim = config['arch']['num_hiddens']
        self.proto_attention = ProtoAttention(config)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.idx_train = config['idx_train']
        self.proto_conv = GATConv(self.hidden_dim, self.hidden_dim)
        self.lin = nn.Linear(config['arch']['num_hiddens'] * 2, config['arch']['num_hiddens'])

    def proto_to_nontrain_edges(self, x):
        non_train = (~self.idx_train).nonzero(as_tuple=False).view(-1)
        n_proto = self.config['data']['num_classes'] * self.config['proto']['num']
        dummy = torch.ones_like(non_train)
        edges = []
        dummy = torch.unsqueeze(dummy, 0)
        non_train = torch.unsqueeze(non_train, 0)
        for i in range(n_proto):
            edges.append(torch.cat((dummy * (i + x.size()[0]), non_train), dim=0))
        proto_edges = torch.cat(edges, dim=1)
        return proto_edges


    def get_prototypes(self, x, y):
        proto_mu = torch.zeros((self.config['data']['num_classes'], self.hidden_dim)).to(x.device)
        proto_labels = torch.ones((self.config['data']['num_classes'] * self.config['proto']['num'])).to(x.device)
        if self.config["proto"]["ce_loss"] == False:
            proto_idx_train = torch.zeros((self.config['data']['num_classes'] * self.config['proto']['num'])).to(x.device)
        else:
            proto_idx_train = torch.ones((self.config['data']['num_classes'] * self.config['proto']['num'])).to(x.device)
        x_train_by_class = []
        # x_mlp = self.mlp1(x)
        for i in range(self.config['data']['num_classes']):
            x_train_by_class.append(x[self.idx_train][y[self.idx_train] == i])
            proto_mu[i] = torch.mean(x[self.idx_train][y[self.idx_train] == i], dim=0)
            for j in range(self.config['proto']['num']):
                proto_labels[i * self.config['proto']['num'] + j] = i
        prototypes = self.proto_attention(x_train_by_class, mu=proto_mu)


        proto_to_nontrain_edges = self.proto_to_nontrain_edges(x)
        proto_to_nontrain_edges = proto_to_nontrain_edges.to(x.device)

        proto_edges = proto_to_nontrain_edges

        return prototypes, proto_mu, proto_labels, proto_idx_train, proto_edges


    def forward(self, prototypes, proto_edges, x, aggr="add"):
        x = self.norm(x)
        x_from_proto = self.proto_attention.proto_conv(prototypes.view(-1, self.hidden_dim), x[~self.idx_train])
        if aggr == 'add':
            x[~self.idx_train] = x[~self.idx_train] + x_from_proto
        elif aggr == 'concat':
            x[~self.idx_train] = torch.cat((x[~self.idx_train], x_from_proto), dim=1)
            x[~self.idx_train] = self.lin(x[~self.idx_train])
        x2 = torch.cat((x, prototypes.view(-1, self.hidden_dim)), dim=0)
        return x, x2




class LINKX_PROTO(nn.Module):
    def __init__(self, config, in_channels, hidden_channels, out_channels, num_nodes, dropout=.5, cache=False,
                 inner_activation=False, inner_dropout=False, init_layers_A=1, init_layers_X=1, num_layers=2):
        super(LINKX_PROTO, self).__init__()
        num_layers = config['linkx']['num_layers']
        self.mlpA = MLP(num_nodes, hidden_channels, hidden_channels, init_layers_A, dropout=0)
        self.mlpX = MLP(in_channels, hidden_channels, hidden_channels, init_layers_X, dropout=0)
        self.W = nn.Linear(2 * hidden_channels, hidden_channels)
        self.mlp_final = MLP(hidden_channels, hidden_channels, out_channels, num_layers, dropout=dropout)
        self.mlp_comp = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Dropout(0.5),
                                      nn.Linear(hidden_channels, hidden_channels))
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.A = None
        self.inner_activation = inner_activation
        self.inner_dropout = inner_dropout

        self.use_proto = config['arch']['proto']

        self.config = config
        self.idx_train = config['idx_train']
        self.ProtoGNN = ProtoGNN(config)

    def reset_parameters(self):
        self.mlpA.reset_parameters()
        self.mlpX.reset_parameters()
        self.W.reset_parameters()
        self.mlp_final.reset_parameters()

    def forward(self, edge_index, x, y):
        m = x.size()[0]
        row, col = edge_index
        row = row - row.min()
        A = SparseTensor(row=row, col=col,
                         sparse_sizes=(m, self.num_nodes)
                         ).to_torch_sparse_coo_tensor()

        xA = self.mlpA(A)
        xX = self.mlpX(x)
        x = torch.cat((xA, xX), axis=-1)
        x = self.W(x)

        if self.use_proto:
            prototypes, proto_mu, proto_labels, proto_idx_train, proto_edges = self.ProtoGNN.get_prototypes(xX, y)
        if self.inner_dropout:
            x = F.dropout(x)
        if self.inner_activation:
            x = F.relu(x)

        if self.use_proto:
            x, x2 = self.ProtoGNN(prototypes, proto_edges, x)
        x = F.relu(x + xA + xX)
        x_comp = self.mlp_comp(x)
        x = self.mlp_final(x)

        if self.use_proto:
            prototypes = self.mlp_final(x2[x.size()[0]:, :])
            prototypes_comp = self.mlp_comp(x2[x.size()[0]:, :])
            x_combined = torch.cat((x, prototypes), dim=0)
            y_combined = torch.cat((y, proto_labels), dim=0).long()
            idx_train_combined = torch.cat((self.idx_train, proto_idx_train.bool()), dim=0)
        else:
            y_combined = y
            idx_train_combined = self.idx_train
            x_combined = x
            prototypes = x
            x_comp=None
            prototypes_comp=None

        if self.training:
            return F.log_softmax(x_combined, dim=1), y_combined, idx_train_combined, prototypes, x_comp, prototypes_comp, self.idx_train
        else:
            return F.log_softmax(x, dim=1)

