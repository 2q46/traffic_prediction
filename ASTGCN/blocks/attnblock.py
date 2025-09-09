from turtle import forward
import torch
import torch.nn as nn
import torch.nn.init as init

class SpatialAttention(nn.Module):

    def __init__(self, params):

        super(SpatialAttention, self).__init__()
        self.params = params

        self.num_nodes = params.num_nodes
        self.in_channels = params.in_features
        self.num_timesteps = params.num_timesteps

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        self.w1 = nn.Parameter(torch.tensor(size=self.params.w1_size))
        self.w2 = nn.Parameter(torch.tensor(size=self.params.w2_size))
        self.w3 = nn.Parameter(torch.tensor(size=self.params.w3_size))
        self.vs = nn.Parameter(torch.tensor(size=self.params.vs_size)) # learnables

        self.bias = nn.Parameter(torch.tensor(size=self.params.s_bias_size))

        self.reset_params()

    def reset_params(self):
        
        init.xavier_uniform_(self.w1)
        init.xavier_uniform_(self.w2)
        init.xavier_uniform_(self.w3)
        init.xavier_uniform_(self.vs)
        init.zeros_(self.bias)

    def forward(self, x):
        '''
        Input shape is (batch_size, nodes, in_features, time_steps) - all in self.params

        Equation: S_prime = softmax(V_s · sigmoid((X W1) W2 (W3 X)^T + bias))

        Output shape should be (batch_size, nodes, nodes)
        '''

        x_w1 = torch.einsum('bnft,t->bnf', x, self.w1) # bnf

        w3_x_transposed = torch.einsum('f,bnft->bnt', self.w3, x).permute(0,2,1) # btn

        x_w1_w2 = torch.matmul(x_w1, self.w2)   # bnf,ft -> bnt

        inner_prod = torch.einsum('bnt,btn->bnn', x_w1_w2, w3_x_transposed) # bnn

        S_prime = self.softmax(self.vs.unsqueeze(0) * self.sigmoid(inner_prod + self.bias.unsqueeze(0)), dim=-1)

        return S_prime # (batch_size, N, N)
        
class TemporalAttention(nn.Module):

    def __init__(self, params):

        super(TemporalAttention, self).__init__()
        self.params = params

        self.num_nodes = params.num_nodes
        self.in_channels = params.in_features
        self.num_timesteps = params.num_timesteps # x shape is (nodes, features, time_steps)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        self.u1 = nn.Parameter(torch.tensor(size=self.params.u1_size))
        self.u2 = nn.Parameter(torch.tensor(size=self.params.u2_size))
        self.u3 = nn.Parameter(torch.tensor(size=self.params.u3_size))
        self.vt = nn.Parameter(torch.tensor(size=self.params.us_size)) # learnables

        self.bias = nn.Parameter(torch.tensor(size=self.params.t_bias_size))

        self.reset_params()

    def reset_params(self):
        
        init.xavier_normal_(self.u1)
        init.xavier_normal_(self.u2)
        init.xavier_normal_(self.u3)
        init.xavier_normal_(self.vt)
        init.zeros_(self.bias)

    def forward(self, x):
        '''
        Input shape is (batch_size, nodes, in_features, time_steps) - all in self.params

        Equation: E_prime = softmax(V_t · sigmoid((X^T U1) U2 (U3 X) + bias))

        Output shape should be (batch_size, time_steps, time_steps)
        '''
        xT = x.permute(0, 3, 2, 1) # (batch_size, T, F, N) 

        # u1 (N,)
        # basically dot product
        # (batch_size, T, N, F) @ (N,) -> (batch_size, T, F) - sum over n dim (dot product)

        xT_u1 = torch.einsum('btfn,n->btf', xT, self.u1) 

        # u3 (F,) = learnable feature weights
        # basically dot product
        # (B, N, F, T) -> (batch_size, N, T) - sum over f dim

        u3_x = torch.einsum('bnft,f->bnt', x, self.u3)

        # u2 (F, N) = Feature-Node relation weight matrix
        # btf,fn -> btn

        xT_u1_u2 = torch.einsum('btf,fn->btn', xT_u1, self.u2)

        # btn, bnt -> btt
        inner_product = torch.einsum('btn, bnt', xT_u1_u2, u3_x) 

        E_prime = self.softmax(self.vt.unsqueeze(0) * self.sigmoid(inner_product + self.bias.unsqueeze(0)), dim=-1)

        return E_prime # (batch_size, T, T)