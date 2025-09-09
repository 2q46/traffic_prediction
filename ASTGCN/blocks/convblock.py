from os import times_result
import torch
import torch.nn as nn

class TemporalConv(nn.Module):

    def __init__(self, params):
        
        super(TemporalConv, self).__init__()

        self.params = params
        
        self.conv = nn.Conv2d(
            in_channels=self.params.in_features,
            out_channels=self.params.out_features,
            kernel_size=(1, self.params.kernel_size),  # (spatial_dim, temporal_dim)
            stride=(1, self.params.stride),
            padding=(0, self.params.padding),
            dilation=(1, self.params.dilation)
        )

        self.relu = nn.ReLU()

    def forward(self, x, temporal_attn):

         # (B, N, O, T) -> (B, O, N, T)
        x = x.permute(0, 2, 1, 3)
        
        # Apply convolution along time dimension
        # conv weight shape: (O, F, 1, kernel_size)
        output = self.conv(x)  # (B, F, N, T)
        
        # Rearrange back: (B, F, N, T) -> (B, N, F, T)
        output = output.permute(0, 2, 1, 3)

        x_temporal = torch.einsum('bnft,btt->bnft', output, temporal_attn)  # Apply attention

        return self.relu(x_temporal)

class SpatialConv(nn.Module):

    '''
    Spectral graph convolution using Chebyshev polynomials for efficient approximation
    instead of computing the graph's eigenvalue-eigenvector decomp.
    '''

    def __init__(self, params):

        super(SpatialConv, self).__init__()

        self.params = params
        self.in_features = params.in_features
        self.out_features = params.out_features
        self.cheb_polynomials = params.cheb_polynomials

        self.polynomial_order = params.polynomial_order
        self.weight = nn.Parameter(torch.tensor(size=params.w_sc_size))
        self.relu = nn.ReLU()
        
        self.reset_params()

    def reset_params(self):

        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, spatial_attn):

            '''
            P_k represents the kth chebyshev polynomial. 
            P_1 represents the identity matrix for zero-hop neighbours, and P_2 is for one-hop neighbours.
            '''
            
            batch_size, num_nodes, in_features, num_timesteps = x.shape

            x_reshaped = x.permute(0,3,1,2) # (B, N, F, T)

            x_reshaped = x_reshaped.reshape(batch_size*num_timesteps, num_nodes, in_features) # (B*T, N, F)

            outputs = torch.zeros(size=(batch_size*num_timesteps, num_nodes, self.out_features)) # (B*T, N, O)

            spatial_attn_expanded = spatial_attn.unsqueeze(1)  # (B, 1, N, N)
            spatial_attn_expanded = spatial_attn_expanded.expand(-1, num_timesteps, -1, -1)  # (B, T, N, N)
            spatial_attn_expanded = spatial_attn_expanded.reshape(num_timesteps*batch_size, num_nodes, num_nodes)  # (B*T, N, N)

            for k in range(self.polynomial_order):

                Pk = self.cheb_polynomials[k] # (N, N)

                theta_k = self.weight[k] # (F, O)

                Pk_attn = Pk.unsqueeze(0) * spatial_attn_expanded # (1, N, N) (B*T, N, N) -> (B*T, N, N)

                aggregated = torch.bmm(Pk_attn, x_reshaped) # (B*T, N, N) (B*T, N, F) -> (B*T, N, F)

                outputs += torch.matmul(aggregated, theta_k) # (B*T, N, F) (F, O) -> (B*T, N, O)

            outputs = outputs.view(batch_size, num_timesteps, num_nodes, self.out_features) # (B*T, N, O) -> (B, T, N, O)
            outputs = outputs.permute(0, 2, 3, 1) # (B, N, O, T)

            return self.relu(outputs)

            


            
                