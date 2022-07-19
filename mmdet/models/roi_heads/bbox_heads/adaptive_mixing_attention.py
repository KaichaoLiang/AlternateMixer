import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveMixingAttention(nn.Module):
    def __init__(self, feat_dim, in_queries, out_queries):
        super(AdaptiveMixingAttention, self).__init__()
        self.feat_dim = feat_dim
        self.in_queries = in_queries
        self.out_queries = out_queries

        self.parameter_generator = nn.Sequential(
            nn.Linear(self.query_dim, self.n_groups*self.total_parameters),
        )

        self.out_proj = nn.Linear(
            self.eff_out_dim*self.out_points*self.n_groups, self.query_dim, bias=True
        )

        self.act = nn.ReLU(inplace=True)

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.parameter_generator[-1].weight)

    def forward(self, x, query):

        # Calculate FLOPs
        self.shadow(x, query)
        B, N, g, P, C = x.size()
        # batch, num_query, group, point, channel
        G = self.n_groups
        assert g == G
        # assert C*g == self.in_dim

        # query: B, N, C
        # x: B, N, G, Px, Cx

        global _dump_i

        '''generate mixing parameters'''
        params = self.parameter_generator(query)
        params = params.reshape(B*N, G, -1)

        out = x.reshape(B*N, G, P, C)

        M, S = params.split(
            [self.m_parameters, self.s_parameters], 2)

        '''you can choose one implementation below'''
        if False:
            out = out.reshape(
                B*N*G, P, C
            )

            M = M.reshape(
                B*N*G, self.eff_in_dim, self.eff_in_dim)
            S = S.reshape(
                B*N*G, self.out_points, self.in_points)

            '''adaptive channel mixing'''
            out = torch.bmm(out, M)
            out = F.layer_norm(out, [out.size(-2), out.size(-1)])
            out = self.act(out)

            '''adaptive spatial mixing'''
            out = torch.bmm(S, out)  # implicitly transpose and matmul
            out = F.layer_norm(out, [out.size(-2), out.size(-1)])
            out = self.act(out)
        else:
            M = M.reshape(
                B*N, G, self.eff_in_dim, self.eff_in_dim)
            S = S.reshape(
                B*N, G, self.out_points, self.in_points)

            '''adaptive channel mixing'''

            out = torch.matmul(out, M)
            out = F.layer_norm(out, [out.size(-2), out.size(-1)])
            out = self.act(out)

            '''adaptive spatial mixing'''
            out = torch.matmul(S, 
            )  # implicitly transpose and matmul
            out = F.layer_norm(out, [out.size(-2), out.size(-1)])
            out = self.act(out)

        '''linear transfomation to query dim'''
        out = out.reshape(B, N, -1)
        out = self.out_proj(out)

        out = query + out

        return out
