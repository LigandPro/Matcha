import torch
from torch import nn
import torch.nn.functional as F


class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()

        if hidden is None:
            hidden = input

        self.layer1 = torch.nn.Linear(input, hidden)
        self.layer2 = torch.nn.Linear(hidden, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x


@torch.jit.script
def gaussian(x_shifted, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * ((x_shifted / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, dist, edge_type):
        mul = self.mul(edge_type)
        bias = self.bias(edge_type)
        dist = mul * dist.unsqueeze(-1) + bias
        dist = dist.expand(-1, -1, -1, self.K)
        dist = dist - self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(dist, std).type_as(self.means.weight)


class DistanceBias(nn.Module):
    """
    Compute 3D attention bias according to the position information for each head.
    """

    def __init__(
        self,
        num_kernel,
        num_attn_heads,
        num_edge_types,
        feature_dim,
        use_reduced_bias=True,
    ):
        super(DistanceBias, self).__init__()
        self.num_kernel = num_kernel
        self.num_attn_heads = num_attn_heads
        self.use_reduced_bias = use_reduced_bias

        self.gaussian = GaussianLayer(
            self.num_kernel,
            edge_types=num_edge_types,
        )
        self.out_proj = NonLinear(self.num_kernel, num_attn_heads)
        self.vector_proj = NonLinear(3, num_attn_heads, hidden=128)

    def compute_edge_feature(self, dist, edge_types):
        edge_feature = self.gaussian(
            dist, edge_types
        )
        edge_feature = self.out_proj(edge_feature)
        return edge_feature

    def forward(self, pos, edge_types, protein_length=None):
        # Initialize pair embeddings:
        if protein_length is not None and self.use_reduced_bias:
            dlm_lig_lig = pos[:, None, :-protein_length] - \
                pos[:, :-protein_length, None]
            dlm_lig_prot = pos[:, None, :-protein_length] - \
                pos[:, -protein_length:, None]

            dist_lig_lig = 1 / ((dlm_lig_lig ** 2).sum(dim=-1) + 1)
            dist_lig_prot = 1 / ((dlm_lig_prot ** 2).sum(dim=-1) + 1)

            dlm_lig_lig = self.vector_proj(dlm_lig_lig)
            dlm_prot_lig = self.vector_proj(-dlm_lig_prot)
            dlm_lig_prot = self.vector_proj(dlm_lig_prot)
        else:
            dlm = pos[:, None] - pos[:, :, None]
            dist = 1 / ((dlm ** 2).sum(dim=-1) + 1)
            dlm = self.vector_proj(dlm)

        n_node = pos.size(1)
        batch_size = pos.size(0)
        if protein_length is not None and self.use_reduced_bias:
            edge_feature = torch.zeros(
                (batch_size, n_node, n_node, self.num_attn_heads), device=pos.device)

            ligand_edge_features = self.compute_edge_feature(
                dist_lig_lig, edge_types[:, :-protein_length, :-protein_length])
            lig_prot_edge_features = self.compute_edge_feature(
                dist_lig_prot, edge_types[:, -protein_length:, :-protein_length])

            edge_feature[:, :-protein_length, :-protein_length,
                         :] = ligand_edge_features + dlm_lig_lig
            edge_feature[:, -protein_length:, :-
                         protein_length] = lig_prot_edge_features + dlm_lig_prot
            edge_feature[:, :-protein_length, -protein_length:] = (
                lig_prot_edge_features + dlm_prot_lig).transpose(1, 2)

        else:
            edge_feature = self.compute_edge_feature(dist, edge_types) + dlm

        edge_feature = edge_feature.permute(0, 3, 1, 2).contiguous()
        edge_feature = edge_feature.reshape(
            (batch_size, self.num_attn_heads, n_node, n_node))

        return edge_feature
