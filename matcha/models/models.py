import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp
from transformers import EsmConfig
from transformers.models.esm.modeling_esm import EsmEncoder

from matcha.models.unimol_distance_embedding import DistanceBias
from matcha.utils.preprocessing import lig_feature_dims
from matcha.utils.transforms import compute_batch_ligand_centers


### ATOM ENCODER ###

class AtomEncoder(torch.nn.Module):
    '''
        The AtomEncoder generates embeddings for atoms, where 'x' represents a tensor of node features.
        Each feature tensor can include categorical features, scalar features, and optionally, language model embeddings.
        Categorical features pass through an embedding layer, producing outputs of dimension 'emb_dim'.
        Scalar features, which also include time variable embeddings, are processed through a linear layer.
    '''

    def __init__(self, emb_dim, feature_dims):
        # first element of feature_dims tuple is a list with the lenght of each categorical feature and the second is the number of scalar features
        super().__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims[0])
        self.num_scalar_features = feature_dims[1]
        for dim in feature_dims[0]:
            # +1 because 0 is the padding index, needed for nn.Embedding
            emb = torch.nn.Embedding(dim + 1, emb_dim, padding_idx=0)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

        if self.num_scalar_features > 0:
            self.linear = torch.nn.Linear(self.num_scalar_features, emb_dim)

    def forward(self, x):
        x_embedding = 0
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, :, i].long())
        if self.num_scalar_features > 0:
            x_embedding += self.linear(
                x[:, :, self.num_categorical_features:self.num_categorical_features + self.num_scalar_features])
        return x_embedding


### TIME EMBEDDER ###

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, scale_factor=1000):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.scale_factor = scale_factor

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0,
                                                 end=half, dtype=t.dtype) / half
        ).to(device=t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t = t * self.scale_factor
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


### COORDINATE POSITIONAL ENCODER ###

class CoordinatePositionalEncoder(nn.Module):
    """
    Embeds xyz positions into vector representations.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, coords):
        return self.mlp(coords)


### POSITIONAL ATTENTION ###

class AttentionWithBiasAndExtraOutput(torch.nn.Module):
    def __init__(self, dim, num_heads, dropout_rate=0.0, return_qk=True):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = torch.nn.Linear(dim, dim, bias=False)
        self.k_proj = torch.nn.Linear(dim, dim, bias=False)
        self.v_proj = torch.nn.Linear(dim, dim, bias=False)
        self.out_proj = torch.nn.Linear(dim, dim)

        self.attn_drop = torch.nn.Dropout(dropout_rate)
        self.proj_drop = torch.nn.Dropout(dropout_rate)
        self.return_qk = return_qk

    def forward(self, q, k, v, bias):
        """
        Args:
            q: Query tensor of shape (B, N, C)
            k: Key tensor of shape (B, S, C)
            v: Value tensor of shape (B, S, C)
            bias: Bias tensor of shape (B, num_heads, N, S)

        Returns:
            If return_qk is True:
                Tuple (output, qk), where:
                    - output: Output tensor of shape (B, N, C)
                    - qk: Attention score tensor of shape (B, num_heads, N, S)
            Otherwise:
                output: Output tensor of shape (B, N, C)
        """
        B, N, C = q.shape  # Batch size (B), sequence length of queries (N), embedding dimension (C)
        _, S, _ = k.shape  # Sequence length of keys/values (S)

        # Project queries, keys, and values
        q = self.q_proj(q).reshape(B, N, self.num_heads, self.head_dim).transpose(
            1, 2)  # Shape: (B, num_heads, N, head_dim)
        k = self.k_proj(k).reshape(B, S, self.num_heads, self.head_dim).transpose(
            1, 2)  # Shape: (B, num_heads, S, head_dim)
        v = self.v_proj(v).reshape(B, S, self.num_heads, self.head_dim).transpose(
            1, 2)  # Shape: (B, num_heads, S, head_dim)

        # Compute scaled dot-product attention
        # Shape: (B, num_heads, N, S)
        qk = (q @ k.transpose(-2, -1)) * self.scale
        # attn = qk + bias.unsqueeze(1)  # Add bias, resulting shape: (B, num_heads, N, S)
        attn = qk + bias
        # Apply softmax, shape remains: (B, num_heads, N, S)
        attn = F.softmax(attn, dim=-1)
        # Apply dropout, shape remains: (B, num_heads, N, S)
        attn = self.attn_drop(attn)

        # Compute attention output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # Shape: (B, N, C)
        x = self.out_proj(x)  # Shape: (B, N, C)
        x = self.proj_drop(x)  # Shape: (B, N, C)

        # Return the output and optionally the attention scores
        if self.return_qk:
            output = x, qk  # Shapes: (B, N, C), (B, num_heads, N, S)
        else:
            output = x  # Shape: (B, N, C)
        return output


### DiT BLOCKS ###

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class SelfAttentionDiTBlock(nn.Module):
    """
    A self-attention DiT block with adaptive layer norm zero (adaLN-Zero) conditioning using time embeddings.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout_rate=0.0, use_time=True):
        super().__init__()
        self.use_time = use_time
        self.attn = AttentionWithBiasAndExtraOutput(
            hidden_size, num_heads=num_heads, dropout_rate=dropout_rate)
        if self.use_time:
            self.norm1 = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6)
            mlp_hidden_dim = int(hidden_size * mlp_ratio)
            def approx_gelu(): return nn.GELU(approximate="tanh")
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden_dim),
                approx_gelu(),
                nn.Linear(mlp_hidden_dim, hidden_size)
            )
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        else:
            self.norm1 = nn.LayerNorm(hidden_size, eps=1e-5)
            self.output = SelfOutput(
                hidden_size, hidden_dropout_prob=dropout_rate)

    def forward(self, x, time_emb, pair_emb, is_padded_mask):
        # prepare correct attention biases
        seq_len = x.size(1)

        def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf")):
            if attn_mask is not None and padding_mask is not None:
                attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
                attn_mask = attn_mask.masked_fill(
                    padding_mask.to(torch.bool),
                    fill_val,
                )
            return attn_mask

        assert pair_emb is not None
        pair_emb = fill_attn_mask(pair_emb, is_padded_mask)

        # the main forward pass
        if self.use_time:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
                time_emb).chunk(6, dim=1)
            modulate_x = modulate(self.norm1(x), shift_msa, scale_msa)
            attn, qk = self.attn(modulate_x, modulate_x, modulate_x, pair_emb)
            x = x + gate_msa.unsqueeze(1) * attn
            pair_emb = pair_emb + qk
            x = x + \
                gate_mlp.unsqueeze(
                    1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            return x, pair_emb
        else:
            x_ln = self.norm1(x)
            attn, qk = self.attn(x_ln, x_ln, x_ln, pair_emb)
            pair_emb = pair_emb + qk
            x_ln = self.output(attn, x)
            return x_ln, pair_emb


def create_attention_mask(batch_num_rot_bonds, dtype):
    max_num_bonds = max(batch_num_rot_bonds)
    range_tensor = torch.arange(
        1, max_num_bonds + 1, device=batch_num_rot_bonds.device)
    attn_mask = batch_num_rot_bonds.unsqueeze(1) >= range_tensor.unsqueeze(0)

    # Makes broadcastable attention and causal masks so that future and masked tokens are ignored
    attn_mask = attn_mask[:, None, None, :]
    attn_mask = attn_mask.to(dtype=dtype)  # fp16 compatibility
    attn_mask = (1.0 - attn_mask) * torch.finfo(dtype).min
    return attn_mask


class MatchaModel(nn.Module):

    def __init__(self, feature_dim, num_heads=8, num_transformer_blocks=6, num_ligand_blocks=2,
                 frequency_embedding_size=256, llm_emb_dim=1280, num_kernel_pos_encoder=128,
                 tr_weight=1., rot_weight=1., tor_weight=1., dropout_rate=0., use_time=True,
                 predict_torsion_angles=True, timestep_scale_factor=1):
        super().__init__()
        self.tr_weight = tr_weight
        self.rot_weight = rot_weight
        self.tor_weight = tor_weight
        self.num_attn_heads = num_heads
        self.use_time = use_time
        self.predict_torsion_angles = predict_torsion_angles
        self.aa_types = ['-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                         'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        self.ligand_atom_types = range(
            lig_feature_dims[0][0] + 1)  # add padding token

        self.timestep_embedding = TimestepEmbedder(hidden_size=feature_dim,
                                                   frequency_embedding_size=frequency_embedding_size,
                                                   scale_factor=timestep_scale_factor)
        self.ligand_atom_encoder = AtomEncoder(feature_dim, lig_feature_dims)

        self.protein_embedder = Mlp(
            in_features=llm_emb_dim, hidden_features=480, out_features=feature_dim, drop=0)

        ligand_blocks = [SelfAttentionDiTBlock(hidden_size=feature_dim, num_heads=num_heads,
                                               dropout_rate=dropout_rate, use_time=use_time)
                         for _ in range(num_ligand_blocks)]
        self.ligand_attention_block_list = nn.ModuleList(ligand_blocks)

        self.ligand_num_types = len(
            self.ligand_atom_types) + 2  # Now two CLS tokens
        self.complex_num_types = self.ligand_num_types + \
            len(self.aa_types)  # add protein types also

        self.ligand_distance_encoder = DistanceBias(
            num_kernel=num_kernel_pos_encoder,
            num_attn_heads=num_heads,
            num_edge_types=self.ligand_num_types ** 2,
            feature_dim=feature_dim)
        self.complex_distance_encoder = DistanceBias(
            num_kernel=num_kernel_pos_encoder,
            num_attn_heads=num_heads,
            num_edge_types=self.complex_num_types ** 2,
            feature_dim=feature_dim)

        # Ligand-protein attention DiT blocks
        self_blocks = [SelfAttentionDiTBlock(hidden_size=feature_dim, num_heads=num_heads,
                                             dropout_rate=dropout_rate, use_time=use_time)
                       for _ in range(num_transformer_blocks)]
        self.self_attention_block_list = nn.ModuleList(self_blocks)

        rot_head_out_dim = 3
        self.tr_head = Mlp(in_features=feature_dim, out_features=3, drop=0)
        self.rot_head = Mlp(in_features=feature_dim,
                            out_features=rot_head_out_dim, drop=0)

        # Two CLS tokens
        self.cls_token_tr = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.cls_token_rot = nn.Parameter(torch.randn(1, 1, feature_dim))

        # coordinate positional encoding
        self.coord_pos_encoding = CoordinatePositionalEncoder(
            hidden_size=feature_dim)

        if self.predict_torsion_angles:
            tor_input_dim = feature_dim
            tor_config = EsmConfig(
                hidden_size=tor_input_dim,
                num_hidden_layers=2,
                num_attention_heads=num_heads,
                intermediate_size=tor_input_dim * 4,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                layer_norm_eps=1e-05,
                emb_layer_norm_before=False,
                token_dropout=True,
                esmfold_config=None,
                vocab_list=None,
                position_embedding_type='rotary',
            )
            tor_config._attn_implementation = 'eager'
            self.tor_decoder = EsmEncoder(tor_config)
            self.tor_head = Mlp(in_features=tor_input_dim,
                                out_features=1, drop=0)
            self.torsion_vector_encoding = CoordinatePositionalEncoder(
                hidden_size=feature_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.timestep_embedding.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embedding.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.use_time:
            for block in self.ligand_attention_block_list:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            for block in self.self_attention_block_list:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def get_edge_types(self, ligand_tokens, protein_tokens, num_types, device):
        # Update to account for two CLS tokens
        concat_list = [
            # CLS for translation
            torch.ones(
                (ligand_tokens.shape[0], 1), device=device, dtype=int) * (num_types - 2),
            # CLS for rotation
            torch.ones(
                (ligand_tokens.shape[0], 1), device=device, dtype=int) * (num_types - 1),
            ligand_tokens,
        ]
        if protein_tokens is not None:
            concat_list.append(protein_tokens + len(self.ligand_atom_types))

        # [batch, mol_sz + pocket_sz + 2]
        node_input = torch.concat(concat_list, dim=1)

        edge_types = node_input.unsqueeze(-1) * \
            num_types + node_input.unsqueeze(-2)

        # Symmetry for interactions
        N = edge_types.shape[1]
        upper_mask = torch.triu(torch.ones(N, N), diagonal=1).bool()
        min_values = torch.min(
            edge_types[:, upper_mask],
            edge_types.transpose(1, 2)[:, upper_mask]
        )

        edge_types[:, upper_mask] = min_values
        edge_types = edge_types.transpose(1, 2)
        edge_types[:, upper_mask] = min_values
        edge_types = edge_types.transpose(1, 2)
        return edge_types

    def get_distance_bias(self, pos, ligand_tokens, protein_tokens):

        if protein_tokens is None:
            # ligand-only distance encoder
            num_types = self.ligand_num_types
            distance_encoder = self.ligand_distance_encoder
        else:
            # complex distance encoder
            num_types = self.complex_num_types
            distance_encoder = self.complex_distance_encoder

        edge_types = self.get_edge_types(ligand_tokens=ligand_tokens,
                                         protein_tokens=protein_tokens,
                                         num_types=num_types,
                                         device=pos.device)
        protein_length = protein_tokens.shape[1] if protein_tokens is not None else None

        pair_emb = distance_encoder(
            pos=pos, edge_types=edge_types, protein_length=protein_length)
        return pair_emb

    def predict_torsion(self, batch, lig_seq):
        # average torsion embeddings for rotated atoms
        rotatable_complexes_sums = batch.ligand.num_rotatable_bonds

        max_num_rot_bonds = batch.ligand.num_rotatable_bonds.max()
        tor_inputs = torch.zeros((batch.ligand.pos.shape[0],
                                  max_num_rot_bonds,
                                  lig_seq.shape[-1]),
                                 dtype=lig_seq.dtype, device=lig_seq.device)

        for init_batch_idx, (mask_rotate, start, end) in enumerate(zip(batch.ligand.mask_rotate,
                                                                       batch.ligand.tor_ptr[:-1],
                                                                       batch.ligand.tor_ptr[1:])):

            if len(mask_rotate) > 0 and rotatable_complexes_sums[init_batch_idx] > 0:

                # get left and right embeddings of rotatable bond
                start_end_indices = batch.ligand.rotatable_bonds[start: end]
                mask_rotate = mask_rotate[:,
                                          :batch.ligand.num_atoms[init_batch_idx]]

                # get left and right embeddings of rotatable bond (and rotation vector)
                left_positions = batch.ligand.pos[init_batch_idx,
                                                  start_end_indices[:, 0]]
                right_positions = batch.ligand.pos[init_batch_idx,
                                                   start_end_indices[:, 1]]
                rot_vectors = left_positions - right_positions
                torsion_vector_encodings = self.torsion_vector_encoding(
                    rot_vectors)

                # Compute complex embedding for rotatable bonds
                complex_embedding = lig_seq[init_batch_idx,
                                            :mask_rotate.shape[1]].unsqueeze(0)

                # Element-wise multiplication with the rotatable atom mask (num_rotatable_bonds, seq_len, d)
                complex_embedding = complex_embedding * \
                    mask_rotate.unsqueeze(-1)

                # sum over seq_len Shape: (num_rotatable_bonds, d)
                complex_embedding = complex_embedding.sum(dim=1)

                # Normalize by the number of rotated (non-masked) elements per rotatable bond
                # Shape: (num_rotatable_bonds, 1)
                mask_sum = mask_rotate.sum(dim=1, keepdim=True)

                # Compute the final output
                # Shape: (batch_size, num_rotatable_bonds, d)
                complex_embedding = complex_embedding / mask_sum

                tor_inputs[init_batch_idx, :mask_rotate.shape[0]
                           ] = complex_embedding + torsion_vector_encodings

        nonempty_num_rot_bonds = batch.ligand.num_rotatable_bonds

        extended_attention_mask = create_attention_mask(
            nonempty_num_rot_bonds, lig_seq.dtype)
        tor_embeddings = self.tor_decoder(
            tor_inputs, attention_mask=extended_attention_mask)['last_hidden_state']

        v_tor = self.tor_head(tor_embeddings)

        # revert to rotatable_bonds shape (stack batch again)
        v_tor = torch.cat([complex_tor[:num_complex_bonds, 0] for complex_tor, num_complex_bonds in
                           zip(v_tor, batch.ligand.num_rotatable_bonds)])
        return v_tor

    def encode_ligand(self, batch, time_emb):
        # Convert ligand categorical atom features:
        lig_seq = self.ligand_atom_encoder(batch.ligand.x)

        # Add CLS tokens for translation and rotation
        ligand_centers = compute_batch_ligand_centers(batch).reshape(-1, 1, 3)
        cls_token_tr = self.cls_token_tr.expand(lig_seq.shape[0], -1, -1)
        cls_token_rot = self.cls_token_rot.expand(lig_seq.shape[0], -1, -1)

        # Concatenate cls_tr, cls_rot, and lig:
        lig_seq = torch.cat((
            cls_token_tr,
            cls_token_rot,
            lig_seq,
        ), dim=1)

        # Concatenate positions: Add two entries for the two CLS tokens
        lig_pos = torch.cat((
            ligand_centers,
            ligand_centers,  # two CLS tokens share the same center
            batch.ligand.pos
        ), dim=1)

        # Compute is_padded_mask for ligand:
        is_padded_mask_ligand = torch.cat([
            torch.zeros(batch.ligand.is_padded_mask.shape[0], 2,
                        device=batch.ligand.is_padded_mask.device, dtype=torch.bool),  # For the two CLS tokens
            batch.ligand.is_padded_mask,
        ], dim=-1)
        is_padded_mask_ligand = is_padded_mask_ligand.unsqueeze(1).unsqueeze(2)

        # Initialize ligand pair embeddings:
        ligand_pair_emb = self.get_distance_bias(lig_pos,
                                                 ligand_tokens=batch.ligand.x[:, :, 0],
                                                 protein_tokens=None)

        for block in self.ligand_attention_block_list:
            lig_seq, ligand_pair_emb = block(
                lig_seq, time_emb, ligand_pair_emb, is_padded_mask_ligand)
        del ligand_pair_emb

        # Add coordinate positional encoding
        lig_seq += self.coord_pos_encoding(lig_pos)

        return lig_seq, is_padded_mask_ligand, lig_pos

    def encode_complex(self, batch, predict_torsion):
        '''
        We assume that batch['ligand'].pos is not a starting position but a position at time batch.t!
        '''

        # Compute ligand time embeddings:
        time_emb = self.timestep_embedding(batch.ligand.t)

        lig_seq, is_padded_mask_ligand, lig_pos = self.encode_ligand(
            batch, time_emb)

        protein_seq = self.protein_embedder(batch.protein.x)

        # Compute is_padded_mask for the whole complex:
        is_padded_mask = torch.cat([
            is_padded_mask_ligand,
            batch.protein.is_padded_mask.unsqueeze(1).unsqueeze(2)
        ], dim=-1)

        # Initialize complex pair embeddings:
        # Add coordinate positional encoding
        protein_seq += self.coord_pos_encoding(batch.protein.pos)

        # Concatenate ligand with the protein:
        seq = torch.cat((
            lig_seq,
            protein_seq,
        ), dim=1)

        # Concatenate ligand and protein positions
        pos = torch.cat((
            lig_pos,
            batch.protein.pos
        ), dim=1)

        pair_emb = self.get_distance_bias(pos, ligand_tokens=batch.ligand.x[:, :, 0],
                                          protein_tokens=batch.protein.seq[:, :, 0])

        for block in self.self_attention_block_list:
            seq, pair_emb = block(seq, time_emb, pair_emb, is_padded_mask)

        lig_seq = seq[:, :lig_seq.shape[1], :]
        protein_seq = seq[:, lig_seq.shape[1]:, :]

        # Cut ligand from [cls_tr, cls_rot, ligand, protein]
        v_tor = None
        if self.predict_torsion_angles and predict_torsion and sum(batch.ligand.num_rotatable_bonds) > 0:
            lig_seq_tor = lig_seq

            v_tor = self.predict_torsion(
                batch, lig_seq=lig_seq_tor[:, 2:batch.ligand.pos.shape[1] + 2, :])
        return lig_seq, v_tor

    def forward_step(self, batch, predict_torsion=True):
        lig_seq, v_tor = self.encode_complex(
            batch, predict_torsion=predict_torsion)

        # Use the translation CLS for d_tr and the rotation CLS for d_rot
        v_tr = None
        v_rot = None
        Rest = None
        scoring_pred = None
        v_tr = self.tr_head(lig_seq[:, 0])  # Translation head
        v_rot = self.rot_head(lig_seq[:, 1])  # Rotation head
        return v_tr, v_rot, v_tor, Rest, scoring_pred

    def forward(self, batch, labels):
        v_tr, v_rot, v_tor, _, _ = self.forward_step(batch)

        loss = None

        output_dict = {
            "v_tr": v_tr,
            "v_rot": v_rot,
            "v_tor": v_tor,
            "loss": loss,
        }
        return output_dict
