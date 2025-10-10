import torch
from timm.models.vision_transformer import Mlp
from matcha.models import MatchaModel


class MatchaScoringModel(MatchaModel):
    def __init__(self, feature_dim, num_heads=8, num_transformer_blocks=6, num_ligand_blocks=2,
                 frequency_embedding_size=256, llm_emb_dim=480, num_kernel_pos_encoder=128,
                 dropout_rate=0.,  objective='ranking'):
        super().__init__(feature_dim=feature_dim, num_heads=num_heads,
                         num_transformer_blocks=num_transformer_blocks, num_ligand_blocks=num_ligand_blocks,
                         frequency_embedding_size=frequency_embedding_size,
                         llm_emb_dim=llm_emb_dim, num_kernel_pos_encoder=num_kernel_pos_encoder,
                         dropout_rate=dropout_rate, use_time=True, predict_torsion_angles=False)
        self.rmsd_head = Mlp(in_features=feature_dim, out_features=1, drop=0)

        self.initialize_weights()

    def forward_step(self, batch):
        seq, _ = self.encode_complex(batch, predict_torsion=False)
        rmsd_pred = (self.rmsd_head(seq[:, 0]), )  # RMSD head
        return rmsd_pred

    def forward(self, batch, labels, is_training=False):
        rmsd_pred = self.forward_step(batch)[0]
        loss = None

        output_dict = {
            "rmsd_pred": rmsd_pred,
            "loss": loss,
        }
        return output_dict
