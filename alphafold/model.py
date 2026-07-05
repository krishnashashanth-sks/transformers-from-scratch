import tensorflow as tf
from layers import DummyModelPass,RecyclingBlock
from tensorflow.keras import layers

# --- AlphaFold Model ---
class AlphaFoldModel(tf.keras.Model):
    def __init__(
        self,
        c_msa: int,
        c_pair: int,
        num_evoformer_blocks: int = 48,
        num_recycling_steps: int = 3,
        c_in_ipa: int = 256,
        c_hidden_scalar_ipa: int = 16,
        c_hidden_point_ipa: int = 4,
        num_heads_ipa: int = 8,
        num_points_ipa: int = 3,
        distogram_bins: int = 64,
        msa_vocab_size: int = 22,
        plddt_bins: int = 50,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.c_msa = c_msa
        self.c_pair = c_pair
        self.num_evoformer_blocks = num_evoformer_blocks
        self.num_recycling_steps = num_recycling_steps

        self.model_pass = DummyModelPass(
            c_msa=c_msa,
            c_pair=c_pair,
            c_in_ipa=c_in_ipa,
            c_hidden_scalar_ipa=c_hidden_scalar_ipa,
            c_hidden_point_ipa=c_hidden_point_ipa,
            num_heads_ipa=num_heads_ipa,
            num_points_ipa=num_points_ipa
        )

        self.recycling_block = RecyclingBlock(
            model_components=self.model_pass,
            num_recycling_steps=num_recycling_steps,
            c_msa=c_msa,
            c_pair=c_pair,
            c_in_ipa=c_in_ipa
        )

        self.distogram_head = layers.Dense(distogram_bins, name="distogram_head")
        self.masked_msa_head = layers.Dense(msa_vocab_size, name="masked_msa_head")
        self.plddt_head = layers.Dense(plddt_bins, name="plddt_head")

    def call(self, inputs):
        msa_input_features = inputs["msa_one_hot"]
        pair_input_features = inputs["co_evolutionary_features"]
        template_features = inputs.get("template_features", None)
        msa_mask = inputs["msa_mask"]
        pair_mask = inputs["pair_mask"]
        atom_mask = inputs["atom_mask"]

        (all_msa_outputs,
         all_pair_outputs,
         all_predicted_frames,
         all_predicted_coords) = self.recycling_block(
            msa_input_features=msa_input_features,
            pair_input_features=pair_input_features,
            template_features=template_features,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            atom_mask=atom_mask
        )

        final_msa_output = all_msa_outputs[-1]
        final_pair_output = all_pair_outputs[-1]

        distogram_logits = self.distogram_head(final_pair_output)

        masked_msa_logits = self.masked_msa_head(final_msa_output[:, 1, :, :])

        plddt_logits = self.plddt_head(final_msa_output[:, 1, :, :])

        predictions = {
            "final_predicted_frames": all_predicted_frames[-1],
            "final_predicted_coords": all_predicted_coords[-1],
            "distogram_logits": distogram_logits,
            "masked_msa_logits": masked_msa_logits,
            "plddt_logits": plddt_logits,
            "all_predicted_frames": all_predicted_frames,
            "all_predicted_coords": all_predicted_coords
        }

        return predictions
