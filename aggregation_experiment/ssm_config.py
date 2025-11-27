# ssm_config.py
from mamba_ssm.models.config_mamba import MambaConfig

configs = {
    'pure_ssm_1_layer': MambaConfig(
        d_model=512,
        d_intermediate=0,
        n_layer=1,
        vocab_size=50257,
        ssm_cfg={'layer': 'Mamba2'},
        attn_layer_idx=[],
        attn_cfg={},
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        pad_vocab_size_multiple=8,
        tie_embeddings=True
    ),
}