import torch
import numpy as np
import pandas as pd
import seaborn as sns
import mlflow
from mlflow.models import infer_signature

EMBEDDER_SIGNATURE = infer_signature(torch.zeros(64, 19, 246).numpy(), torch.zeros(64, 19, 122).numpy())
GNN_BNORM_SIGNATURE = infer_signature(torch.zeros(64, 19, 122).numpy(), torch.zeros(64, 19, 122).numpy())
LIN_SIGNATURE = infer_signature(torch.zeros(64, 19, 122).numpy(), torch.zeros(64, 19, 122).numpy())
LEARNABLE_PADDING_SIGNATURE = infer_signature(torch.zeros(64, 19, 124).numpy(), torch.zeros(64, 19, 246).numpy())
TRANSFORMER1_SIGNATURE = infer_signature({'tgt': torch.zeros(64, 19, 246).numpy(), 'memory': torch.zeros(64, 19, 122).numpy()}, torch.zeros(64, 19, 246).numpy())
TRANSFORMER2_SIGNATURE = infer_signature({'tgt': torch.zeros(64, 19, 246).numpy(), 'memory': torch.zeros(64, 19, 122).numpy()}, torch.zeros(64, 19, 246).numpy())

def logEncoderParams(encoder):
    for band, encoder_decoder in encoder.encoder_decoders.items():
        patch_embedder = encoder_decoder.patch_embedder
        mlflow.pytorch.log_model(patch_embedder, f"{band}_patch_embedder", signature=EMBEDDER_SIGNATURE)

        # TODO: Cannot log GATConv since it is more complicated than tensor in tensor out
        gnn_bnorm = encoder_decoder.gnn_bnorm
        mlflow.pytorch.log_model(patch_embedder, f"{band}_gnn_bnorm", signature=GNN_BNORM_SIGNATURE)

        lin = encoder_decoder.lin
        mlflow.pytorch.log_model(lin, f"{band}_gnn_linear", signature=LIN_SIGNATURE)

        learnable_padding = encoder_decoder.learnable_padding
        mlflow.pytorch.log_model(learnable_padding, f"{band}_learnable_padding", signature=LEARNABLE_PADDING_SIGNATURE)

        transformer1 = encoder_decoder.transformer_decoder1
        mlflow.pytorch.log_model(transformer1, f"{band}_transformer_decoder1", signature=TRANSFORMER1_SIGNATURE)

        transformer2 = encoder_decoder.transformer_decoder2
        mlflow.pytorch.log_model(transformer2, f"{band}_transformer_decoder2", signature=TRANSFORMER2_SIGNATURE)

POSITION_ENCODER_SIGNATURE = infer_signature(torch.zeros(64, 19, 124).numpy(), torch.zeros(64, 19, 124).numpy())
E2R_SIGNATURE = infer_signature(torch.zeros(64, 19, 124).numpy(), torch.zeros(64, 4, 19, 19).numpy())
ATTENTION_MANIFOLD_SIGNATURE = infer_signature(torch.zeros(64, 4, 19, 19).numpy(), torch.zeros(64, 4, 19, 19).numpy())
R2E_SIGNATURE = infer_signature(torch.zeros(64, 4, 19, 19).numpy(), torch.zeros(64, 760).numpy())
R2E_LIN_SIGNATURE = infer_signature(torch.zeros(64, 760).numpy(), torch.zeros(64, 190).numpy())

def logContextualizerParams(contextualizer):
    for band, position_encoder in contextualizer.position_encoder.items():
        mlflow.pytorch.log_model(position_encoder, f"{band}_position_encoder", signature=POSITION_ENCODER_SIGNATURE)

    for band, band_e2r in contextualizer.wavelet_e2r.items():
        mlflow.pytorch.log_model(band_e2r, f"{band}_e2r", signature=E2R_SIGNATURE)
    
    for band, attention_manifold in contextualizer.wavelet_attention_manifolds.items():
        mlflow.pytorch.log_model(attention_manifold, f"{band}_attention_manifold", signature=ATTENTION_MANIFOLD_SIGNATURE)

    combined_attention = contextualizer.combined_attention
    mlflow.pytorch.log_model(combined_attention, "combined_attention", signature=ATTENTION_MANIFOLD_SIGNATURE)

    combined_r2e_tangent_space = contextualizer.combined_r2e_tangent_space
    mlflow.pytorch.log_model(combined_r2e_tangent_space, "combined_r2e_tangent_space", signature=R2E_SIGNATURE)

    combined_r2e_lin = contextualizer.combined_r2e_lin
    mlflow.pytorch.log_model(combined_r2e_lin, "combined_r2e_lin", signature=R2E_LIN_SIGNATURE)

TEMP_SIGNATURE = infer_signature(torch.zeros(1).numpy(), torch.zeros(1).numpy())
MAE_MASK_SIGNATURE = infer_signature(torch.zeros(64, 760).numpy(), torch.zeros(64, 760).numpy())

def logMENDRTrainerParams(temp1, band_mask):
    mlflow.pytorch.log(temp1, "LOO Temperature 1", signature=TEMP_SIGNATURE)
    for band, mask in band_mask.items():
        mlflow.pytorch.log(mask, f"{band}_MAE_mask", signature=MAE_MASK_SIGNATURE)
