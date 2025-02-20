import torch
import os
import mlflow
from mlflow.models import infer_signature

def logEncoderParams(encoder, step):
    torch.set_printoptions(precision=5, threshold=1e6, linewidth=1e3)

    for band, encoder_decoder in encoder.encoder_decoders.items():
        patch_embedder = encoder_decoder.patch_embedder
        mlflow.log_dict(patch_embedder.state_dict(), artifact_file="{band}_patch_embedder_weights_{step}.json")

        gnn_encoder = encoder_decoder.gnn_encoder
        mlflow.log_dict(gnn_encoder.state_dict(), artifact_file=f"{band}_gnn_encoder_weights_{step}.json")

        gnn_bnorm = encoder_decoder.gnn_bnorm
        mlflow.log_dict(gnn_bnorm.state_dict(), artifact_file=f"{band}_gnn_bnorm_weights_{step}.json")

        lin = encoder_decoder.lin
        mlflow.log_dict(lin.state_dict(), artifact_file=f"{band}_gnn_lin_weights_{step}.json")

        learnable_padding = encoder_decoder.learnable_padding
        mlflow.log_dict(learnable_padding, artifact_file=f"{band}_learnable_padding_{step}.json")

        transformer1 = encoder_decoder.transformer_decoder1
        mlflow.log_dict(transformer1.state_dict(), artifact_file=f"{band}_transformer_1_{step}.json")

        transformer2 = encoder_decoder.transformer_decoder2
        mlflow.log_dict(transformer2.state_dict(), artifact_file=f"{band}_transformer_2_{step}.json")

    torch.set_printoptions()

def logContextualizerParams(contextualizer, step):
    torch.set_printoptions(precision=5, threshold=1e6, linewidth=1e6)

    for band, position_encoder in contextualizer.position_encoder.items():
        mlflow.log_dict(position_encoder.state_dict(), artifact_file=f"{band}_position_encoder_{step}.json")

    ''' E2R doesn't have any weights
    for band, band_e2r in contextualizer.wavelet_e2r.items():
        mlflow.log_dict(band_e2r.state_dict(), artifact_file=f"{band}_band_e2r_weights_{step}.json")
    '''

    for band, attention_manifold in contextualizer.wavelet_attention_manifolds.items():
        mlflow.log_dict(attention_manifold.state_dict(), artifact_file=f"{band}_attention_manifold_{step}.json")

    combined_attention = contextualizer.combined_attention
    mlflow.log_dict(combined_attention.state_dict(), artifact_file=f"combined_attention_{step}.json")

    ''' R2E only has row, col indices as "params", which are the same no matter what
    combined_r2e_tangent_space = contextualizer.combined_r2e_tangent_space
    mlflow.log_dict(combined_r2e_tangent_space.state_dict(), artifact_file=f"combined_r2e_tangent_space_{step}.json")
    '''

    combined_r2e_lin = contextualizer.combined_r2e_lin
    mlflow.log_dict(combined_r2e_lin.state_dict(), artifact_file=f"combined_r2e_lin_weights_{step}.json")

    torch.set_printoptions()

def logMENDRTrainerParams(temp1, band_mask, step):
    torch.set_printoptions(precision=7)

    mlflow.log_dict({"LOO Temperature 1": str(temp1.item())}, artifact_file=f"LOO_Temperature_{step}.json")
    mask_log_dict = dict()
    for band, mask in band_mask.items():
        mask_log_dict[band] = mask.data
    mlflow.log_dict(mask_log_dict, f"mask_{step}.json")

    torch.set_printoptions()
