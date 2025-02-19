import torch
import numpy as np
import pandas as pd
import seaborn as sns
import mlflow
from mlflow.models import infer_signature

import sys
sys.path.append("../Model/MENDR")

from MENDREncoder import MENDRAutoEncoder

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
        mlflow.pytorch.log_model(learnable_padding, f"{band}_learnable_padding", signature=)

        transformer1 = encoder_decoder.transformer_decoder1
        mlflow.pytorch.log_model(transformer1, f"{band}_transformer_decoder1", signature=TRANSFORMER1_SIGNATURE)

        transformer2 = encoder_decoder.transformer_decoder2
        mlflow.pytorch.log_model(transformer2, f"{band}_transformer_decoder2", signature=TRANSFORMER2_SIGNATURE)
