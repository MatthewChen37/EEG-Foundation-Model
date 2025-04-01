import torch
import os
import mlflow
from torch.utils.tensorboard import SummaryWriter


class MENDRLogger(object):

    def __init__(self):
        assert mlflow.active_run != None, "Need Active MLFLow Run"
        self.writer = SummaryWriter(log_dir=f"/home/hice1/mchen439/scratch/EEG-Foundation-Model/tensorboard_runs/{mlflow.active_run().info.run_id}")

    def log_model_weights(self, model, epoch):
            for name, param in model.named_parameters():
                self.writer.add_histogram(tag=f'{name}_weights', values=param, global_step=epoch)

    def log_model_gradients(self, model, epoch, name=None):
        if isinstance(model, torch.nn.Module):
            for name, module in model.named_parameters(): # Sometimes skip "High"
                if module.grad is None:
                    continue
                grad = module.grad.cpu()
                self.writer.add_histogram(f'{name}_gradient', values=grad, global_step=epoch)
        elif isinstance(model, torch.nn.Parameter) and name != None:
            grad = model.grad.cpu()
            self.writer.add_histogram(f'{name}_gradient', values=grad, global_step=epoch)
        else:
            raise Exception(f"Unknown object type: {type(model)}")
 
        
    ''' 
    MLFlow's Logging is extremely inconvenient for logging model weights
    Using Tensorboard instead
    '''
    def logEncoderParams(self, encoder, step):
        self.log_model_weights(encoder, step)
        '''
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
        '''
    def logContextualizerParams(self, contextualizer, step):
        self.log_model_weights(contextualizer, step)
        '''
        torch.set_printoptions(precision=5, threshold=1e6, linewidth=1e6)
        for band, position_encoder in contextualizer.WaveletContextualizer.position_encoder.items():
            mlflow.log_dict(position_encoder.state_dict(), artifact_file=f"{band}_position_encoder_{step}.json")

        E2R doesn't have any weights
        for band, band_e2r in contextualizer.wavelet_e2r.items():
            mlflow.log_dict(band_e2r.state_dict(), artifact_file=f"{band}_band_e2r_weights_{step}.json")

        for band, attention_manifold in contextualizer.WaveletContextualizer.wavelet_attention_manifolds.items():
            mlflow.log_dict(attention_manifold.state_dict(), artifact_file=f"{band}_attention_manifold_{step}.json")

        for band, spd_transform in contextualizer.WaveletContextualizer.wavelet_spd_transforms.items():
            mlflow.log_dict(spd_transform.state_dict(), artifact_file=f"{band}_spd_transform_{step}.json")

        combined_attention = contextualizer.CombinedContextualizer.combined_attention
        mlflow.log_dict(combined_attention.state_dict(), artifact_file=f"combined_attention_{step}.json")

        R2E only has row, col indices as "params", which are the same no matter what
        combined_r2e_tangent_space = contextualizer.combined_r2e_tangent_space
        mlflow.log_dict(combined_r2e_tangent_space.state_dict(), artifact_file=f"combined_r2e_tangent_space_{step}.json")

        combined_spd_transform1 = contextualizer.CombinedContextualizer.combined_spd_transform1
        mlflow.log_dict(combined_spd_transform1.state_dict(), artifact_file=f"combined_spd_transform_1_weights_{step}.json")

        combined_spd_transform2 = contextualizer.CombinedContextualizer.combined_spd_transform2
        mlflow.log_dict(combined_spd_transform2.state_dict(), artifact_file=f"combined_spd_transform_2_weights_{step}.json")

        torch.set_printoptions()
        '''
    def logMENDRTrainerParams(self, temp1, mask, step):
        torch.set_printoptions(precision=12)

        mlflow.log_dict({"LOO Temperature 1": str(temp1.item())}, artifact_file=f"LOO_Temperature_{step}.json")
        mlflow.log_dict({"Mask": mask.data}, artifact_file=f"Mask_{step}.json")

        torch.set_printoptions()

    def closeWriter(self):
        self.writer.close()
