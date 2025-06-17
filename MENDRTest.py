import torch
import torch.nn as nn
import torch.utils.data as torchdata
from torch.utils.data import ConcatDataset
from torch.profiler import profile, record_function, ProfilerActivity

import numpy as np
import pandas as pd
import random, os, math
from torch_geometric.data import Data
from Model.MENDR.MENDRCommon import _make_mask_idxes
from Model.MENDR.Autoencoder.MENDREncoder import MENDRPatchEncoder
from Model.MENDR.Autoencoder.GNNSpatialHarmonizer import GNNSpatialHarmonizer, Dropout1dWithIndexTracking
from Model.MENDR.Contextualizer.Large.MENDRContextualizerLarge import MENDRWaveletContextualizer, MENDRCombinedContextualizer
from Model.MENDR.Contextualizer.Tiny.MENDRContextualizerTiny import MENDRContextualizerTiny
from Model.MENDR.MENDR import MENDR_model
from Model.MENDR.Contextualizer.Tiny.MENDRTinyPreTrainer import MENDRTinyPreTrainer
from Model.MENDR.Contextualizer.Large.MENDRLargeWaveletPreTrainer import MENDRLargeWaveletPreTrainer
from Model.MENDR.Contextualizer.Large.MENDRLargeCombinedPreTrainer import MENDRLargeCombinedPreTrainer
from Model.MENDR.mAtt.mAtt import tensor_exp, tensor_log, WaveletLogEuclideanMean
#from Model.MENDR.MENDRPreTrainer import MENDRPreTrainer
from Model.MENDR.mAtt.optimizer import MixOptimizer
from Datasets.datasetPretrain import WaveletPretrainDataset
from types import SimpleNamespace
from time import perf_counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BANDS = {'delta', 'theta', 'alpha', 'beta', 'gamma'}

def random_spd_batch(batch_size, n):
    return torch.tensor(np.array([
        random_spd_matrix(n) for i in range(batch_size)
    ])).float()

def random_spd_matrix(n):
    A = np.random.rand(n, n)
    return np.dot(A, A.transpose())

def check_sanity(m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d, nn.GroupNorm)):
            assert m.weight.data.mean() != 0

def testMENDRSuperPatching():
    example_input = {
            'delta': torch.randn(128, 19, 240).to(device).float(),
            'theta': torch.randn(128, 19, 240).to(device).float(),
            'alpha': torch.randn(128, 19, 480).to(device).float(),
            'beta': torch.randn(128, 19, 960).to(device).float(),
            'gamma': torch.randn(128, 19, 1920).to(device).float()
    }

    model = MENDRPatchEncoder(num_channels=19,
                            sampling_rate=128,
                            super_patch_seconds=5,
                            hidden_gnn_mlp_ratio=1,
                            n_gnn_transformer_layers=1,
                            num_subjects=1,
                            n_gnn_heads=2,
                            device=device,
                            )

    assert model.WAVELET_LENGTHS == {'delta': 4, 'theta': 4, 'alpha': 8, 'beta': 16,  'gamma': 32, 'high': 64}, f"model.WAVELET_LENGTHS: {model.WAVELET_LENGTHS}"
    assert model.WAVELET_SUPER_PATCH_LENGTHS == {'delta': 20, 'theta': 20, 'alpha': 40, 'beta': 80,  'gamma': 160, 'high': 320}, f"model.WAVELET_SUPER_PATCH_LENGTHS: {model.WAVELET_SUPER_PATCH_LENGTHS}"


    time_start = perf_counter()
    for i in range(0, 10):
        patchified_data = model._super_patchify(example_input)
    time_end = perf_counter()
    print(f"Time taken to patchify 5 seconds: {time_end - time_start}")

    expected_shape = {
        'delta': torch.Size([128, 12, 19, 20]),
        'theta': torch.Size([128, 12, 19, 20]),
        'alpha': torch.Size([128, 12, 19, 40]),
        'beta':  torch.Size([128, 12, 19, 80]),
        'gamma': torch.Size([128, 12, 19, 160]),
    }

    for band in patchified_data:
        assert patchified_data[band].shape == expected_shape[band], f"{band}: Actual Shape: {patchified_data[band].shape} Expected Shape: {expected_shape[band]}"

    model = MENDRPatchEncoder(num_channels=19,
                            sampling_rate=128,
                            super_patch_seconds=1,
                            hidden_gnn_mlp_ratio=1,
                            n_gnn_transformer_layers=1,
                            num_subjects=1,
                            n_gnn_heads=2,
                            device=device,
                            )
    time_start = perf_counter()
    for i in range(0, 10):
        patchified_data = model._super_patchify(example_input)
    time_end = perf_counter()
    print(f"Time taken to patchify 1 second: {time_end - time_start}")

    model = MENDRPatchEncoder(num_channels=19,
                            sampling_rate=128,
                            super_patch_seconds=10,
                            hidden_gnn_mlp_ratio=1,
                            n_gnn_transformer_layers=1,
                            num_subjects=1,
                            n_gnn_heads=2,
                            device=device,
                            )
    time_start = perf_counter()
    for i in range(0, 10):
        patchified_data = model._super_patchify(example_input)
    time_end = perf_counter()
    print(f"Time taken to patchify 10 seconds: {time_end - time_start}")

def testDropout1dWithIndexTracking():
    torch.manual_seed(42)
    np.random.seed(42)
    example_input = torch.randn(2, 2, 10, 8).to(device).float()

    dropout = Dropout1dWithIndexTracking(p=0.1)

    with torch.no_grad():
        output = dropout(example_input)

    assert output.shape == example_input.shape, f"Output Shape: {output.shape} does not match {example_input.shape}"

    dropped_indices = dropout.dropped_indices
    assert dropped_indices.shape == torch.Size([2, 2, 10]), f"Dropped Indices Shape: {dropped_indices.shape} does not match {torch.Size([2, 2, 10])}"

    assert output[0, 0, 5].all() == 0, f"Output[0, 0, 2] is not 0: {output[0, 0, 5]}"
    
def testMakeMaskIdxes():
    torch.manual_seed(42)
    np.random.seed(42)
    batch_size = 2
    num_patches = 3
    mask_ratio = 0.5
    mask_idxes = _make_mask_idxes(batch_size, num_patches, mask_ratio)
    assert mask_idxes.shape == torch.Size([batch_size, num_patches]), f"Mask Indices Shape: {mask_idxes.shape}"
    assert mask_idxes.dtype == torch.bool, f"Mask Indices dtype: {mask_idxes.dtype}"
    mask = torch.tensor(random_spd_matrix(3)).float().to(device)

    test_batch = torch.randn(batch_size, num_patches, 3, 3).to(device)
    
    test_batch[mask_idxes] = mask

    for batch_idx in range(batch_size):
        for masked_epoch_idx in range(num_patches):
            if mask_idxes[batch_idx, masked_epoch_idx]:
                assert torch.allclose(test_batch[batch_idx, masked_epoch_idx], mask), f"Masked epoch {masked_epoch_idx} not equal to mask: \n {test_batch[batch_idx, masked_epoch_idx]} \n {mask}"
            else:
                assert torch.allclose(test_batch[batch_idx, masked_epoch_idx], test_batch[batch_idx, masked_epoch_idx]), f"Masked epoch {masked_epoch_idx} not equal to unmasked: \n {test_batch[batch_idx, masked_epoch_idx]} \n {mask}"

def testMENDRBatchWiseMatrixSimilarity():
    args = SimpleNamespace(
        learning_rate = 0.001,
        l2_weight_decay = 0.001,
        save_model_directory = None,
        mask_span = 5,
        temp = 0.01,
        training_params = SimpleNamespace(
            mask_ratio = 0.5,
            gradient_clip_value = 1e7,
            scheduler_after_batch = False,
            ckpt_dir = None
        ),
        negatives_loo = 10,
        enc_feat_l2 = 0.001,
        multi_gpu = False,
        ckpt_dir="./checkpoint",
        random_state=42,
        T_max=10,
        eta_min=0.001,
    )

    autoencoder = MENDRPatchEncoder(num_channels=19,
                            sampling_rate=128,
                            super_patch_seconds=5,
                            hidden_gnn_mlp_ratio=1,
                            n_gnn_transformer_layers=1,
                            num_subjects=1,
                            n_gnn_heads=2,
                            device=device,
                            )

    wavelet_contextualizer = MENDRWaveletContextualizer(num_channels=19, out_dim=24).to(device)
    optim_params = list(autoencoder.parameters()) + list(wavelet_contextualizer.parameters())
    optimizer = torch.optim.AdamW(optim_params,
                betas=(0.9, 0.99),
                lr=args.learning_rate,
                weight_decay=args.l2_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                T_max=args.T_max,
                eta_min=args.eta_min)
    mix_optimizer = MixOptimizer(optimizer, scheduler)
    trainer = MENDRLargeWaveletPreTrainer(autoencoder, wavelet_contextualizer,
                                    mix_optimizer, cfg=args, cuda=device)

    batch_size = 2
    num_patches = 3

    batch_zeros = torch.randn(1, 19, 19).to(device).float()
    batch_zeros = batch_zeros.repeat(batch_size*num_patches, 1, 1)
    output = trainer._batchWiseMatrixSimilarity(batch_zeros, batch_zeros)
    assert torch.all(output > 0), "Batchwise Matrix Similarity not greater than zero"
    assert torch.allclose(output, output.mT), "Batchwise Matrix Similarity not symmetric"

    batch_A = torch.ones(batch_size*num_patches, 19, 19).to(device).float()
    output = trainer._batchWiseMatrixSimilarity(batch_A, batch_A)

    batch_B = torch.zeros(batch_size*num_patches, 19, 19).to(device).float()
    output = trainer._batchWiseMatrixSimilarity(batch_A, batch_B)

    assert output.shape == torch.Size([batch_size * num_patches, batch_size * num_patches]), f"Output shape: {output.shape} does not match {torch.Size([batch_size * num_patches, batch_size * num_patches])}"
    assert torch.allclose(output, output.mT), "Batchwise Matrix Similarity not symmetric"


def testEncoder():
    example_input = {
            'delta': torch.randn(128, 19, 240).to(device).float(),
            'theta': torch.randn(128, 19, 240).to(device).float(),
            'alpha': torch.randn(128, 19, 480).to(device).float(),
            'beta': torch.randn(128, 19, 960).to(device).float(),
            'gamma': torch.randn(128, 19, 1920).to(device).float()
    }

    edge_indices = []
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]).to(device)
    for idx in range(4): # 4 for batch_size
        edge_index_copy = edge_index.clone()
        edge_index_copy += idx * 19
        edge_indices.append(edge_index_copy)
    batch_edge_index = torch.cat(edge_indices, dim=1)

    edge_attributes = []
    for idx in range(4):
        edge_attributes.append(torch.randn(4, 1).clone())
    batch_edge_attributes = torch.cat(edge_attributes, dim=0)

    example_graph = Data(edge_index=batch_edge_index, edge_attr=batch_edge_attributes).to(device)

    example_input['graph'] = example_graph
    subjects = torch.randint(0, 128, (128,)).to(device)
    example_input['subject_idx'] = subjects

    encoder = MENDRPatchEncoder(num_channels=19,
                            sampling_rate=128,
                            super_patch_seconds=2,
                            hidden_gnn_mlp_ratio=1,
                            n_gnn_transformer_layers=1,
                            num_subjects=128,
                            n_gnn_heads=2,
                            device=device,
                            ).to(device)

    _, encodings, decodings = encoder(example_input)
    assert encodings.keys() == BANDS
    assert decodings.keys() == BANDS

    assert decodings['delta'].shape == torch.Size([3840, 19, 8]), f"Actual Shape: {decodings['delta'].shape}" 
    assert decodings['theta'].shape == torch.Size([3840, 19, 8]), f"Actual Shape: {decodings['theta'].shape}" 
    assert decodings['alpha'].shape == torch.Size([3840, 19, 16]), f"Actual Shape: {decodings['alpha'].shape}" 
    assert decodings['beta'].shape == torch.Size([3840, 19, 32]), f"Actual Shape: {decodings['beta'].shape}" 
    assert decodings['gamma'].shape == torch.Size([3840, 19, 64]), f"Actual Shape: {decodings['gamma'].shape}"

    assert encodings['delta'].shape == torch.Size([128, 30, 19, 192]), f"Actual Shape: {encodings['delta'].shape}" 
    assert encodings['theta'].shape == torch.Size([128, 30, 19, 192]), f"Actual Shape: {encodings['theta'].shape}"
    assert encodings['alpha'].shape == torch.Size([128, 30, 19, 384]), f"Actual Shape: {encodings['alpha'].shape}" 
    assert encodings['beta'].shape == torch.Size([128,  30, 19, 768]), f"Actual Shape: {encodings['beta'].shape}" 
    assert encodings['gamma'].shape == torch.Size([128, 30, 19, 1536]), f"Actual Shape: {encodings['gamma'].shape}"

def testLargeContextualizerBatchLEM():
    # Eigenvalues are 1, 3
    example_SPD = torch.tensor([
        [2, 1],
        [1, 2]
    ]).float().to(device)

    # Batch size is 4, patches = 4
    example_SPD_batch = example_SPD.repeat(4, 1, 1).to(device)

    example_input = [
        example_SPD_batch.clone().to(device),
        example_SPD_batch.clone().to(device),
        example_SPD_batch.clone().to(device),
        example_SPD_batch.clone().to(device)
    ]

    wavelet_contextualizer = MENDRWaveletContextualizer(num_channels=19, out_dim=24).to(device)
    batch_output = wavelet_contextualizer._batch_LogEuclideanMean(example_input, 'delta')

    for batch_idx in range(batch_output.shape[0]):
        assert torch.allclose(batch_output[batch_idx], example_SPD), f"Batch LEM Not equal: \n Actual: {batch_output[batch_idx]} \n Expected: {example_SPD}"


def testWaveletLEM():
    # Eigenvalues are 1, 3
    example_SPD = torch.tensor([
        [2, 1],
        [1, 2]
    ]).float().to(device)

    # Batch size is 4
    example_SPD_batch = example_SPD.repeat(4, 1, 1).to(device)

    expected_orthonormal_eigenvectors = torch.tensor([
        [math.sqrt(2) / 2, -math.sqrt(2) / 2],
        [math.sqrt(2) / 2, math.sqrt(2) / 2],
    ])

    example_input = {
            'delta': example_SPD_batch.clone().to(device),
            'theta': example_SPD_batch.clone().to(device),
            'alpha': example_SPD_batch.clone().to(device),
            'beta':  example_SPD_batch.clone().to(device),
            'gamma': example_SPD_batch.clone().to(device)
    }

    combined_output = WaveletLogEuclideanMean(example_input)
    for batch_idx in range(combined_output.shape[0]):
        assert torch.allclose(combined_output[batch_idx, 0], example_SPD), f"Combined LEM Not equal: \n Actual: {combined_output[batch_idx, 0]} \n Expected: {example_SPD}"


def testContextualizerTiny():
    example_input = {
            'delta': torch.randn(4, 10, 19, 8*24).to(device).float(),
            'theta': torch.randn(4, 10, 19, 8*24).to(device).float(),
            'alpha': torch.randn(4, 10, 19, 16*24).to(device).float(),
            'beta':  torch.randn(4, 10, 19, 32*24).to(device).float(),
            'gamma': torch.randn(4, 10, 19, 64*24).to(device).float()
    }

    with torch.no_grad():
        contextualizer = MENDRContextualizerTiny(num_channels=19, out_dim=24).to(device)
        combined_manifold_output, combined_manifold_output_masked, mask_idxes = contextualizer(example_input, batch_size=4, num_patches=10)

        print("Positional Encoder Parameters: ", sum(p.numel() for p in contextualizer.position_encoders.parameters() if p.requires_grad))
        print("Total number of Tiny parameters: ", sum(p.numel() for p in contextualizer.parameters() if p.requires_grad))
        assert combined_manifold_output.shape == torch.Size([4, 10, 19, 19]), f"Incorrect output shape: {combined_manifold_output.shape}"
        assert combined_manifold_output_masked.shape == torch.Size([4, 10, 19, 19]), f"Incorrect output shape: {combined_manifold_output_masked.shape}"

def testContextualizerLarge():
    example_input = {
            'delta': torch.randn(4, 11, 19, 37).to(device).float(),
            'theta': torch.randn(4, 11, 19, 37).to(device).float(),
            'alpha': torch.randn(4, 11, 38, 37).to(device).float(),
            'beta': torch.randn(4, 11, 76, 37).to(device).float(),
            'gamma': torch.randn(4, 11, 114, 37).to(device).float()
    }

    with torch.no_grad():
        contextualizer = MENDRContextualizerLarge(device,
            delta_encoded_h=19,
            theta_encoded_h=19,
            alpha_encoded_h=38,
            beta_encoded_h=76,
            gamma_encoded_h=114,
            high_encoded_h=152,
            temp=10.0,
        )
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            combined_manifold_output, wavelet_manifold_output, _ = contextualizer(example_input, batch_size=4, patch_num=11)
        df = pd.DataFrame({e.key:e.__dict__ for e in prof.key_averages()}).T
        df[['count', 'cpu_time_total', 'device_time_total']].sort_values(['device_time_total', 'cpu_time_total'], ascending=False)
        df.to_csv("ProfileData/LargeContextualizer.csv", float_format='%.5f')
        print("Total number of Large parameters: ", sum(p.numel() for p in contextualizer.parameters() if p.requires_grad))

        wavelet_manifold_output_delta = wavelet_manifold_output['delta']
        wavelet_manifold_output_theta = wavelet_manifold_output['theta']
        wavelet_manifold_output_alpha = wavelet_manifold_output['alpha']
        wavelet_manifold_output_beta = wavelet_manifold_output['beta']
        wavelet_manifold_output_gamma = wavelet_manifold_output['gamma']

        assert wavelet_manifold_output_delta.shape == torch.Size([4, 11, 19, 19]), f'Delta Wavelet Manifold Shape:{wavelet_manifold_output_delta.shape}'
        assert wavelet_manifold_output_theta.shape == torch.Size([4, 11, 19, 19]), f'Theta Wavelet Manifold Shape:{wavelet_manifold_output_theta.shape}'
        assert wavelet_manifold_output_alpha.shape == torch.Size([4, 11, 19, 19]), f'Alpha Wavelet Manifold Shape:{wavelet_manifold_output_alpha.shape}'
        assert wavelet_manifold_output_beta.shape == torch.Size([4, 11, 19, 19]), f'Beta Wavelet Manifold Shape:{wavelet_manifold_output_beta.shape}'
        assert wavelet_manifold_output_gamma.shape == torch.Size([4, 11, 19, 19]), f'Gamma Wavelet Manifold Shape:{wavelet_manifold_output_gamma.shape}'

        assert combined_manifold_output.shape == torch.Size([4, 11, 19, 19]), f'Combined Manifold Shape: {combined_manifold_output.shape}'
        assert not torch.any(torch.isnan(combined_manifold_output)), "Combined Manifold contains NaN values"


def testMENDRLargeCombinedContextualizerMasking():
    example_input = {
            'delta': torch.randn(44, 19, 19).to(device).float(),
            'theta': torch.randn(44, 19, 19).to(device).float(),
            'alpha': torch.randn(44, 19, 19).to(device).float(),
            'beta': torch.randn(44, 19, 19).to(device).float(),
            'gamma': torch.randn(44, 19, 19).to(device).float()
    }

    with torch.no_grad():
        contextualizer = MENDRContextualizerLarge(device,
            delta_encoded_h=19,
            theta_encoded_h=19,
            alpha_encoded_h=38,
            beta_encoded_h=76,
            gamma_encoded_h=114,
            high_encoded_h=152,
            temp=10.0,
        )
        true_LEM = contextualizer.CombinedContextualizer._wavelet_LogEuclideanMean(example_input)
        combined_manifold_output, mask_idxes = contextualizer.CombinedContextualizer(
            example_input, [4, 11, -1], mask_ratio=0.5)

        true_LEM = true_LEM.view(4, 11, 19, 19)
        combined_manifold_output = combined_manifold_output.view(4, 11, 19, 19)

        assert len(mask_idxes) == 4, f"Did not correctly make batch indices: {len(mask_idxes)}"
        for batch_idx in range(4):
            for batch_mask_idx in range(11):
                if mask_idxes[batch_idx, batch_mask_idx]:
                    assert not torch.allclose(true_LEM[batch_idx, batch_mask_idx], combined_manifold_output[batch_idx, batch_mask_idx])

def testMENDRPreTrainerLOOLoss():
    args = SimpleNamespace(
    encoder_grad_frac = 0.5,
    learning_rate = 0.001,
    l2_weight_decay = 0.001,
    save_model_directory = None,
    mask_ratio = 0.5,
    mask_span = 5,
    temp = 0.01,
    delta_reconstructive_loss_pref = 1.0,
    theta_reconstructive_loss_pref = 1.0,
    alpha_reconstructive_loss_pref = 1.0,
    beta_reconstructive_loss_pref = 1.0,
    gamma_reconstructive_loss_pref = 1.0,
    contrastive_combined_loss_pref = 1e3,
    contrastive_wavelet_loss_pref = 1e3,
    gradient_clip_value = 1e7,
    negatives_loo = 10,
    enc_feat_l2 = 0.001,
    multi_gpu = False,
    ckpt_dir="./checkpoint",
    random_state=42
    )

    mendr = MENDR_model(device)
    trainer = MENDRPreTrainer(mendr, args)
    optimizer = torch.optim.Adam(trainer.parameters())
    optimizer = MixOptimizer(optimizer)
    trainer.set_optimizer(optimizer)
    with torch.autograd.detect_anomaly():
        embeddings = {
            'delta': random_spd_batch(8, 19).to(device),
            'theta': random_spd_batch(8, 19).to(device),
            'alpha': random_spd_batch(8, 19).to(device),
            'beta': random_spd_batch(8, 19).to(device),
            'gamma': random_spd_batch(8, 19).to(device),
        }
        loss, correct, pairs = trainer.leave_one_out(embeddings, nn.CrossEntropyLoss(), negatives=3)

    assert pairs == 30, f"Pairs is not 30: {pairs}"
    assert loss > 0, f"Loss is not greater than 0: {loss}"


def testMENDRPreTrainerMAEReconLoss():
    args = SimpleNamespace(
    encoder_grad_frac = 0.5,
    learning_rate = 0.001,
    l2_weight_decay = 0.001,
    save_model_directory = None,
    mask_ratio = 0.5,
    mask_span = 5,
    delta_reconstructive_loss_pref = 1.0,
    theta_reconstructive_loss_pref = 1.0,
    alpha_reconstructive_loss_pref = 1.0,
    beta_reconstructive_loss_pref = 1.0,
    gamma_reconstructive_loss_pref = 1.0,
    gradient_clip_value = 1e7,
    contrastive_combined_loss_pref = 1e3,
    contrastive_wavelet_loss_pref = 1e3,
    temp = 0.01,
    negatives_loo = 10,
    enc_feat_l2 = 0.001,
    multi_gpu = False,
    ckpt_dir="./checkpoint",
    random_state=42
    )

    mendr = MENDR_model(device)
    trainer = MENDRPreTrainer(mendr, args)
    optimizer = torch.optim.Adam(trainer.parameters())
    optimizer = MixOptimizer(optimizer)
    trainer.set_optimizer(optimizer)
    with torch.autograd.detect_anomaly():
        wavelet_manifold_output = {
            'delta': random_spd_batch(8, 19).to(device),
            'theta': random_spd_batch(8, 19).to(device),
            'alpha': random_spd_batch(8, 19).to(device),
            'beta': random_spd_batch(8, 19).to(device),
            'gamma': random_spd_batch(8, 19).to(device),
        }

        for band, batch in wavelet_manifold_output.items():
            assert torch.allclose(batch, batch.mT, atol=(10 ** -7)), f"{band}"

        riemannian_loss, combined_manifold_output, combined_manifold_output_masked, mask_idxes = trainer.epochMaskedRecon(wavelet_manifold_output, [2, 4, -1], nn.MSELoss())
    assert riemannian_loss > 0, f"Loss is not greater than 0: {riemannian_loss}"

    assert combined_manifold_output.shape == torch.Size([2, 4, 19, 19]), f"Combined Manifold Shape does not match {combined_manifold_output.shape}"
    assert combined_manifold_output_masked.shape == torch.Size([2, 4, 19, 19]), f"Combined Manifold Masked Shape does not match {combined_manifold_output_masked.shape}"

    assert not torch.any(torch.isnan(combined_manifold_output)), "Combined Manifold contains NaN values"
    assert not torch.any(torch.isnan(combined_manifold_output_masked)), "Combined Manifold Masked contains NaN values"

    assert len(mask_idxes) == 2, f"Did not correctly make batch indices: {len(mask_idxes)}"
    for mask_idx in mask_idxes:
        assert torch.sum(mask_idx) == 2

    
def testMENDRPreTrainerTinyMAEReconLoss():
    args = SimpleNamespace(
        learning_rate = 0.001,
        l2_weight_decay = 0.001,
        save_model_directory = None,
        mask_span = 5,
        temp = 0.01,
        training_params = SimpleNamespace(
            mask_ratio = 0.5,
            gradient_clip_value = 1e7,
            scheduler_after_batch = False,
            ckpt_dir = None
        ),
        negatives_loo = 10,
        enc_feat_l2 = 0.001,
        multi_gpu = False,
        ckpt_dir="./checkpoint",
        random_state=42,
        T_max=10,
        eta_min=0.001,
    )

    autoencoder = MENDRPatchEncoder(num_channels=19,
                            sampling_rate=128,
                            super_patch_seconds=5,
                            hidden_gnn_mlp_ratio=1,
                            n_gnn_transformer_layers=1,
                            num_subjects=1,
                            n_gnn_heads=2,
                            device=device,
                            )
    contextualizer = MENDRContextualizerTiny(num_channels=19, out_dim=24).to(device)
    optim_params = list(autoencoder.parameters()) + list(contextualizer.parameters())
    optimizer = torch.optim.AdamW(optim_params,
                betas=(0.9, 0.99),
                lr=args.learning_rate,
                weight_decay=args.l2_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                T_max=args.T_max,
                eta_min=args.eta_min)
    mix_optimizer = MixOptimizer(optimizer, scheduler)
    trainer = MENDRTinyPreTrainer(autoencoder, contextualizer, mix_optimizer, cfg=args, cuda=device)

    with torch.autograd.detect_anomaly():
        example_input = {
                'delta': torch.randn(4, 10, 19, 8*24).to(device).float(),
                'theta': torch.randn(4, 10, 19, 8*24).to(device).float(),
                'alpha': torch.randn(4, 10, 19, 16*24).to(device).float(),
                'beta':  torch.randn(4, 10, 19, 32*24).to(device).float(),
                'gamma': torch.randn(4, 10, 19, 64*24).to(device).float()
        }

        riemannian_loss, combined_manifold_output, combined_manifold_output_masked, mask_idxes = trainer._epochMaskedRecon(example_input, nn.MSELoss())

        assert riemannian_loss > 0, f"Loss is not greater than 0: {riemannian_loss}"

        assert combined_manifold_output.shape == torch.Size([4, 10, 19, 19]), f"Combined Manifold Shape does not match {combined_manifold_output.shape}"
        assert combined_manifold_output_masked.shape == torch.Size([4, 10, 19, 19]), f"Combined Manifold Masked Shape does not match {combined_manifold_output_masked.shape}"

        assert not torch.any(torch.isnan(combined_manifold_output)), "Combined Manifold contains NaN values"
        assert not torch.any(torch.isnan(combined_manifold_output_masked)), "Combined Manifold Masked contains NaN values"

        assert len(mask_idxes) == 4, f"Did not correctly make batch indices: {len(mask_idxes)}"
        for mask_idx in mask_idxes:
            assert torch.sum(mask_idx) == 5

def testMENDRPreTrainerWithTiny():
    args = SimpleNamespace(
        learning_rate = 0.001,
        l2_weight_decay = 0.001,
        save_model_directory = None,
        mask_span = 5,
        temp = 0.01,
        training_params = SimpleNamespace(
            mask_ratio = 0.5,
            gradient_clip_value = 1e7,
            scheduler_after_batch = False,
            ckpt_dir = None,
            num_workers = 32,
            batch_size = 16,
            epochs=1,
        ),
        meta_params = SimpleNamespace(
            experiment_name = "test",
            run_name = "test",
            log_model_params_and_grads = False,
            save_model = False,
            save_final_model = False,
            log_system_metrics = False,
        ),
        negatives_loo = 10,
        enc_feat_l2 = 0.001,
        multi_gpu = False,
        ckpt_dir="./checkpoint",
        random_state=42,
        T_max=10,
        eta_min=0.001,
    )

    autoencoder = MENDRPatchEncoder(num_channels=19,
                            sampling_rate=128,
                            super_patch_seconds=2,
                            hidden_gnn_mlp_ratio=1,
                            n_gnn_transformer_layers=1,
                            num_subjects=1,
                            n_gnn_heads=2,
                            device=device,
                            )
    contextualizer = MENDRContextualizerTiny(num_channels=19, out_dim=24).to(device)
    optim_params = list(autoencoder.parameters()) + list(contextualizer.parameters())
    optimizer = torch.optim.AdamW(optim_params,
                betas=(0.9, 0.99),
                lr=args.learning_rate,
                weight_decay=args.l2_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                T_max=args.T_max,
                eta_min=args.eta_min)
    mix_optimizer = MixOptimizer(optimizer, scheduler)
    trainer = MENDRTinyPreTrainer(autoencoder, contextualizer, mix_optimizer, cfg=args, cuda=device)
    training_dataset = WaveletPretrainDataset(root="/storage/ice1/shared/bmed6780/mip_group_6/ef/TUH-128Hz", frac=0.001)
    val_dataset = WaveletPretrainDataset(root="/storage/ice1/shared/bmed6780/mip_group_6/ef/TUH-128Hz", frac=0.001)

    trainer.fit(training_dataset=training_dataset, cfg=args, validation_dataset=val_dataset)

    autoencoder.apply(check_sanity)
    contextualizer.apply(check_sanity)

def testMENDRPreTrainerNoValidation():
    args = SimpleNamespace(
    encoder_grad_frac = 0.5,
    learning_rate = 0.001,
    l2_weight_decay = 0.001,
    save_model_directory = None,
    mask_ratio = 0.01,
    delta_reconstructive_loss_pref = 1.0,
    theta_reconstructive_loss_pref = 1.0,
    alpha_reconstructive_loss_pref = 1.0,
    beta_reconstructive_loss_pref = 1.0,
    gamma_reconstructive_loss_pref = 1.0,
    contrastive_combined_loss_pref = 1e3,
    contrastive_wavelet_loss_pref = 1e3,
    gradient_clip_value = 1e7,
    mask_span = 5,
    temp = 0.01,
    negatives_loo=10,
    enc_feat_l2 = 0.001,
    multi_gpu = False,
    ckpt_dir="./checkpoint",
    random_state=42
    )

    mendr = MENDR_model(device, temp=args.temp)
    trainer = MENDRPreTrainer(mendr, args)
    optimizer = torch.optim.Adam(trainer.parameters())
    optimizer = MixOptimizer(optimizer)
    trainer.set_optimizer(optimizer)
    dataset = WaveletPretrainDataset(root="/storage/ice1/shared/bmed6780/mip_group_6/ef/TUH-128Hz", frac=0.001)
    print(f"Total Number of Parameters: {sum(p.numel() for p in mendr.parameters() if p.requires_grad)}")
    trainer.fit(training_dataset=dataset, epochs=1, batch_size=32)

    mendr.mendr_encoder.apply(check_sanity)
    mendr.mendr_contextualizer.apply(check_sanity)

def testMENDRParameters():
    args = SimpleNamespace(
    encoder_grad_frac = 0.5,
    learning_rate = 0.001,
    l2_weight_decay = 0.001,
    save_model_directory = None,
    temp = 0.01,
    mask_ratio = 0.5,
    delta_reconstructive_loss_pref = 1.0,
    theta_reconstructive_loss_pref = 1.0,
    alpha_reconstructive_loss_pref = 1.0,
    beta_reconstructive_loss_pref = 1.0,
    gamma_reconstructive_loss_pref = 1.0,
    contrastive_combined_loss_pref = 1e3,
    contrastive_wavelet_loss_pref = 1e3,
    gradient_clip_value = 1e7,
    negatives_loo=10,
    enc_feat_l2 = 0.001,
    multi_gpu = False,
    ckpt_dir="./checkpoint",
    random_state=42
    )

    mendr = MENDR_model(device, temp=args.temp)
    trainer = MENDRPreTrainer(mendr, args)
    optimizer = torch.optim.Adam(trainer.parameters())
    optimizer = MixOptimizer(optimizer)
    trainer.set_optimizer(optimizer)
    dataset = WaveletPretrainDataset(root="/storage/ice1/shared/bmed6780/mip_group_6/ef/TUH-128Hz", frac=0.002)

    wavelet_spd_transform_params = dict()

    def checkOrthogonal(x):
        # Returns True if not orthogonal
        # False if orthogonal
        product = np.dot(x, x.T)
        np.fill_diagonal(product,0)
        return (product.any() == 0)

    for band, spd_transform in mendr.mendr_contextualizer.WaveletContextualizer.pre_attention_spd_transform.items():
        spd_weight = spd_transform.weight.data.clone().detach().cpu().numpy()
        assert not checkOrthogonal(spd_weight), f"Wavelet Contextualizer SPD weights not orthogonal: {spd_weight}"
        wavelet_spd_transform_params[band] = spd_weight

    spd_transform1_weight = dict()
    spd_transform2_weight = dict()
    for band, spd_transform in mendr.mendr_contextualizer.WaveletContextualizer.wavelet_spd_transforms.items():
        spd_weight = mendr.mendr_contextualizer.WaveletContextualizer.wavelet_spd_transforms[band][0].weight.data.clone().detach().cpu().numpy()
        assert not checkOrthogonal(spd_weight), f"Wavelet Contextualizer SPD weights not orthogonal: {spd_weight}"
        spd_transform1_weight[band] = spd_weight
        spd_weight = mendr.mendr_contextualizer.WaveletContextualizer.wavelet_spd_transforms[band][2].weight.data.clone().detach().cpu().numpy()
        assert not checkOrthogonal(spd_weight), f"Wavelet Contextualizer SPD weights not orthogonal: {spd_weight}"
        spd_transform2_weight[band] = spd_weight

    spd_weight = mendr.mendr_contextualizer.CombinedContextualizer.combined_spd_transform[0].weight.data.clone().detach().cpu().numpy()
    assert not checkOrthogonal(spd_weight), f"Combined Contextualizer SPD 1 weights not orthogonal: {spd_weight}"
    combined_spd_transform1_params = spd_weight

    spd_weight = mendr.mendr_contextualizer.CombinedContextualizer.combined_spd_transform[2].weight.data.clone().detach().cpu().numpy()
    assert not checkOrthogonal(spd_weight), f"Combined Contextualizer SPD 2 weights not orthogonal: {spd_weight}"
    combined_spd_transform2_params = spd_weight

    with torch.autograd.detect_anomaly():
        trainer.fit(training_dataset=dataset, epochs=1, batch_size=32)
    
    for band, spd_transform in mendr.mendr_contextualizer.WaveletContextualizer.pre_attention_spd_transform.items():
        if band != "high":
            new_weight = spd_transform.weight.data.cpu().numpy()
            assert not np.allclose(wavelet_spd_transform_params[band], new_weight), f"Wavelet Contextualizer SPD weights not updated: {band}: {wavelet_spd_transform_params[band]} == {new_weight} "
            assert not checkOrthogonal(new_weight), f"New Wavelet Contextualizer SPD weights not orthogonal: {band}"

    for band, spd_transform in mendr.mendr_contextualizer.WaveletContextualizer.wavelet_spd_transforms.items():
        if band != "high":
            new_weight = spd_transform[0].weight.data.cpu().numpy()
            assert not np.allclose(spd_transform1_weight[band], new_weight), f"Wavelet Contextualizer SPD 1 weights not updated: {band}: {spd_transform1_weight[band]} == {new_weight} "
            assert not checkOrthogonal(new_weight), f"New Wavelet Contextualizer SPD 1 not orthogonal: {band}"
            new_weight = spd_transform[2].weight.data.cpu().numpy()
            assert not np.allclose(spd_transform2_weight[band], new_weight), f"Wavelet Contextualizer SPD 2 weights not updated: {band}: {spd_transform2_weight[band]} == {new_weight} "
            assert not checkOrthogonal(new_weight), f"New Wavelet Contextualizer SPD 2 not orthogonal: {band}" 
    
    new_weight = mendr.mendr_contextualizer.CombinedContextualizer.combined_spd_transform[0].weight.data.cpu().numpy()
    assert not np.allclose(combined_spd_transform1_params, new_weight), f"Combined Contextualizer SPD 1 weights not updated: {combined_spd_transform1_params} == {new_weight}"
    assert not checkOrthogonal(new_weight), f"New Combined Contedxtualizer SPD 1 not orthogonal"

    new_weight = mendr.mendr_contextualizer.CombinedContextualizer.combined_spd_transform[2].weight.data.cpu().numpy()
    assert not np.allclose(combined_spd_transform2_params, new_weight), f"Combined Contextualizer SPD 2 weights not updated: {combined_spd_transform2_params} == {new_weight}"
    assert not checkOrthogonal(new_weight), f"New Combined Contedxtualizer SPD 2 not orthogonal"

    mendr.mendr_encoder.apply(check_sanity)
    mendr.mendr_contextualizer.apply(check_sanity)

def testMENDRPreTrainerWithValidation():
    args = SimpleNamespace(
    learning_rate = 0.001,
    l2_weight_decay = 0.001,
    save_model_directory = None,
    temp = 0.1,
    negatives_loo=20,
    enc_feat_l2 = 0.001,
    mask_ratio = 0.5,
    delta_reconstructive_loss_pref = 1.0,
    theta_reconstructive_loss_pref = 1.0,
    alpha_reconstructive_loss_pref = 1.0,
    beta_reconstructive_loss_pref = 1.0,        
    gamma_reconstructive_loss_pref = 1.0,
    contrastive_combined_loss_pref = 1e2,
    contrastive_wavelet_loss_pref = 5e3,
    gradient_clip_value = 1e10,
    multi_gpu = False,
    train_frac=0.8,
    val_frac=0.2,
    ckpt_dir="./checkpoint",
    random_state=42
    )

    mendr = MENDR_model(device, temp=args.temp)
    trainer = MENDRPreTrainer(mendr, args)
    optimizer = torch.optim.Adam(trainer.parameters())
    optimizer = MixOptimizer(optimizer)
    trainer.set_optimizer(optimizer)
    dataset = WaveletPretrainDataset(root="/storage/ice1/shared/bmed6780/mip_group_6/ef/TUH-128Hz", frac=0.01)
    print(f"Total Number of Parameters: {sum(p.numel() for p in mendr.parameters() if p.requires_grad)}")
    num_train = int(len(dataset) * (args.train_frac / (args.train_frac + args.val_frac)))
    num_val = len(dataset) - num_train
    train_dataset, val_dataset = torchdata.random_split(dataset, [num_train, num_val])
    print("Train and Validation Dataset Length: ", len(train_dataset), len(val_dataset))
    trainer.fit(training_dataset=train_dataset, validation_dataset=val_dataset, epochs=2, batch_size=64)

    mendr.mendr_encoder.apply(check_sanity)
    mendr.mendr_contextualizer.apply(check_sanity)

    assert len(os.listdir("./checkpoint")), "Model Checkpoint Directory is empty"

def testMENDRPretrainerTinyContextualizerWithValidation():
    args = SimpleNamespace(
    encoder_grad_frac = 0.5,
    learning_rate = 0.001,
    l2_weight_decay = 0.001,
    save_model_directory = None,
    mask_ratio = 0.5,
    delta_reconstructive_loss_pref = 1.0,
    theta_reconstructive_loss_pref = 1.0,
    alpha_reconstructive_loss_pref = 1.0,
    beta_reconstructive_loss_pref = 1.0,
    gamma_reconstructive_loss_pref = 1.0,
    contrastive_combined_loss_pref = 1e3,
    contrastive_wavelet_loss_pref = 1e3,
    gradient_clip_value = 1e7,
    temp = 0.01,
    negatives_loo=10,
    enc_feat_l2 = 0.001,
    multi_gpu = False,
    train_frac=0.8,
    val_frac=0.2,
    ckpt_dir="./checkpoint",
    random_state=42
    )

    mendr = MENDR_model(device, contextualizer_size="TINY", temp=args.temp)
    trainer = MENDRPreTrainer(mendr, args)
    optimizer = torch.optim.Adam(trainer.parameters())
    optimizer = MixOptimizer(optimizer)
    trainer.set_optimizer(optimizer)
    dataset = WaveletPretrainDataset(root="/storage/ice1/shared/bmed6780/mip_group_6/ef/TUH-128Hz", frac=0.001)
    print(f"Total Number of Parameters: {sum(p.numel() for p in mendr.parameters() if p.requires_grad)}")
    num_train = int(len(dataset) * (args.train_frac / (args.train_frac + args.val_frac)))
    num_val = len(dataset) - num_train
    train_dataset, val_dataset = torchdata.random_split(dataset, [num_train, num_val])
    print("Train and Validation Dataset Length: ", len(train_dataset), len(val_dataset))
    trainer.fit(training_dataset=train_dataset, validation_dataset=val_dataset, epochs=1, batch_size=32)

    mendr.mendr_encoder.apply(check_sanity)
    mendr.mendr_contextualizer.apply(check_sanity)
    assert len(os.listdir("./checkpoint")), "Model Checkpoint Directory is empty"


def testMENDRPreTrainerLoadFromCheckpoint():
    args = SimpleNamespace(
    encoder_grad_frac = 0.5,
    learning_rate = 0.001,
    l2_weight_decay = 0.001,
    save_model_directory = None,
    temp = 0.01,
    negatives_loo=10,
    enc_feat_l2 = 0.001,
    mask_ratio = 0.5,
    delta_reconstructive_loss_pref = 1.0,
    theta_reconstructive_loss_pref = 1.0,
    alpha_reconstructive_loss_pref = 1.0,
    beta_reconstructive_loss_pref = 1.0,
    gamma_reconstructive_loss_pref = 1.0,
    contrastive_combined_loss_pref = 1e3,
    contrastive_wavelet_loss_pref = 1e3,
    gradient_clip_value = 1e7,
    contrastive_loss_pref = 1e3,
    multi_gpu = False,
    train_frac=0.8,
    val_frac=0.2,
    ckpt_dir="./checkpoint",
    random_state=42,
    load_from_ckpt="./checkpoint/MockCkptLarge"
    )

    mendr = MENDR_model(device, temp=args.temp)
    trainer = MENDRPreTrainer(mendr, args)
    optimizer = torch.optim.Adam(trainer.parameters())
    optimizer = MixOptimizer(optimizer)
    trainer.set_optimizer(optimizer)
    trainer.load_from_ckpt(args.load_from_ckpt)

    mendr.mendr_encoder.apply(check_sanity)
    mendr.mendr_contextualizer.apply(check_sanity)

def testMENDRPreTrainerLoadFromCheckpointTiny():
    args = SimpleNamespace(
    encoder_grad_frac = 0.5,
    learning_rate = 0.001,
    l2_weight_decay = 0.001,
    save_model_directory = None,
    temp = 0.01,
    negatives_loo=10,
    enc_feat_l2 = 0.001,
    multi_gpu = False,
    train_frac=0.8,
    val_frac=0.2,
    mask_ratio = 0.5,
    contrastive_combined_loss_pref = 1e3,
    contrastive_wavelet_loss_pref = 1e3,
    delta_reconstructive_loss_pref = 1.0,
    theta_reconstructive_loss_pref = 1.0,
    alpha_reconstructive_loss_pref = 1.0,
    beta_reconstructive_loss_pref = 1.0,    
    gamma_reconstructive_loss_pref = 1.0,
    gradient_clip_value = 1e7,
    ckpt_dir="./checkpoint",
    random_state=42,
    load_from_ckpt="./checkpoint/MockCkptTiny"
    )

    mendr = MENDR_model(device, contextualizer_size="TINY", temp=args.temp)
    trainer = MENDRPreTrainer(mendr, args)
    optimizer = torch.optim.Adam(trainer.parameters())
    optimizer = MixOptimizer(optimizer)
    trainer.set_optimizer(optimizer)
    trainer.load_from_ckpt(args.load_from_ckpt)

    mendr.mendr_encoder.apply(check_sanity)
    mendr.mendr_contextualizer.apply(check_sanity)

if __name__ == "__main__":
    ### Seed ###
    torch.cuda.empty_cache()
    random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    ## CUDNN ##
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("Testing MENDR Super patching...")
    testMENDRSuperPatching()
    print("MENDR Super Patching Test Passed!")

    print("Testing Dropout1dWithIndexTracking...")
    testDropout1dWithIndexTracking()
    print("Dropout1dWithIndexTracking Test Passed!")

    print("Testing MENDR Make Mask Idxes...")
    testMakeMaskIdxes()
    print("MENDR Make Mask Idxes Test Passed!")

    print("Testing Encoder...")
    testEncoder()
    print("Encoder test passed!")

    print("Testing Batchwise Matrix Similarity...")
    testMENDRBatchWiseMatrixSimilarity()
    print("Batchwise Matrix Similarity test passed!")

    print("Testing Large Contextualizer Batch LEM...")
    testLargeContextualizerBatchLEM()
    print("Contextualizer Wavelet Batch test passed!")

    print("Testing Large Contextualizer Wavelet LEM...")
    testWaveletLEM()
    print("Contextualizer Wavelet LEM test passed!")

    print("Testing Tiny Contextualizer...")
    testContextualizerTiny()
    print("Tiny Contextualizer test passed!")

    '''
    print("Testing Large Contextualizer...")
    testContextualizerLarge()
    print("Large Contextualizer test passed!")

    print("Testing Large Contextualizer masking...")
    testMENDRLargeCombinedContextualizerMasking()
    print("Large Contextualizer masking test passed!")
    print("Testing pretrainer LOO contrastive loss...")
    testMENDRPreTrainerLOOLoss()
    print("PreTrainer LOO contrastive loss test passed! ")

    print("Testing pretrainer MAE Recon loss...")
    testMENDRPreTrainerMAEReconLoss()
    print("PreTrainer MAE Recon loss test passed! ")
    '''

    print("Testing pretrainer Tiny MAE Recon loss...")
    testMENDRPreTrainerTinyMAEReconLoss()
    print("PreTrainer Tiny MAE Recon loss test passed! ")

    print("Testing pretrainer with tiny contextualizer...")
    testMENDRPreTrainerWithTiny()
    print("PreTrainer with tiny contextualizer test passed!")

    '''
    print("Testing pretrainer fit without validation...")
    testMENDRPreTrainerNoValidation()
    print("PreTrainer fit without validation test passed!")

    print("Testing MENDR Parameters...")
    testMENDRParameters()
    print("Testing MENDR Parameters passed!")


    print("Testing pretrainer fit with validation...")
    testMENDRPreTrainerWithValidation()
    print("PreTrainer fit with validation test passed!")


    print("Testing pretrainer fit with tiny contextualizer with validation...")
    testMENDRPretrainerTinyContextualizerWithValidation()
    print("PreTrainer fit with tiny contextualizer with validation test passed!")

    print("Testing pretrainer load from checkpoint...")
    testMENDRPreTrainerLoadFromCheckpoint()
    print("PreTrainer load from checkpoint test passed!")

    print("Testing pretrainer load from checkpoint tiny...")
    testMENDRPreTrainerLoadFromCheckpointTiny()
    print("PreTrainer load from checkpoint tiny test passed!")
    '''

    print("All tests passed! Make sure to delete any artifacts generated during testing such as checkpoints.")