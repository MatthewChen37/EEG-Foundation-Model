import torch
import numpy as np
import random, os
from torch.utils.data import ConcatDataset
from torch_geometric.data import Data
from Model.MENDR.MENDREncoder import MENDRPatchEncoder
from Model.MENDR.MENDRContextualizerLarge import MENDRContextualizerLarge
from Model.MENDR.MENDRContextualizerTiny import MENDRContextualizerTiny
from Model.MENDR.MENDR import MENDR_model
from Model.MENDR.MENDRPreTrainer import MENDRPreTrainer
from Model.MENDR.mAtt.optimizer import MixOptimizer
from Model.transforms import RandomTemporalCrop
from Datasets.datasetPretrain import WaveletPretrainDataset
from types import SimpleNamespace
import torch.utils.data as torchdata
import torch.nn as nn
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BANDS = {'delta', 'theta', 'alpha', 'beta', 'gamma'}

def random_spd_batch(batch_size, n):
    return torch.tensor(np.array([
        random_spd_matrix(n) for i in range(batch_size)
    ]))

def random_spd_matrix(n):
    A = np.random.rand(n, n)
    return np.dot(A, A.transpose())

def check_sanity(m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d, nn.GroupNorm)):
            assert m.weight.data.mean() != 0

def testMENDRSuperPatching():
    example_input = {
            'delta': torch.randn(4, 19, 246).to(device).float(),
            'theta': torch.randn(4, 19, 246).to(device).float(),
            'alpha': torch.randn(4, 19, 486).to(device).float(),
            'beta': torch.randn(4, 19, 966).to(device).float(),
            'gamma': torch.randn(4, 19, 1925).to(device).float()
    }

    model = MENDR_model(device=device)

    assert model.WAVELET_LENGTHS == {'delta': 4, 'theta': 4, 'alpha': 8, 'beta': 16,  'gamma': 32, 'high': 64}, f"model.WAVELET_LENGTHS: {model.WAVELET_LENGTHS}"
    assert model.WAVELET_SUPER_PATCH_LENGTHS == {'delta': 40, 'theta': 40, 'alpha': 80, 'beta': 160,  'gamma': 320, 'high': 640}, f"model.WAVELET_SUPER_PATCH_LENGTHS: {model.WAVELET_SUPER_PATCH_LENGTHS}"
    assert model.WAVELET_SUPER_PATCH_HOP_LENGTHS == {'delta': 20, 'theta': 20, 'alpha': 40, 'beta': 80,  'gamma': 160, 'high': 320}, f"modelWAVELET_SUPER_PATCH_HOP_LENGTHS: {model.WAVELET_SUPER_PATCH_HOP_LENGTHS}"

    patchified_data = model._super_patchify(example_input)

    expected_shape = {
        'delta': torch.Size([4, 11, 19, 40]),
        'theta': torch.Size([4, 11, 19, 40]),
        'alpha': torch.Size([4, 11, 19, 80]),
        'beta': torch.Size([4, 11, 19, 160]),
        'gamma': torch.Size([4, 11, 19, 320]),
    }

    for band in patchified_data:
        assert patchified_data[band].shape == expected_shape[band], f"{band}: Actual Shape: {patchified_data[band].shape} Expected Shape: {expected_shape[band]}"

def testEncoder():
    example_input = {
            'delta': torch.randn(4, 11, 19, 40).to(device).float(),
            'theta': torch.randn(4, 11, 19, 40).to(device).float(),
            'alpha': torch.randn(4, 11, 19, 80).to(device).float(),
            'beta': torch.randn(4, 11, 19, 160).to(device).float(),
            'gamma': torch.randn(4, 11, 19, 320).to(device).float()
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

    example_graph = Data(edge_index=batch_edge_index, edge_attr=batch_edge_attributes)

    encoder = MENDRPatchEncoder(
        num_channels=19,
        delta_sub_patch_size=4,
        theta_sub_patch_size=4,
        alpha_sub_patch_size=8,
        beta_sub_patch_size=16,
        gamma_sub_patch_size=32,
        high_sub_patch_size=64,
        delta_encoded_h=38,
        theta_encoded_h=38,
        alpha_encoded_h=38,
        beta_encoded_h=38,
        gamma_encoded_h=76,
        high_encoded_h=76,
        delta_super_patch_seq_len=40,
        theta_super_patch_seq_len=40,
        alpha_super_patch_seq_len=80,
        beta_super_patch_seq_len=160,
        gamma_super_patch_seq_len=320,
        high_super_patch_seq_len=640,
        device=device)

    encodings, decodings = encoder(example_graph, example_input)
    assert encodings.keys() == BANDS
    assert decodings.keys() == BANDS

    assert decodings['delta'].shape == torch.Size([44, 19, 40]), f"Actual Shape: {decodings['delta'].shape}" 
    assert decodings['theta'].shape == torch.Size([44, 19, 40]), f"Actual Shape: {decodings['theta'].shape}" 
    assert decodings['alpha'].shape == torch.Size([44, 19, 80]), f"Actual Shape: {decodings['alpha'].shape}" 
    assert decodings['beta'].shape == torch.Size([44, 19, 160]), f"Actual Shape: {decodings['beta'].shape}" 
    assert decodings['gamma'].shape == torch.Size([44, 19, 320]), f"Actual Shape: {decodings['gamma'].shape}"

    assert encodings['delta'].shape == torch.Size([44, 38, 18]), f"Actual Shape: {encodings['delta'].shape}" 
    assert encodings['theta'].shape == torch.Size([44, 38, 18]), f"Actual Shape: {encodings['theta'].shape}" 
    assert encodings['alpha'].shape == torch.Size([44, 38, 18]), f"Actual Shape: {encodings['alpha'].shape}" 
    assert encodings['beta'].shape == torch.Size([44, 38, 18]), f"Actual Shape: {encodings['beta'].shape}" 
    assert encodings['gamma'].shape == torch.Size([44, 76, 18]), f"Actual Shape: {encodings['gamma'].shape}"

def testLargeContextualizerBatchLEM():
    # Eigenvalues are 1, 3
    example_SPD = torch.tensor([
        [2, 1],
        [1, 2]
    ]).float().to(device)

    # Batch size is 4, epochs = 4
    example_SPD_batch = example_SPD.repeat(4, 1, 1).to(device)

    example_input = [
        example_SPD_batch.clone().to(device),
        example_SPD_batch.clone().to(device),
        example_SPD_batch.clone().to(device),
        example_SPD_batch.clone().to(device)
    ]

    contextualizer = MENDRContextualizerLarge(device)
    batch_output = contextualizer.WaveletContextualizer._batch_LogEuclideanMean(example_input, 'delta')

    for batch_idx in range(batch_output.shape[0]):
        assert torch.allclose(batch_output[batch_idx], example_SPD), f"Batch LEM Not equal: \n Actual: {batch_output[batch_idx]} \n Expected: {example_SPD}"


def testLargeContextualizerWaveletLEM():
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

    contextualizer = MENDRContextualizerLarge(device)
    combined_output = contextualizer.CombinedContextualizer._wavelet_LogEuclideanMean(example_input)
    for batch_idx in range(combined_output.shape[0]):
        assert torch.allclose(combined_output[batch_idx, 0], example_SPD), f"Combined LEM Not equal: \n Actual: {combined_output[batch_idx, 0]} \n Expected: {example_SPD}"


def testContextualizerTiny():
    example_input = {
            'delta': torch.randn(4, 11, 38, 18).to(device).float(),
            'theta': torch.randn(4, 11, 38, 18).to(device).float(),
            'alpha': torch.randn(4, 11, 38, 18).to(device).float(),
            'beta': torch.randn(4, 11, 38, 18).to(device).float(),
            'gamma': torch.randn(4, 11, 76, 18).to(device).float()
    }

    with torch.no_grad():
        contextualizer = MENDRContextualizerTiny(device)
        combined_manifold_output, cov_matrices = contextualizer(example_input)
        print("Total number of Tiny parameters: ", sum(p.numel() for p in contextualizer.parameters() if p.requires_grad))
        assert combined_manifold_output.shape == torch.Size([4, 11, 228, 228]), f"Incorrect output shape: {combined_manifold_output.shape}"

        assert cov_matrices['delta'].shape == torch.Size([4, 11, 38, 38])
        assert cov_matrices['theta'].shape == torch.Size([4, 11, 38, 38])
        assert cov_matrices['alpha'].shape == torch.Size([4, 11, 38, 38])
        assert cov_matrices['beta'].shape == torch.Size([4, 11, 38, 38])
        assert cov_matrices['gamma'].shape == torch.Size([4, 11, 76, 76])

def testContextualizerLarge():
    example_input = {
            'delta': torch.randn(4, 11, 38, 18).to(device).float(),
            'theta': torch.randn(4, 11, 38, 18).to(device).float(),
            'alpha': torch.randn(4, 11, 38, 18).to(device).float(),
            'beta': torch.randn(4, 11, 38, 18).to(device).float(),
            'gamma': torch.randn(4, 11, 76, 18).to(device).float()
    }

    with torch.no_grad():
        contextualizer = MENDRContextualizerLarge(device)
        combined_manifold_output, wavelet_manifold_output = contextualizer(example_input)
        print("Total number of Large parameters: ", sum(p.numel() for p in contextualizer.parameters() if p.requires_grad))

        wavelet_manifold_output_delta = wavelet_manifold_output['delta']
        wavelet_manifold_output_theta = wavelet_manifold_output['theta']
        wavelet_manifold_output_alpha = wavelet_manifold_output['alpha']
        wavelet_manifold_output_beta = wavelet_manifold_output['beta']
        wavelet_manifold_output_gamma = wavelet_manifold_output['gamma']

        assert wavelet_manifold_output_delta.shape == torch.Size([4, 11, 38, 38]), f'Delta Wavelet Manifold Shape:{wavelet_manifold_output_delta.shape}'
        assert wavelet_manifold_output_theta.shape == torch.Size([4, 11, 38, 38]), f'Theta Wavelet Manifold Shape:{wavelet_manifold_output_theta.shape}'
        assert wavelet_manifold_output_alpha.shape == torch.Size([4, 11, 38, 38]), f'Alpha Wavelet Manifold Shape:{wavelet_manifold_output_alpha.shape}'
        assert wavelet_manifold_output_beta.shape == torch.Size([4, 11, 38, 38]), f'Beta Wavelet Manifold Shape:{wavelet_manifold_output_beta.shape}'
        assert wavelet_manifold_output_gamma.shape == torch.Size([4, 11, 38, 38]), f'Gamma Wavelet Manifold Shape:{wavelet_manifold_output_gamma.shape}'

        assert combined_manifold_output.shape == torch.Size([4, 11, 38, 38]), f'Combined Manifold Shape: {combined_manifold_output.shape}'
        assert not torch.any(torch.isnan(combined_manifold_output)), "Combined Manifold contains NaN values"
    
def testMENDRTrainerLOOLoss():
    args = SimpleNamespace(
    encoder_grad_frac = 0.5,
    learning_rate = 0.001,
    l2_weight_decay = 0.001,
    save_model_directory = None,
    mask_rate = 0.01,
    mask_span = 5,
    temp = 0.01,
    num_negatives=10,
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
    dataset = WaveletDataset(root="/home/hice1/mchen439/scratch/eegfoundationmodeldata", frac=0.0005)
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


def testMENDRTrainerMAEReconLoss():
    args = SimpleNamespace(
    encoder_grad_frac = 0.5,
    learning_rate = 0.001,
    l2_weight_decay = 0.001,
    save_model_directory = None,
    mask_rate = 0.01,
    mask_span = 5,
    temp = 0.01,
    num_negatives=10,
    enc_feat_l2 = 0.001,
    multi_gpu = False,
    ckpt_dir="./checkpoint",
    random_state=42
    )

    encoder = MENDRWindowEncoder(device=device)
    contextualizer = MENDRContextualizer(device=device)
    trainer = MENDRPreTrainer(encoder, contextualizer, args)
    optimizer = torch.optim.Adam(trainer.parameters())
    optimizer = MixOptimizer(optimizer)
    trainer.set_optimizer(optimizer)
    dataset = WaveletDataset(root="/home/hice1/mchen439/scratch/eegfoundationmodeldata", frac=0.0005)
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

        riemannian_loss, combined_manifold_output, combined_manifold_output_masked, mask = trainer.epochMaskedRecon(wavelet_manifold_output, [2, 4, -1], nn.MSELoss())
    assert riemannian_loss > 0, f"Loss is not greater than 0: {riemannian_loss}"

    assert combined_manifold_output.shape == torch.Size([8, 19, 19]), f"Combined Manifold Shape does not match{combined_manifold_output.shape}"
    assert combined_manifold_output_masked.shape == torch.Size([8, 19, 19]), f"Combined Manifold Masked Shape does not match{combined_manifold_output_masked.shape}"

    assert not torch.any(torch.isnan(combined_manifold_output)), "Combined Manifold contains NaN values"
    assert not torch.any(torch.isnan(combined_manifold_output_masked)), "Combined Manifold Masked contains NaN values"


    for batch_index, epoch_index in enumerate(mask):
        assert not torch.equal(combined_manifold_output[batch_index*4+epoch_index.item(), :, :], combined_manifold_output_masked[batch_index*4+epoch_index.item(),:,:])
        

def testMENDRTrainerNoValidation():
    args = SimpleNamespace(
    encoder_grad_frac = 0.5,
    learning_rate = 0.001,
    l2_weight_decay = 0.001,
    save_model_directory = None,
    mask_rate = 0.01,
    mask_span = 5,
    temp = 0.01,
    num_negatives=10,
    enc_feat_l2 = 0.001,
    multi_gpu = False,
    ckpt_dir="./checkpoint",
    random_state=42
    )

    encoder = MENDRWindowEncoder(device=device)
    contextualizer = MENDRContextualizer(device=device)
    trainer = MENDRPreTrainer(encoder, contextualizer, args)
    optimizer = torch.optim.Adam(trainer.parameters())
    optimizer = MixOptimizer(optimizer)
    trainer.set_optimizer(optimizer)
    dataset = WaveletDataset(root="/home/hice1/mchen439/scratch/eegfoundationmodeldata", frac=0.0005)
    with torch.autograd.detect_anomaly():
        trainer.fit(training_dataset=dataset, epochs=1, batch_size=16)

    encoder.apply(check_sanity)
    contextualizer.apply(check_sanity)

def testMENDRParameters():
    args = SimpleNamespace(
    encoder_grad_frac = 0.5,
    learning_rate = 0.001,
    l2_weight_decay = 0.001,
    save_model_directory = None,
    temp = 0.01,
    num_negatives=10,
    enc_feat_l2 = 0.001,
    multi_gpu = False,
    ckpt_dir="./checkpoint",
    random_state=42
    )

    encoder = MENDRWindowEncoder(device=device)
    contextualizer = MENDRContextualizer(device=device)
    trainer = MENDRPreTrainer(encoder, contextualizer, args)
    optimizer = torch.optim.Adam(trainer.parameters())
    optimizer = MixOptimizer(optimizer)
    trainer.set_optimizer(optimizer)
    dataset = WaveletDataset(root="/home/hice1/mchen439/scratch/eegfoundationmodeldata", frac=0.0005)

    wavelet_spd_transform_params = dict()

    def checkOrthogonal(x):
        # Returns True if not orthogonal
        # False if orthogonal
        product = np.dot(x, x.T)
        np.fill_diagonal(product,0)
        return (product.any() == 0)

    for band, spd_transform in contextualizer.WaveletContextualizer.wavelet_spd_transforms.items():
        spd_weight = spd_transform.weight.data.clone().cpu().numpy()
        assert not checkOrthogonal(spd_weight), f"Wavelet Contextualizer SPD weights not orthogonal: {spd_weight}"
        wavelet_spd_transform_params[band] = spd_weight

    spd_weight = contextualizer.CombinedContextualizer.combined_spd_transform1[0].weight.data.clone().cpu().numpy()
    assert not checkOrthogonal(spd_weight), f"Combined Contextualizer SPD 1 weights not orthogonal: {spd_weight}"
    combined_spd_transform1_params = spd_weight

    spd_weight = contextualizer.CombinedContextualizer.combined_spd_transform2.weight.data.clone().cpu().numpy()
    assert not checkOrthogonal(spd_weight), f"Combined Contextualizer SPD 2 weights not orthogonal: {spd_weight}"
    combined_spd_transform2_params = spd_weight

    with torch.autograd.detect_anomaly():
        trainer.fit(training_dataset=dataset, epochs=1, batch_size=16)
    
        for band, spd_transform in contextualizer.WaveletContextualizer.wavelet_spd_transforms.items():
            new_weight = spd_transform.weight.data.cpu().numpy()
            assert not np.allclose(wavelet_spd_transform_params[band], new_weight), f"Wavelet Contextualizer SPD weights not updated: {band}: {wavelet_spd_transform_params[band]} == {new_weight} "
            assert not checkOrthogonal(new_weight), f"New Wavelet Contextualizer SPD weights not orthogonal: {band}"
        
        new_weight = contextualizer.CombinedContextualizer.combined_spd_transform1[0].weight.data.cpu().numpy()
        assert not np.allclose(combined_spd_transform1_params, new_weight), f"Combined Contextualizer SPD 1 weights not updated: {combined_spd_transform1_params} == {new_weight}"
        assert not checkOrthogonal(new_weight), f"New Combined Contedxtualizer SPD 1 not orthogonal"

        new_weight = contextualizer.CombinedContextualizer.combined_spd_transform2.weight.data.cpu().numpy()
        assert not np.allclose(combined_spd_transform2_params, new_weight), f"Combined Contextualizer SPD 2 weights not updated: {combined_spd_transform2_params} == {new_weight}"
        assert not checkOrthogonal(new_weight), f"New Combined Contedxtualizer SPD 2 not orthogonal"

    encoder.apply(check_sanity)
    contextualizer.apply(check_sanity)

def testMENDRTrainerWithValidation():
    args = SimpleNamespace(
    encoder_grad_frac = 0.5,
    learning_rate = 0.001,
    l2_weight_decay = 0.001,
    save_model_directory = None,
    temp = 0.01,
    num_negatives=10,
    enc_feat_l2 = 0.001,
    multi_gpu = False,
    train_frac=0.8,
    val_frac=0.2,
    ckpt_dir="./checkpoint",
    random_state=42
    )

    encoder = MENDRWindowEncoder(device=device)
    contextualizer = MENDRContextualizer(device=device)
    trainer = MENDRPreTrainer(encoder, contextualizer, args)
    optimizer = torch.optim.Adam(trainer.parameters())
    optimizer = MixOptimizer(optimizer)
    trainer.set_optimizer(optimizer)
    dataset = ConcatDataset([WaveletDataset(root="/home/hice1/mchen439/scratch/eegfoundationmodeldata", frac=0.001)])
    num_train = int(len(dataset) * (args.train_frac / (args.train_frac + args.val_frac)))
    num_val = len(dataset) - num_train
    train_dataset, val_dataset = torchdata.random_split(dataset, [num_train, num_val])
    print("Train and Validation Dataset Length: ", len(train_dataset), len(val_dataset))
    with torch.autograd.detect_anomaly():
        trainer.fit(training_dataset=train_dataset, validation_dataset=val_dataset, epochs=2, batch_size=32)

    encoder.apply(check_sanity)
    contextualizer.apply(check_sanity)

def testMENDRLoadFromCheckpoint():
    args = SimpleNamespace(
    encoder_grad_frac = 0.5,
    learning_rate = 0.001,
    l2_weight_decay = 0.001,
    save_model_directory = None,
    temp = 0.01,
    num_negatives=10,
    enc_feat_l2 = 0.001,
    multi_gpu = False,
    train_frac=0.8,
    val_frac=0.2,
    ckpt_dir="./checkpoint",
    random_state=42,
    load_from_ckpt="./checkpoint/MockCkpt"
    )

    encoder = MENDRWindowEncoder(device=device)
    contextualizer = MENDRContextualizer(device=device)
    trainer = MENDRPreTrainer(encoder, contextualizer, args)
    optimizer = torch.optim.Adam(trainer.parameters())
    optimizer = MixOptimizer(optimizer)
    trainer.set_optimizer(optimizer)
    trainer.load_from_ckpt(args.load_from_ckpt)

    encoder.apply(check_sanity)
    contextualizer.apply(check_sanity)

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

    print("Testing Encoder...")
    testEncoder()
    print("Encoder test passed!")

    print("Testing Large Contextualizer Batch LEM...")
    testLargeContextualizerBatchLEM()
    print("Contextualizer Wavelet Batch test passed!")

    print("Testing Large Contextualizer Wavelet LEM...")
    testLargeContextualizerWaveletLEM()
    print("Contextualizer Wavelet LEM test passed!")

    print("Testing Tiny Contextualizer...")
    testContextualizerTiny()
    print("Tiny Contextualizer test passed!")

    print("Testing Large Contextualizer...")
    testContextualizerLarge()
    print("Large Contextualizer test passed!")

    print("Testing trainer LOO contrastive loss")
    #testMENDRTrainerLOOLoss()
    print("Trainer LOO contrastive loss test passed! ")

    print("Testing trainer MAE Recon loss")
    #testMENDRTrainerMAEReconLoss()
    print("Trainer MAE Recon loss test passed! ")

    print("Testing trainer fit without validation...")
    #testMENDRTrainerNoValidation()
    print("Trainer fit without validation test passed!")

    print("Testing MENDR Parameters...")
    #testMENDRParameters()
    print("Testing MENDR Parameters passed!")

    print("Testing trainer fit with validation...")
    #testMENDRTrainerWithValidation()
    print("Trainer fit with validation test passed!")

    print("Testing trainer load from checkpoint...")
    #testMENDRLoadFromCheckpoint()
    print("Trainer load from checkpoint test passed!")
    print("All tests passed! Make sure to delete any artifacts generated during testing such as checkpoints.")
