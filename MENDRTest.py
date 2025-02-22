import torch
import numpy as np
import random, os
from torch.utils.data import ConcatDataset
from torch_geometric.data import Data
from Model.MENDR.MENDREncoder import MENDRAutoEncoder, WaveletEncoderDecoder
from Model.MENDR.MENDRContextualizer import MENDRContextualizer
from Model.MENDR.MENDRTrainer import MENDRTrainer
from Model.MENDR.mAtt.optimizer import MixOptimizer
from Model.transforms import RandomTemporalCrop
from dataset import WaveletDataset
from types import SimpleNamespace
import torch.utils.data as torchdata
import torch.nn as nn

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

def testEncoder():
    example_input = {
            'delta': torch.randn(4, 19, 246).to(device).float(),
            'theta': torch.randn(4, 19, 246).to(device).float(),
            'alpha': torch.randn(4, 19, 486).to(device).float(),
            'beta': torch.randn(4, 19, 966).to(device).float(),
            'gamma': torch.randn(4, 19, 1925).to(device).float()
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
    encoder = MENDRAutoEncoder(device)

    output = encoder(example_graph, example_input)

    assert output.keys() == BANDS
    for band, (encoding, decoding) in output.items():
        assert decoding.shape == example_input[band].shape

    assert output['delta'][0].shape == torch.Size([4, 19, 124]), f"Actual Shape: {output['delta'][0].shape}" 
    assert output['theta'][0].shape == torch.Size([4, 19, 124]), f"Actual Shape: {output['theta'][0].shape}" 
    assert output['alpha'][0].shape == torch.Size([4, 19, 244]), f"Actual Shape: {output['alpha'][0].shape}" 
    assert output['beta'][0].shape == torch.Size([4, 19, 484]), f"Actual Shape: {output['beta'][0].shape}" 
    assert output['gamma'][0].shape == torch.Size([4, 19, 484]), f"Actual Shape: {output['gamma'][0].shape}" 

def testContextualizer():
    example_input = {
            'delta': (torch.randn(8, 19, 124).to(device), None),
            'theta': (torch.randn(8, 19, 124).to(device), None),
            'alpha': (torch.randn(8, 19, 244).to(device), None),
            'beta': (torch.randn(8, 19, 484).to(device), None),
            'gamma': (torch.randn(8, 19, 484).to(device), None)
        }

    contextualizer = MENDRContextualizer(device)
    combined_r2e_output, combined_manifold_output, wavelet_manifold_output = contextualizer(example_input)

    for band, v in example_input.items():
        output = wavelet_manifold_output[band]
        assert output.shape == torch.Size([32, 19, 19]), f'{band} Wavelet Manifold Shape: {wavelet_manifold_output[band].shape}'


    assert combined_r2e_output.shape == torch.Size([8, 760]), f'Combined R2E Shape: {combined_r2e_output.shape}'
    assert combined_manifold_output.shape == torch.Size([32, 19, 19]), f'Combined Manifold Shape: {combined_manifold_output.shape}'
    
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

    encoder = MENDRAutoEncoder(device=device)
    contextualizer = MENDRContextualizer(device=device)
    trainer = MENDRTrainer(encoder, contextualizer, args)
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

    encoder = MENDRAutoEncoder(device=device)
    contextualizer = MENDRContextualizer(device=device)
    trainer = MENDRTrainer(encoder, contextualizer, args)
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

    encoder = MENDRAutoEncoder(device=device)
    contextualizer = MENDRContextualizer(device=device)
    trainer = MENDRTrainer(encoder, contextualizer, args)
    optimizer = torch.optim.Adam(trainer.parameters())
    optimizer = MixOptimizer(optimizer)
    trainer.set_optimizer(optimizer)
    dataset = WaveletDataset(root="/home/hice1/mchen439/scratch/eegfoundationmodeldata", frac=0.0005)
    with torch.autograd.detect_anomaly():
        trainer.fit(training_dataset=dataset, epochs=1, batch_size=16)

    encoder.apply(check_sanity)
    contextualizer.apply(check_sanity)

def testMENDRTrainerWithValidation():
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
    train_frac=0.8,
    val_frac=0.2,
    ckpt_dir="./checkpoint",
    random_state=42
    )

    encoder = MENDRAutoEncoder(device=device)
    contextualizer = MENDRContextualizer(device=device)
    trainer = MENDRTrainer(encoder, contextualizer, args)
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
    mask_rate = 0.01,
    mask_span = 5,
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

    encoder = MENDRAutoEncoder(device=device)
    contextualizer = MENDRContextualizer(device=device)
    trainer = MENDRTrainer(encoder, contextualizer, args)
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

    print("Testing Encoder...")
    testEncoder()
    print("Encoder test passed!")

    print("Testing Contextualizer...")
    testContextualizer()
    print("Contextualizer test passed!")

    print("Testing trainer LOO contrastive loss")
    testMENDRTrainerLOOLoss()
    print("Trainer LOO contrastive loss test passed! ")

    print("Testing trainer MAE Recon loss")
    testMENDRTrainerMAEReconLoss()
    print("Trainer MAE Recon loss test passed! ")

    print("Testing trainer fit without validation...")
    testMENDRTrainerNoValidation()
    print("Trainer fit without validation test passed!")

    print("Testing trainer fit with validation...")
    testMENDRTrainerWithValidation()
    print("Trainer fit with validation test passed!")

    print("Testing trainer load from checkpoint...")
    testMENDRLoadFromCheckpoint()
    print("Trainer load from checkpoint test passed!")
    print("All tests passed! Make sure to delete any artifacts generated during testing such as checkpoints.")
