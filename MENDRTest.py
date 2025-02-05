import torch
from torch_geometric.data import Data
from Model.MENDR.MENDREncoder import MENDREncoder, WaveletEncoderDecoder
from Model.MENDR.MENDRContextualizer import MENDRContextualizer
from Model.MENDR.MENDRTrainer import MENDRTrainer
from Model.MENDR.mAtt.optimizer import MixOptimizer
from Model.transforms import RandomTemporalCrop
from dataset import WaveletDataset
from types import SimpleNamespace
import torch.utils.data as torchdata

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu' 
BANDS = {'delta', 'theta', 'alpha', 'beta', 'gamma'}

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
    encoder = MENDREncoder(device)

    output = encoder(example_graph, example_input)

    assert output.keys() == BANDS
    for band, (encoding, decoding) in output.items():
        assert decoding.shape == example_input[band].shape

    assert output['delta'][0].shape == torch.Size([4, 19, 120]), f"Actual Shape: {output['delta'][0].shape}" 
    assert output['theta'][0].shape == torch.Size([4, 19, 120]), f"Actual Shape: {output['theta'][0].shape}" 
    assert output['alpha'][0].shape == torch.Size([4, 19, 240]), f"Actual Shape: {output['alpha'][0].shape}" 
    assert output['beta'][0].shape == torch.Size([4, 19, 480]), f"Actual Shape: {output['beta'][0].shape}" 
    assert output['gamma'][0].shape == torch.Size([4, 19, 960]), f"Actual Shape: {output['gamma'][0].shape}" 

def testContextualizer():
    example_input = {
            'delta': (torch.randn(8, 19, 480).to(device), None),
            'theta': (torch.randn(8, 19, 480).to(device), None),
            'alpha': (torch.randn(8, 19, 960).to(device), None),
            'beta': (torch.randn(8, 19, 1920).to(device), None),
            'gamma': (torch.randn(8, 19, 3840).to(device), None)
        }
    contextualizer = MENDRContextualizer(device)
    output, shape, wavelet_output = contextualizer(example_input)

    assert output.shape == torch.Size([32, 32, 32])
    assert shape == [8, 4, -1]

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
    multi_gpu = False
    )

    encoder = MENDREncoder(device=device)
    contextualizer = MENDRContextualizer(device=device)
    trainer = MENDRTrainer(encoder, contextualizer, args)
    optimizer = torch.optim.Adam(trainer.parameters())
    optimizer = MixOptimizer(optimizer)
    trainer.set_optimizer(optimizer)
    dataset = WaveletDataset(root="/home/hice1/mchen439/scratch/eegfoundationmodeldata", frac=0.0005)
    with torch.autograd.detect_anomaly():
        trainer.fit(training_dataset=dataset, epochs=3, batch_size=16) # As long as it runs it works lol

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
    val_frac=0.2
    )

    encoder = MENDREncoder(device=device)
    contextualizer = MENDRContextualizer(device=device)
    trainer = MENDRTrainer(encoder, contextualizer, args)
    optimizer = torch.optim.Adam(trainer.parameters())
    optimizer = MixOptimizer(optimizer)
    trainer.set_optimizer(optimizer)
    dataset = WaveletDataset(root="/home/hice1/mchen439/scratch/eegfoundationmodeldata", frac=0.001)
    num_train = int(len(dataset) * (args.train_frac / (args.train_frac + args.val_frac)))
    num_val = len(dataset) - num_train
    train_dataset, val_dataset = torchdata.random_split(dataset, [num_train, num_val])
    print("Train and Validation Dataset Length: ", len(train_dataset), len(val_dataset))
    with torch.autograd.detect_anomaly():
        torch.nn.utils.clip_grad_norm_(trainer.parameters(), max_norm=10, error_if_nonfinite=True)
        trainer.fit(training_dataset=train_dataset, validation_dataset=val_dataset, epochs=2, batch_size=32)


if __name__ == "__main__":
    print("Testing Encoder...")
    #testEncoder()
    print("Encoder test passed!")

    print("Testing Contextualizer...")
    #testContextualizer()
    print("Contextualizer test passed!")

    print("Testing trainer fit without validation...")
    #testMENDRTrainerNoValidation()
    print("Trainer fit without validation test passed!")

    print("Testing trainer fit with validation...")
    testMENDRTrainerWithValidation()
    print("Trainer fit with validation test passed!")

    print("All tests passed!")
