import torch
from torch_geometric.data import Data
from Model.MENDR.MENDREncoder import MENDREncoder, WaveletEncoderDecoder
from Model.MENDR.MENDRContextualizer import MENDRContextualizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    assert output['delta'][0].shape == torch.Size([4, 19, 480]), f"Actual Shape: {output['delta'][0].shape}" 
    assert output['theta'][0].shape == torch.Size([4, 19, 480]), f"Actual Shape: {output['theta'][0].shape}" 
    assert output['alpha'][0].shape == torch.Size([4, 19, 960]), f"Actual Shape: {output['alpha'][0].shape}" 
    assert output['beta'][0].shape == torch.Size([4, 19, 1920]), f"Actual Shape: {output['beta'][0].shape}" 
    assert output['gamma'][0].shape == torch.Size([4, 19, 3840]), f"Actual Shape: {output['gamma'][0].shape}" 

def testContextualizer():
    example_input = {
            'delta': (torch.randn(4, 19, 480).to(device), None),
            'theta': (torch.randn(4, 19, 480).to(device), None),
            'alpha': (torch.randn(4, 19, 960).to(device), None),
            'beta': (torch.randn(4, 19, 1920).to(device), None),
            'gamma': (torch.randn(4, 19, 3840).to(device), None)
        }
    contextualizer = MENDRContextualizer(device)
    output, shape = contextualizer(example_input)

    assert output.shape == torch.Size([8, 64, 64])
    assert shape == [4, 2, -1]

if __name__ == "__main__":
    print("Testing Encoder...")
    testEncoder()

    print("Testing Contextualizer...")
    testContextualizer()

    print("All tests passed!")
