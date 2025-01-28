import torch
from Model.MENDR.MENDRContextualizer import MENDRContextualizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def testContextualizer():
    example_input = {
            'delta': (torch.randn(4, 19, 120).to(device), None),
            'theta': (torch.randn(4, 19, 120).to(device), None),
            'alpha': (torch.randn(4, 19, 240).to(device), None),
            'beta': (torch.randn(4, 19, 480).to(device), None),
            'gamma': (torch.randn(4, 19, 960).to(device), None)
        }
    contextualizer = MENDRContextualizer(device)
    output, shape = contextualizer(example_input)

    assert output.shape == torch.Size([8, 64, 64])
    assert shape == [4, 2, -1]

if __name__ == "__main__":
    print("Testing Contextualizer...")
    testContextualizer()

    print("All tests passed!")
