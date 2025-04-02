import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plotReconstruction(inputs, decodings, title):
    assert len(inputs) == len(decodings), f"Input and Decoding Lengths do not match: {len(inputs)} {len(decodings)}"
    fig, axs = plt.subplots(inputs['delta'].shape[0], 5, figsize=(30, 15))
    for batch_idx in range(inputs['delta'].shape[0]):
        for idx, band in enumerate(['delta', 'theta', 'alpha', 'beta', 'gamma']):
            for channel in range(19):
                axs[batch_idx, idx].plot(inputs[band][batch_idx, channel, :], label=f"Input {band} Channel {channel}", c="g")
                axs[batch_idx, idx].plot(decodings[band][batch_idx, channel, :], label=f"Output {band} Channel {channel}", c="r")
            axs[batch_idx, idx].axis('off')
            
    for ax, col in zip(axs[0], ['delta', 'theta', 'alpha', 'beta', 'gamma']):
        ax.set_title(f"{title} Band: {col}")
        ax.set_xlabel("Time")
    
    for ax, row, in zip(axs[:, 0], [f'Patch {i + 1}' for i in range(inputs['delta'].shape[0])]):
        ax.set_ylabel(f"Amplitude (mV) Patch: {row}")

    input_handle = Patch(color='green', label='Input')
    recon = Patch(color='red', label='Reconstruction')
    fig.legend(handles=[input_handle, recon], loc='upper left')
    fig.suptitle(f"{title}")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import numpy as np
    random_array = np.random.rand(4, 19, 240)
    encodings = {
        'delta': random_array,
        'theta': random_array,
        'alpha': random_array,
        'beta': random_array,
        'gamma': random_array,
    }

    random_array2 = np.random.rand(4, 19, 240)
    decodings = {
        'delta': random_array2,
        'theta': random_array2,
        'alpha': random_array2,
        'beta': random_array2,
        'gamma': random_array2,
    }
    fig = plotReconstruction(encodings, decodings, "Test")
    plt.savefig("./test_figures/ReconstructionTEST.png") 
    plt.close(fig)