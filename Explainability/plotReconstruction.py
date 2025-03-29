import matplotlib.pyplot as plt

def plotReconstruction(inputs, decodings, title):
    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        fig = plt.figure(figsize=(13, 6))
        for channel in range(19):
            plt.plot(patchified_inputs[band][channel, :], label=f"Input {band} Channel {channel}", c="g")
            plt.plot(decodings[band][channel, :], label=f"Output {band} Channel {channel}", c="r")

        fig.title(f"{title} Band: {band}")
        fig.xlabel("Time")
        fig.ylabel("Amplitude (mV)")

        input_handle = mpatches.Patch(color='green', label='Input')
        recon = mpatches.Patch(color='red', label='Reconstruction')
        plt.legend(handles=[input_handle, recon], loc='upper left')
        fig.tight_layout()

        return fig