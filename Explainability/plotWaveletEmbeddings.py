from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from .riemannien_dimension_reduction.src.riem_riem_algo import Riem_Riem_MDS, Riem_Riem_tSNE
from .riemannien_dimension_reduction.src.utils import plot_results_R
from pyriemann.utils.test import is_sym_pos_def
import numpy as np 
import torch
import umap
import plotly.io as pio

def plotWaveletEmbeddingsEuclidean(wavelet_manifold_output, combined_manifold_output, title, num_sample_patches=32):

    with torch.no_grad():
        umap_reducer = umap.UMAP(n_components=2)
        fig = plt.figure(figsize=(12, 12))
        data = []
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            data.append(wavelet_manifold_output[band][:num_sample_patches].clone().detach().cpu().numpy())
        data.append(combined_manifold_output[:num_sample_patches].clone().detach().cpu().numpy())
        data = np.concatenate(data, axis=0)
        embedding = umap_reducer.fit_transform(data)
        assert embedding.shape[0] == num_sample_patches * 6, f"Embedding Shape: {embedding.shape} Expected Shape: {num_sample_patches * 6}"
        legend_elements = []
        for i in range(num_sample_patches):
            plt.scatter(embedding[i, 0], embedding[i, 1], label=f"Delta Embedding {i}", c=f"C{i}", alpha=0.5)
            plt.scatter(embedding[16 + i, 0], embedding[16 + i, 1], label=f"Theta Embedding {i}", c=f"C{i}", alpha=0.5)
            plt.scatter(embedding[32 + i, 0], embedding[32 + i, 1], label=f"Alpha Embedding {i}", c=f"C{i}", alpha=0.5)
            plt.scatter(embedding[48 + i, 0], embedding[48 + i, 1], label=f"Beta Embedding {i}", c=f"C{i}", alpha=0.5)
            plt.scatter(embedding[64 + i, 0], embedding[64 + i, 1], label=f"Gamma Embedding {i}", c=f"C{i}", alpha=0.5)   
            plt.scatter(embedding[80 + i, 0], embedding[80 + i, 1], label=f"Combined Embedding {i}", c=f"C{i}", alpha=1.0)
            LegendElement = Patch(color=f"C{i}", label=f"Sample {i}")
            legend_elements.append(LegendElement)

        plt.title(f"{title}")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend(handles=legend_elements, loc='upper left')
        plt.tight_layout()

    return fig

def plotWaveletEmbeddingsRiemannian(wavelet_manifold_output, combined_manifold_output, title, num_samples=1, num_sample_patches=11, reduction="TSNE"):
    with torch.no_grad():
        if reduction == "TSNE":
            riemannian_reducer = Riem_Riem_tSNE(perplexity=int(0.75 * num_sample_patches * 6), max_it=10000, max_time=6000)
        elif reduction == "MDS":
            riemannian_reducer = Riem_Riem_MDS(max_it=100000, max_time=6000)
        else:
            raise ValueError("Reduction must be either 'TSNE' or 'MDS'")

        data = []
        labels = []
        for sample_idx in range(num_samples):
            for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                data.append(wavelet_manifold_output[band][sample_idx][:num_sample_patches].clone().detach().cpu().numpy())
            data.append(combined_manifold_output[sample_idx][:num_sample_patches].clone().detach().cpu().numpy())

            for embedding_idx in range(num_sample_patches):
                labels.append('Delta')
                labels.append('Theta')
                labels.append('Alpha')
                labels.append('Beta')
                labels.append('Gamma')
                labels.append('Combined')

        labels = np.array(labels)
        legends = np.unique(labels)

        data = np.concatenate(data, axis=0)
        embedding = riemannian_reducer.fit(data)
        assert embedding.shape[0] == num_samples * num_sample_patches * 6, f"Embedding Shape: {embedding.shape} Expected Shape: {num_samples * num_sample_patches * 6}"

        return plot_results_R(
            embedding,
            labels,
            f"Riemannian-Riemannian {title}-{reduction}",
            legends,
        )




if __name__ == "__main__":
    random_tensor = torch.ones(44, 190)
    wavelet_manifold_output = {
        'delta': random_tensor.clone().detach().cpu(),
        'theta': random_tensor.clone().detach().cpu(),
        'alpha': random_tensor.clone().detach().cpu(),
        'beta': random_tensor.clone().detach().cpu(),
        'gamma': random_tensor.clone().detach().cpu(),
    }
    combined_manifold_output = random_tensor.clone().detach().cpu()
    fig = plotWaveletEmbeddingsEuclidean(wavelet_manifold_output, combined_manifold_output, "Test")
    plt.savefig("./test_figures/WaveletEmbeddingsEuclidean.png") 
    plt.close(fig)
    def random_spd_batch(batch_size, n):
        return torch.tensor(np.array([
            random_spd_matrix(n) for i in range(batch_size)
        ])).float()

    def random_spd_matrix(n, regularization=10):
        A = np.random.rand(n, n)
        return np.dot(A, A.transpose()) + np.ones((n, n)) * regularization

    device = 'cpu'

    example_input = {
        'delta': random_spd_batch(44, 19).to(device).float(),
        'theta': random_spd_batch(44, 19).to(device).float(),
        'alpha': random_spd_batch(44, 19).to(device).float(),
        'beta': random_spd_batch(44, 19).to(device).float(),
        'beta': random_spd_batch(44, 19).to(device).float(),
        'gamma': random_spd_batch(44, 19).to(device).float(),
    }

    combined_manifold_output = random_spd_batch(44, 19).to(device).float()

    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        assert is_sym_pos_def(example_input[band]), f"Input {band} is not symmetric positive definite"

    assert is_sym_pos_def(combined_manifold_output), "Combined Manifold is not symmetric positive definite"

    fig = plotWaveletEmbeddingsRiemannian(example_input, combined_manifold_output, "Test", reduction="TSNE")
    fig.write_html("./test_figures/WaveletEmbeddingsRiemannianTSNE.html")