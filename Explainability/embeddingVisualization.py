import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plotSPDEmbedding(wavelet_manifold_output, combined_manifold_output, max_figs=16):
    wavelet_manifold_output = {band: wavelet[0] for band, wavelet in wavelet_manifold_output.items()}
    B, N, N = combined_manifold_output.shape
    MAX_FIGS = max_figs
    NUM_FIGS_PER_ROW = 4
    num_cols = MAX_FIGS // NUM_FIGS_PER_ROW # Hopefully MAX_FIGS = NUM_FIGS_PER_ROW ** 2

    wavelet_figs = dict()
    # Wavelet Manifold Embeddings
    for band, wavelet_batch in wavelet_manifold_output.items():
        wavelet_fig = plt.figure(figsize=(MAX_FIGS, MAX_FIGS))  # Square figure
        _plotBatch(wavelet_fig, NUM_FIGS_PER_ROW, num_cols, wavelet_batch)
        wavelet_fig.suptitle(f"{band} SPD Embeddings")
        wavelet_figs[band] = wavelet_fig # Figure is a BATCH_SIZE / NUM_FIGS_PER_ROW for each manifold embedding

    combined_fig = plt.figure(figsize=(MAX_FIGS, MAX_FIGS))  # Square figure
    _plotBatch(combined_fig, NUM_FIGS_PER_ROW, num_cols, combined_manifold_output)
    combined_fig.suptitle("Combined SPD Embeddings")

    return wavelet_figs, combined_fig

def _plotBatch(fig, num_figs_per_row, num_cols, spd_batch):
    for row in range(num_figs_per_row):
        for col in range(num_cols):
            flattened_index = num_figs_per_row * row + col
            spd_matrix = spd_batch[flattened_index, :, :]
            ax = fig.add_subplot(num_figs_per_row, num_cols, flattened_index + 1, projection='3d')
            ax.view_init(elev=45, azim=45, roll=45) # Always look through the view of positive octant
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            _plot_ellipsoid_3D_PCA(spd_matrix, ax)
            ax.set_title(f"{flattened_index + 1}")

def _plot_ellipsoid_3D_PCA(spd_matrix, ax):
    if isinstance(spd_matrix, torch.Tensor):
        spd_matrix = spd_matrix.clone().cpu().numpy()

    eigenvalues, eigenvectors = np.linalg.eigh(spd_matrix, UPLO='L')
    descending_indices = np.argsort(eigenvalues)[::-1]
    top_eigenvalues = eigenvalues[descending_indices[:3]]
    top_eigenvectors = eigenvectors[:, descending_indices[:3]][:3]
    # Note that unlike PCA the points already start out in 3D, we are just projecting the points along the basis if 
    # we were to truncate that basis into 3 Dimensions 

    # From https://stackoverflow.com/questions/75796504/plotting-an-ellipse-with-eigenvectors-using-matplotlib-and-numpy
    coefs = top_eigenvalues # eigenvals = (a0/c, a1/c, a2/c)
    #coefs = (1, 2, 2)  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1 
    # Radii corresponding to the coefficients:
    rx, ry, rz = 1/np.sqrt(coefs)

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100) # We sample 100^2 points for plotting
    v = np.linspace(0, np.pi, 100)

    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))


    points = np.stack([x.flatten(), y.flatten(), z.flatten()])
    # Rotate the ellipsoid according to the eigenvectors
    points_rotated = top_eigenvectors @ points
    x = points_rotated[0, :].reshape(x.shape[0], x.shape[1])
    y = points_rotated[1, :].reshape(y.shape[0], y.shape[1])
    z = points_rotated[2, :].reshape(z.shape[0], z.shape[1])

    # Plot:
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b')

    # Adjustment of the axes, so that they all have the same span:
    max_radius = max(rx, ry, rz)
    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

def _ellipsoid_sample(S, z_hat, m_FA, Gamma_Threshold=1.0):
    # Based on https://www.onera.fr/sites/default/files/297/C013_-_Dezert_-_YBSTributeMonterey2001.pdf
    # And https://math.stackexchange.com/questions/2174751/generate-random-points-within-n-dimensional-ellipsoid

    # S: SPD Matrix
    # z_hat: Center of ellipsoid
    # m_FA: Number of false alarms, i.e. number of points to sample
    # Gamma Threshold: Distance between prediction and true w.r.t to center (Probabiltity true measurement falls into validation gate)

    nz = S.shape[0]
    z_hat = z_hat.reshape(nz,1)

    X_Cnz = np.random.normal(size=(nz, m_FA))

    rss_array = np.sqrt(np.sum(np.square(X_Cnz),axis=0))
    kron_prod = np.kron( np.ones((nz,1)), rss_array)

    X_Cnz = X_Cnz / kron_prod       # Points uniformly distributed on hypersphere surface

    R = np.ones((nz,1))*( np.power( np.random.rand(1,m_FA), (1./nz)))

    unif_sph=R*X_Cnz;               # m_FA points within the hypersphere
    T = np.asmatrix(cholesky(S))    # Cholesky factorization of S => S=T’T


    unif_ell = T.H*unif_sph ; # Hypersphere to hyperellipsoid mapping

    # Translation and scaling about the center
    z_fa=(unif_ell * np.sqrt(Gamma_Threshold)+(z_hat * np.ones((1,m_FA))))

    return np.array(z_fa)