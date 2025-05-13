import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plotSPDEmbedding(wavelet_manifold_output, combined_manifold_output, combined_manifold_output_masked, subject_names, masked_idxes, num_patches=11, num_rows=4, num_cols=11):
    batch_size = masked_idxes.shape[0]
    B = batch_size * num_patches
    N = combined_manifold_output.shape[-1]
    assert B % num_patches == 0, f"Batch Size {B} is not divisible by {num_patches}"
    #print(f"B: {B} N: {N} num_patches: {num_patches} num_rows: {num_rows} num_cols: {num_cols}")
    wavelet_figs = dict()
    if wavelet_manifold_output is not None:
        # Wavelet Manifold Embeddings
        for band, wavelet_batch in wavelet_manifold_output.items():
            wavelet_fig, wavelet_axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3), subplot_kw=dict(projection='3d', elev=45, azim=45, roll=45)) # Always look through the view of positive octant

            _plotBatchWavelet(wavelet_axs, num_rows, num_cols,
                            wavelet_batch.clone().reshape(batch_size, -1, N, N)[:len(subject_names)], 
                            subject_names)
            wavelet_fig.suptitle(f"{band} SPD Embeddings")
            wavelet_fig.tight_layout()
            wavelet_figs[band] = wavelet_fig # Figure is a BATCH_SIZE / NUM_FIGS_PER_ROW for each manifold embedding

    combined_fig, combined_axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3), subplot_kw=dict(projection='3d', elev=45, azim=45, roll=45)) # Always look through the view of positive octant
    _plotBatchCombined(combined_axs, num_rows=num_rows, num_cols=num_patches,
                      output=combined_manifold_output.clone().reshape(batch_size, num_patches, N, N)[:len(subject_names)],
                      output_masked=combined_manifold_output_masked.clone().reshape(batch_size, num_patches, N, N)[:len(subject_names)],
                      masked_idexes=masked_idxes,
                      subject_names=subject_names)
    handles, labels = combined_axs[0, 0].get_legend_handles_labels()
    combined_fig.legend(handles, labels, loc='upper left')
    combined_fig.suptitle("Combined SPD Embeddings")
    combined_fig.tight_layout()
    return wavelet_figs, combined_fig

def _plotBatchCombined(axs, num_rows, num_cols, output, output_masked, masked_idexes, subject_names):
    for row in range(num_rows):
        for col in range(num_cols):
            output_matrix = output[row, col, :, :]
            output_masked_matrix = output_masked[row, col, :, :]
            #axs[row, col].view_init(elev=45, azim=45, roll=45) 
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
            axs[row, col].set_zticks([])
            _plot_ellipsoid_3D_PCA(output_matrix, axs[row, col], color='b', label='Original', alpha=0.5)
            _plot_ellipsoid_3D_PCA(output_masked_matrix, axs[row, col], color='r', label='Reconstruction', alpha=0.5)
            if masked_idexes[row, col] == True:
                axs[row, col].set_title(f"Patch {col + 1} (MASKED)")
            else:
                axs[row, col].set_title(f"Subject {subject_names[row]}, Patch {col + 1}")
            #handles, labels = axs[row, col].get_legend_handles_labels()

    #fig.legend(handles, labels, loc='upper left')

def _plotBatchWavelet(axs, num_rows, num_cols, spd_batch, subject_names):
    #print(f"SPD Shape: {spd_batch.shape} Subject_names: {len(subject_names)}")
    for row in range(num_rows):
        for col in range(num_cols):
            spd_matrix = spd_batch[row, col, :, :]
            #axs[row, col].view_init(elev=45, azim=45, roll=45) # Always look through the view of positive octant
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
            axs[row, col].set_zticks([])
            _plot_ellipsoid_3D_PCA(spd_matrix, axs[row, col])
            axs[row, col].set_title(f"Subject {subject_names[row]}, Patch {col + 1}")

def _plot_ellipsoid_3D_PCA(spd_matrix, ax, color='b', label='Original', alpha=1):
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
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color=color, label=label, alpha=alpha)

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