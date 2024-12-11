import numpy as np
from mne_connectivity import spectral_connectivity_epochs, spectral_connectivity_time


def createSpectralConnectivityMatrix(epochs):
  adjMatrix = spectral_connectivity_epochs(epochs,
											method = 'wpli',
											sfreq = 128,
											fmin = 0.5,
											fmax = 40,
											faverage = True,
											n_jobs = 4, verbose=False).get_data('dense')[:, :, 0]
  adjMatrix = adjMatrix + adjMatrix.T - np.diag(np.diag(adjMatrix))
  adjMatrix[adjMatrix < (adjMatrix.mean() + adjMatrix.std())] = 0
  return adjMatrix


def createConnectivityTime(epochs):
	adjMatrix = spectral_connectivity_time(epochs,
											method = 'wpli',
											sfreq = 128,
											fmin = 0.5,
											fmax = 40,
											faverage = True,
											n_jobs = 4, verbose=False).get_data('dense')[:, :, 0]
	adjMatrix = adjMatrix + adjMatrix.T - np.diag(np.diag(adjMatrix))
	adjMatrix[adjMatrix < (adjMatrix.mean() + adjMatrix.std())] = 0
	return adjMatrix


def createDistanceMatrix(info):
	'''
	Compute the distance matrix between all pairs of electrodes.
	'''
	positions = _get_channel_positions(info['chs'])
	n_channels = len(positions)
	distance_matrix = np.zeros((n_channels, n_channels))
	for i in range(n_channels):
		for j in range(n_channels):
			distance_matrix[i, j] = np.linalg.norm(positions[i] - positions[j])
	return distance_matrix

def _get_channel_positions(list):
	'''
	Gets electrode position from list of channel dicts as a triple.
	'''
	positions = np.empty((len(list), 3))
	for channel_idx in range(len(list)):
		positions[channel_idx] = list[channel_idx]['loc'][:3]
	return positions