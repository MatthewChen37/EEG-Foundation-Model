import numpy as np
from mne_connectivity import spectral_connectivity_epochs, spectral_connectivity_time
import torch

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

	# Normalize values between 0 and 1 as per https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
	distance_matrix = (distance_matrix - distance_matrix.min())/ (distance_matrix.max() - distance_matrix.min())
	return distance_matrix

def _get_channel_positions(list):
	'''
	Gets electrode position from list of channel dicts as a triple.
	'''
	positions = np.empty((len(list), 3))
	for channel_idx in range(len(list)):
		positions[channel_idx] = list[channel_idx]['loc'][:3]
	return positions

def _createEdgeIndexTensor(adjMatrix):
	'''
	Converts the adjacency matrix to a tensor of edge indices.
	'''
	edge_indices = np.argwhere(adjMatrix)
	edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
	return edge_indices

def createEdges(adjMatrixList):
	'''
	Converts a list of adjacency matrices to a tensor of edge features
	'''
	adjMatrixDim = adjMatrixList[0].shape[0]
	# Assert that all adjacency matrices have the same dimensions
	for adjMatrix in adjMatrixList:
		assert adjMatrix.shape[0] == adjMatrix.shape[1]
		assert adjMatrix.shape[0] == adjMatrixDim
	
	edge_indices = _createEdgeIndexTensor(adjMatrixList[0])
	edge_weights = torch.zeros((edge_indices.shape[1], len(adjMatrixList)), dtype=torch.float32)
	for i in range(edge_indices.shape[1]):
		for j in range(len(adjMatrixList)):
			edge_weights[i][j] = adjMatrixList[j][edge_indices[0][i], edge_indices[1][i]]
	return edge_indices, edge_weights

if __name__ == "__main__":
	adj_matrix = np.random.rand(5, 5)
	adj_matrix_list = [adj_matrix, adj_matrix, adj_matrix]
	edge_indices, edge_weights = createEdges(adj_matrix_list)
	print(edge_indices.shape, edge_weights.shape)

	assert edge_indices.shape[1] == edge_weights.shape[0]
	assert len(adj_matrix_list) == edge_weights.shape[1]

	# Ensure adj matrices are properly flattened
	for i in range(len(edge_indices[1])):
		for j in range(len(adj_matrix_list)):
			assert edge_weights[i][j] == adj_matrix_list[j][edge_indices[0][i], edge_indices[1][i]]
	print("All tests passed!")	

