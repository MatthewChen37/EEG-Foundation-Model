import numpy as np
from mne_connectivity import spectral_connectivity_time
import torch
import math

'''
This is a "Non-Deep" way of featurizing the relationship between
channels in the EEG data.

Not really used, but is a good reference.
'''
def createConnectivityTime(epochs):
	min_freq = 0.5
	max_freq = 128

	# Provide the freq points at 0.25 Hz intervals
	freqs = np.linspace(min_freq, max_freq, int((max_freq - min_freq) * 4 + 1))

	adj_matrices = torch.zeros((epochs.shape[0], epochs.shape[1], epochs.shape[1]), dtype=torch.float32)

	symmetric_adjMatrix = spectral_connectivity_time(epochs,
										   freqs=freqs,
										   method = ['coh', 'wpli'],
										   sfreq = 256,
										   fmin = min_freq,
										   fmax = max_freq,
										   faverage = True,
										   n_jobs = 4,
										   verbose=False)
	for i in range(len(symmetric_adjMatrix)):
		for epoch in range(epochs.shape[0]):
			adjMatrix = symmetric_adjMatrix[i].get_data('dense')[epoch, :, :, 0]
			adjMatrix = adjMatrix + adjMatrix.T - np.diag(np.diag(adjMatrix))
			adjMatrix[adjMatrix < (adjMatrix.mean() + adjMatrix.std())] = 0
			adj_matrices[epoch] = torch.tensor(adjMatrix, dtype=torch.float32)
	return adj_matrices


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

def createGeodesicDistanceMatrix(info):
	'''
	Compute the geodesic distance matrix between all pairs of electrodes.
	'''
	positions = _get_channel_positions(info['chs'])
	n_channels = len(positions)
	distance_matrix = np.zeros((n_channels, n_channels))

	r = 0.095 # Sphere "Default" https://mne.tools/stable/generated/mne.viz.plot_montage.html

	for i in range(n_channels):
		for j in range(n_channels):
			x1, y1, z1 = positions[i]
			x2, y2, z2 = positions[j]
			# From https://github.com/neerajwagh/eeg-gcnn/blob/ec9f692a9a5ce53fea27711a3ffcec024581116f/code_psd_deep_eeg_gcnn/EEGGraphDataset.py#L76
			distance_matrix[i, j] = r * math.acos(round(((x1 * x2) + (y1 * y2) + (z1 * z2)) / (r**2), 2))

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
	edge_indices = np.argwhere(adjMatrix != None)
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

def createPositionMatrix(info):
	'''
	Converts the electrode positions to a tensor.
	'''
	positions = _get_channel_positions(info['chs'])
	positions = torch.tensor(positions, dtype=torch.float32)
	return positions

def midpoint_on_sphere(p1, p2):
	'''
	Computes the midpoint between two points on the unit sphere.
	https://stackoverflow.com/questions/50451331/find-midpoint-between-two-points-on-a-sphere
	'''
	x1, y1, z1 = p1
	x2, y2, z2 = p2

	x3 = (x1 + x2) / 2
	y3 = (y1 + y2) / 2
	z3 = (z1 + z2) / 2

	r = math.sqrt(x3**2 + y3**2 + z3**2)
	return (x3/r, y3/r, z3/r)

if __name__ == "__main__":
	adj_matrix = np.random.rand(19, 19)
	adj_matrix_list = [adj_matrix]
	edge_indices, edge_weights = createEdges(adj_matrix_list)
	print("Edge Indices:", edge_indices.shape, "Edge Weights:", edge_weights.shape)

	assert edge_indices.shape[1] == edge_weights.shape[0]
	assert len(adj_matrix_list) == edge_weights.shape[1]

	# Ensure adj matrices are properly flattened
	for i in range(len(edge_indices[1])):
		for j in range(len(adj_matrix_list)):
			assert edge_weights[i][j] == adj_matrix_list[j][edge_indices[0][i], edge_indices[1][i]]

	info = {'chs': [{'loc': [0.0, 0.0, 0.0]}, {'loc': [0.05, 0.05, 0.05]}, {'loc': [0.02, 0.002, 0.002]}]}
	geodesic_distance_matrix = createGeodesicDistanceMatrix(info)

	print("All tests passed!")	