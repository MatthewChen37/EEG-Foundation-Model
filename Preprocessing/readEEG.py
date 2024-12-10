import mne
import os
from mne_bids import BIDSPath, read_raw_bids
import numpy as np

HBN_ELECTRODE_MAP = {
	'Fp1': 'E22',
	'Fp2': 'E9',
	'F7': 'E33',
	'F3': 'E24',
	'Fz': 'E11',
	'F4': 'E124',
	'F8': 'E122',
	'T3': 'E45',
	'C3': 'E36',
	'Cz': 'Cz',
	'C4': 'E104',
	'T4': 'E108',
	'T5': 'E58',
	'P3': 'E52',
	'Pz': 'E62',
	'P4': 'E92',
	'T6': 'E96',
	'O1': 'E70',
	'O2': 'E83',
	'A1': 'E49',
	'A2': 'E113'
}

def readEEG(bidsPath, dataset):
	'''
	Read EEG data from a BIDS dataset.
	'''
	raw = read_raw_bids(bidsPath)
	if dataset == "TUH":
		raw.load_data()
		raw.set_eeg_reference(ref_channels=['A1', 'A2'])
		raw.set_eeg_reference(ref_channels=['Cz'])
	elif dataset == "HBN":
		electrodes = [f'E{i}' for i in range(1, 129)]
		to_drop = [electrode for electrode in electrodes if electrode not in HBN_ELECTRODE_MAP.values()]

		# For some reason MNE reads in positions as cm so convert back to m
		transformation_matrix = np.asarray([[0, -100, 0, 0], [100, 0, 0, 0], [0, 0, 100, 0], [0, 0, 0, 100]])
		montage = raw.get_montage()
		montage.apply_trans(mne.transforms.Transform(fro='ctf_head', to='unknown', trans=transformation_matrix))
		raw.set_montage(montage)
	return raw