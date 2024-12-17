import mne
import os
from mne_bids import BIDSPath, read_raw_bids
import numpy as np


# Based on 
# https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/File/_eeg/BP_EGI_Compatibility_Comparison_V002.pdf
# and
# https://www.egi.com/images/HydroCelGSN_10-10.pdf (specifically 256-Channel HCGSN 58 Centimeters)

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
}

'''
HBN_ELECTRODE_MAP = {
	'Fp1': 'E37',
	'Fp2': 'E18',
	'F7': 'E47',
	'F3': 'E36',
	'Fz': 'E21',
	'F4': 'E224',
	'F8': 'E2',
	'T3': 'E69',
	'C3': 'E59',
	'Cz': 'Cz',
	'C4': 'E183',
	'T4': 'E202',
	'T5': 'E96',
	'P3': 'E87',
	'Pz': 'E101',
	'P4': 'E153',
	'T6': 'E270',
	'O1': 'E116',
	'O2': 'E150',
	'A1': 'E49',
	'A2': 'E113'
}
'''

def readEEG(bidsPath, dataset):
	'''
	Read EEG data from a BIDS dataset.
	'''
	raw = read_raw_bids(bidsPath, extra_params={'preload':True,'verbose':False}, verbose=False)
	if dataset == "TUH":
		raw.set_eeg_reference(ref_channels=['A1', 'A2'])
		raw.set_eeg_reference(ref_channels=['Cz'])
	elif dataset == "HBN":
		electrodes = [f'E{i}' for i in range(1, 129)]
		to_drop = [electrode for electrode in electrodes if electrode not in HBN_ELECTRODE_MAP.values()]
		raw.drop_channels(to_drop)

		# For some reason the data is rotated 90 degrees clockwise so we unrotate it for visualization purposes
		transformation_matrix = np.asarray([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
		montage = raw.get_montage()
		montage.apply_trans(mne.transforms.Transform(fro='ctf_head', to='unknown', trans=transformation_matrix))
		raw.set_montage(montage)
	return raw