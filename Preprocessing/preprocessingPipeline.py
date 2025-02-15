import numpy as np
from mne import make_fixed_length_epochs

def simplePipelineOld(raw):
	'''
	Version 1
	Temporal scales of 60 second EEG data	
	'''
	epochs = make_fixed_length_epochs(raw, duration=60, preload=True)
	epochs = epochs.resample(256)
	return epochs

def simplePipeline(raw):
	'''
	Version 2
	1. Currently data is in volts where data is around 1e-5
	Scale EEG data to a base unit of 0.1 mV by multiplying by 10,000
	so that data is between -1 and 1

	2. Downsample to 128 Hz

	3. Filter between 0.1 and 75 Hz as done in LaBRaM (https://arxiv.org/pdf/2405.18765)

	4. Notch filter 60 Hz power line noise and its harmonics

	'''
	
	raw.filter(0.1, 75, verbose=False)
	raw.notch_filter((60, 120), verbose=False)
	epochs = make_fixed_length_epochs(raw, duration=60, preload=True)
	epochs = epochs.load_data()
	'''
	if len(epochs) <= 3:
		raise ValueError("Not enough epochs")
	#epochs.drop([0, len(epochs) - 1], verbose=False)
	'''
	epochs.drop_bad(verbose=False)
	epochs.resample(128, verbose=False)
	epochs.apply_function(lambda x: x * 1e5, verbose=False)
	return epochs