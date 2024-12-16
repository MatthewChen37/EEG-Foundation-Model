import numpy as np
from mne import make_fixed_length_epochs

def simplePipeline(raw):
	'''
	Version 1
	Temporal scales of 60 second EEG data	
	'''
	epochs = make_fixed_length_epochs(raw, duration=60, preload=True)
	epochs = epochs.resample(256)
	return epochs