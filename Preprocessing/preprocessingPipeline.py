import numpy as np
from autoreject import AutoReject
from mne import make_fixed_length_epochs

def simplePipeline(raw):
	'''
	Version 1

	Temporal scales of 1, 5, 15, and 60 second EEG data	
	'''
	raw.filter(0.5, 50) # Should be max 40 but we set it to 50 just in case
	raw = raw.resample(128) # Downsample to 128 Hz 
	epochs = [make_fixed_length_epochs(raw, duration=duration, preload=True) for duration in [1, 5, 15, 60]]
	cleaned_epochs = [AutoReject(verbose=False, cv=min(5, len(epoch))).fit_transform(epoch) for epoch in epochs]
	return cleaned_epochs