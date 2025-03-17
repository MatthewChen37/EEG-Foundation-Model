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

def simplePipeline(raw, sample_rate=128, low_pass=75, exclude_epochs=[0, -1], exclude_short_epochs=True):
	'''
	Version 2
	1. Currently data is in volts where data is around 1e-5
	Scale EEG data to a base unit of 0.1 mV by multiplying by 10,000
	so that data is between -1 and 1

	2. Downsample to 128 Hz

	3. Filter between 0.1 and 75 Hz as done in LaBRaM (https://arxiv.org/pdf/2405.18765)

	4. Notch filter 60 Hz power line noise and its harmonics

	5. Exclude first and last epoch of TUH like CBraMod
	'''
	raw.filter(0.1, low_pass, verbose=False)
	raw.notch_filter((60, 120), verbose=False)
	epochs = make_fixed_length_epochs(raw, duration=60, preload=True)
	epochs = epochs.load_data()
	if exclude_short_epochs and len(epochs) <= 3:
		raise ValueError("Not enough epochs")
	if len(exclude_epochs) > 0:
		epochs.drop(exclude_epochs, verbose=False)
	epochs.drop_bad(verbose=False)
	# Although downsampling after epoching causes spectral leakage,
	# https://gist.github.com/larsoner/01642cb3789992fbca59
	# https://mne.discourse.group/t/are-annotations-created-from-events-robust-to-resampling/6344/2
	# shows that downsampling before epoching causes event jittering
	# and so we apply this fact to all data even in the self-supervised learning
	# case. Since the epoch is 1 minute it should only create small 
	# edge artifacts in each sample.
	epochs.resample(sample_rate, verbose=False)
	epochs.apply_function(lambda x: x * 1e5, verbose=False)
	return epochs

def group_list(data, size):
    return [data[i:i + size] for i in range(0, len(data), size)]