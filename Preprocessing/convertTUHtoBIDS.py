import mne
from mne_bids import BIDSPath, read_raw_bids, write_raw_bids
import pandas as pd
import numpy as np


def convertTUHtoBIDS(filepath):





def convertTUSZtoBIDS():






# From https://github.com/braindecode/braindecode/blob/master/braindecode/datasets/tuh.py
def _rename_channels(raw):
	"""
	Renames the EEG channels using mne conventions and sets their type to 'eeg'.

	See https://isip.piconepress.com/publications/reports/2020/tuh_eeg/electrodes/
	"""
	# remove ref suffix and prefix:
	# TODO: replace with removesuffix and removeprefix when 3.8 is dropped
	mapping_strip = {
		c: c.replace("-REF", "").replace("-LE", "").replace("EEG ", "")
		for c in raw.ch_names
	}
	raw.rename_channels(mapping_strip)

	montage1005 = mne.channels.make_standard_montage("standard_1005")

	# Changed this line
	mapping_eeg_names = {
		c.upper(): c.upper() for c in montage1005.ch_names if c.upper() in raw.ch_names
	}

	# Set channels whose type could not be inferred (defaulted to "eeg") to "misc":
	non_eeg_names = [c for c in raw.ch_names if c not in mapping_eeg_names]
	if non_eeg_names:
		non_eeg_types = raw.get_channel_types(picks=non_eeg_names)
		mapping_non_eeg_types = {
			c: "misc" for c, t in zip(non_eeg_names, non_eeg_types) if t == "eeg"
		}
		if mapping_non_eeg_types:
			raw.set_channel_types(mapping_non_eeg_types)

	if mapping_eeg_names:
		# Set 1005 channels type to "eeg":
		raw.set_channel_types(
			{c: "eeg" for c in mapping_eeg_names}, on_unit_change="ignore"
		)
		# Fix capitalized EEG channel names:
		raw.rename_channels(mapping_eeg_names)