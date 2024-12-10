import mne
import os


def readEEG(filepath, preload=True):
	dirname = os.path.dirname(filepath)
	filename = os.path.basename(filepath)
	if filepath[-4:] == ".set": # EEG Lab
		subject_name = filename.split("_")[0]
		EEG = mne.io.read_raw_eeglab(filepath, preload=preload, verbose=False)
		try: 
			annotations_df = pd.read_csv(os.path.join(dirname, f'{subject_name}_rs-HEP_events.tsv'), sep='\t')
			annotations_df['duration'] = 0
			annotations = mne.Annotations(onset=annotations_df["Onset_ms"] / 512, duration=annotations_df["duration"], description=annotations_df["TrialType"])
			EEG.set_annotations(annotations)
		except FileNotFoundError:
			try:
				annotations_df = pd.read_csv(os.path.join(dirname, f'{subject_name}_task-40HzAuditoryEntrainment_events.tsv'), sep='\t')
				annotations = mne.Annotations(onset=annotations_df["onset"], duration=annotations_df["duration"], description=annotations_df["trial_type"])
				EEG.set_annotations(annotations)
			except FileNotFoundError:
				print("No events file. Skipping...")
		EEG.crop(tmin=60, tmax=360)
		return EEG
	elif filepath[-5:] == ".vhdr": # Brainvision
		EEG = mne.io.read_raw_brainvision(filepath, preload=preload, verbose=False)
		events, event_ids = mne.events_from_annotations(EEG, verbose=False)
		event_time_eyes_closed_start = events[events[:, 2] == 10][0][0] / EEG.info['sfreq']
		EEG = mne.add_reference_channels(EEG, ['FCz'])
		custom_montage = mne.channels.read_custom_montage("/home/matthew_chen/EEGHarmonizationPipeline/DeviceMetadata/AP-128.bvef")
		EEG.set_montage(custom_montage, on_missing="ignore", verbose=False)
		EEG.crop(tmin=event_time_eyes_closed_start, tmax=event_time_eyes_closed_start + 300)
		return EEG
	elif filepath[-4:]	== ".edf":
		subject_name = filename.split("_")[0]
		position_file = os.path.join(dirname, f"{subject_name}_electrodes.tsv")
		remove_lines_with_string(position_file, 'n/a')
		custom_montage = mne.channels.read_custom_montage(position_file)
		EEG = mne.io.read_raw_edf(filepath, preload=preload, exclude=['EOI', 'EOD', 'ECG'], infer_types=True, verbose=False)
		EEG = removeRef(EEG, custom_montage, subject_name)
		EEG.set_montage(custom_montage, on_missing='ignore', verbose=False)
		for i in range(len(EEG.info['chs'])):
			EEG.info['chs'][i]['loc'] /= 1000	
		EEG = dropLocationlessChannels(EEG)
		annotations_df = pd.read_csv(os.path.join(dirname, f'{subject_name}_task-protmap_events.tsv'), sep='\t')
		annotations = mne.Annotations(onset=annotations_df["onset"], duration=annotations_df["duration"], description=annotations_df["trial_type"])
		EEG.set_annotations(annotations)
		EEG.crop(tmin=60, tmax=360)
		return EEG
	raise Exception(f"No EEG found for file type: {filepath}")

