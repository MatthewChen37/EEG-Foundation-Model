import mne
from mne_bids import BIDSPath, write_raw_bids
import sys
sys.path.append("../")
import numpy as np
import pandas as pd
import os, argparse
from pathlib import Path
import warnings
from convertTUHtoBIDS import CHANNELS_TO_KEEP
from defusedxml import ElementTree

INDICES = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
		    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
			  'Fz', 'Cz', 'Pz']


def main(args):
	folders = os.listdir(args.input_directory)
	for i, folder in enumerate(folders):
		print(f"{i}: Processing {folder}")
		process_folder(args, folder)

def process_folder(args, folder):
	subject = folder.split("-")[0]
	session = folder.split("-")[1]
		
	raw_eeg_pd = pd.read_csv(os.path.join(args.input_directory, folder, "eeg.txt"), sep="\t", skiprows=1, header=None)
	raw_eeg_impedances_pd = pd.read_csv(os.path.join(args.input_directory, folder, "impedances-before.txt"), sep="\t", skiprows=20)
	raw_conductor_pd = pd.read_csv(os.path.join(args.input_directory, folder, "conductor.txt"), sep="\t", skiprows=2, header=None)
	joints = pd.read_csv(os.path.join(args.input_directory, folder, "joints.txt"), sep="\t", skiprows=2, header=None)
	# Create the montage
	montage, to_exclude = _safe_montage_creation(os.path.join(args.input_directory, folder, "digitizer.bvct"))

	ch_names = list(raw_eeg_impedances_pd['Phys. Chn.'].to_numpy())
	ch_names = [ch_name.strip() for ch_name in ch_names][:-2]
	channels_to_drop = [ch_name for ch_name in ch_names if ch_name not in CHANNELS_TO_KEEP]
	channels_to_drop.remove("C5") # T3 = C5
	channels_to_drop.remove("C6") # T4 = C6
	channels_to_drop.remove("P5") # T5 = P5
	channels_to_drop.remove("P6") # T6 = P6

	chtypes = []
	for ch_name in ch_names:
		if ch_name in ['FT9', "FT10", "TP9", "TP10"]:
			chtypes.append("eog")
		else:
			chtypes.append("eeg")

	info = mne.create_info(ch_names=ch_names, sfreq=100, ch_types=chtypes)
	data = raw_eeg_pd.iloc[:, 1:65].to_numpy()
	data = data.T

	raw = mne.io.RawArray(data, info)
	raw.info['line_freq'] = 60
	raw.drop_channels(channels_to_drop)
	raw.set_montage(montage)

	mapping = {
		"C5": "T3",
		"C6": "T4",
		"P5": "T5",
		"P6": "T6"
	}

	mne.rename_channels(raw.info, mapping)

	bids_path = BIDSPath(subject=subject, session=session, task="unknown", root=args.output_dir, datatype="eeg", suffix="eeg", extension=".fif")
	write_raw_bids(raw, bids_path, verbose=False, format="BrainVision", allow_preload=True, overwrite=True)

	joints_df = joints.iloc[:, 1:7]
	joints_df.columns = ["GHR", "GKR", "GAR", "GHL", "GKL", "GAL"]
	joints_df.to_csv(os.path.join(bids_path.directory, f"{subject}_{session}_joints.csv"), index=False)

	raw_conductor_pd.columns = ["Time", "Event"]	
	raw_conductor_pd.to_csv(os.path.join(bids_path.directory, f"{subject}_{session}_conductor.csv"), index=False)

# Copied from _parse_brainvision_digmontage in mne
def _safe_montage_creation(filename):
    FID_NAME_MAP = {"Nasion": "nasion", "RPA": "rpa", "LPA": "lpa"}
    et = ElementTree.parse(filename)
    sensors = et.find("CapTrakElectrodeList")
    fids, dig_ch_pos = dict(), dict()

    for s in sensors:
        name = s.find("Name").text

        is_fid = name in FID_NAME_MAP
        coordinates = 1e-3 * np.array(
            [float(s.find("X").text), float(s.find("Y").text), float(s.find("Z").text)]
        )

        # Fiducials
        if is_fid:
            fids[FID_NAME_MAP[name]] = coordinates
        # EEG Channels
        else:
            dig_ch_pos[name] = coordinates

        montage = mne.channels.make_dig_montage(ch_pos=dig_ch_pos, coord_frame="unknown", nasion=fids.get("nasion", None), lpa=fids.get("lpa", None), rpa=fids.get("rpa", None))

    excluded_sensors = et.find("ExcludedElectrodes")
    to_exclude = []
    for s in excluded_sensors:
        name = s.text
        to_exclude.append(name)

    return montage, to_exclude

def parse_args():
	parser = argparse.ArgumentParser(description='Preprocess TUH EEG data')
	parser.add_argument('--input_directory', type=str, help='Path to the directory containing the TUH EEG data')
	parser.add_argument('--output_dir', type=str, help='Path to the directory where the preprocessed data will be stored')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	# Ignore warnings
	warnings.filterwarnings("ignore")

	args = parse_args()
	Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	main(args)