import mne
from mne_bids import BIDSPath, read_raw_bids
import numpy as np
import pandas as pd
import sys
sys.path.append("../")
from preprocessingPipeline import simplePipeline
import os, argparse
from pathlib import Path
import warnings
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def main(args):
    subfolders = [f.path for f in os.scandir(args.input_directory) if f.is_dir()]
    subjects = [x.split("/")[-1][4:] for x in subfolders if "sub-" in x]
    print(f"Found {len(subjects)} subjects")

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_process_subject, args, subject) for subject in subjects]
        for future in tqdm(futures):
            future.result()
            

def _process_subject(args, subject):
    subject_out_dir = Path(args.output_dir) / subject / "epochs"
    subject_out_dir.mkdir(parents=True, exist_ok=True)

    subject = subject[-4:]
    subject_dir = Path(args.input_directory) / f"sub-{subject}"
    sessions = [x.name[4:] for x in subject_dir.iterdir() if x.is_dir() and "ses-" in x.name]

    for session in sessions:
        bids_path = BIDSPath(root=args.input_directory, subject=subject, session=session, extension=".eeg")
        for bp in bids_path.match():
            if bp.fpath.suffix == ".vhdr":
                raw = read_raw_bids(bp, verbose=False, extra_params={'preload': True})
                assert len(raw.ch_names) == 19, f"Unexpected number of EEG channels: {len(raw.ch_names)}"
                joints_csv = Path(bp.directory) / f"{subject}_{session}_joints.csv"
                joints = pd.read_csv(joints_csv)

                joints_np = joints.to_numpy().T
                info = mne.create_info(
                    ch_names=["GHR", "GKR", "GAR", "GHL", "GKL", "GAL"],
                    sfreq=raw.info["sfreq"],
                    ch_types=["misc"] * 6
                )
                joints_raw = mne.io.RawArray(joints_np, info, verbose=False)

                raw.add_channels([joints_raw], force_update_info=True)

                epochs = simplePipeline(
                    raw,
                    sample_rate=128,
                    low_pass=49.5,
                    exclude_epochs=[],
                    exclude_short_epochs=False
                )

                output_path = subject_out_dir / f"{session}_epo.fif"
                epochs.save(output_path, overwrite=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess MOBI EEG + joint angle data')
    parser.add_argument('--input_directory', type=str, help='Path to BIDS-formatted MOBI dataset')
    parser.add_argument('--output_dir', type=str, help='Path to output directory for epochs')
    return parser.parse_args()

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", module="mne_bids")
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)