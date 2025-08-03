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
    subjects = [x.split(os.sep)[-1][4:] for x in subfolders if "sub-" in x]
    print(f"Found {len(subjects)} subjects")

    train_root = Path(args.output_dir) / "train"
    eval_root = Path(args.output_dir) / "eval"
    train_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_process_subject, args, subject, train_root, eval_root) for subject in subjects]
        for future in tqdm(futures):
            future.result()

def _process_subject(args, subject, train_root, eval_root):
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

                assert raw.n_times == joints_np.shape[1], f"Mismatch in timepoints: raw {raw.n_times}, joints {joints_np.shape[1]}"

                # 15 min train, 5 min eval split
                train_sec = 15 * 60
                eval_sec = 5 * 60
                sfreq = raw.info['sfreq']
                train_samples = int(train_sec * sfreq)
                eval_samples = int(eval_sec * sfreq)

                if raw.n_times < train_samples + eval_samples:
                    print(f"Warning: raw too short for train+eval split sub-{subject} ses-{session}, skipping")
                    continue

                raw_train = raw.copy().crop(tmin=0, tmax=train_sec, include_tmax=False)
                raw_eval = raw.copy().crop(tmin=train_sec, tmax=train_sec + eval_sec, include_tmax=False)

                # Normalize joints by dividing by 90 degrees
                joint_chs = ["GHR", "GKR", "GAR", "GHL", "GKL", "GAL"]
                for ch in joint_chs:
                    raw_train._data[raw_train.ch_names.index(ch), :] /= 90.0
                    raw_eval._data[raw_eval.ch_names.index(ch), :] /= 90.0

                # Epoch the splits (6 sec windows)
                epochs_train = simplePipeline(raw_train, sample_rate=128, low_pass=49.5,
                                             exclude_epochs=[], exclude_short_epochs=False)
                epochs_eval = simplePipeline(raw_eval, sample_rate=128, low_pass=49.5,
                                            exclude_epochs=[], exclude_short_epochs=False)

                epochs_train = epochs_train.drop_channels(joint_chs)
                epochs_eval = epochs_eval.drop_channels(joint_chs)

                # Save all train epochs + labels in train_root
                train_epo_dir = train_root / "epochs"
                train_epo_dir.mkdir(parents=True, exist_ok=True)
                train_out_path = train_epo_dir / f"sub-{subject}_ses-{session}_epo.fif"
                epochs_train.save(train_out_path, overwrite=True)

                train_labels = epochs_train.get_data(picks=joint_chs)  # shape (n_epochs, 6, n_times)
                train_labels_path = train_epo_dir / f"sub-{subject}_ses-{session}_labels.npy"
                np.save(train_labels_path, train_labels)

                # Save all eval epochs + labels in eval_root
                eval_epo_dir = eval_root / "epochs"
                eval_epo_dir.mkdir(parents=True, exist_ok=True)
                eval_out_path = eval_epo_dir / f"sub-{subject}_ses-{session}_epo.fif"
                epochs_eval.save(eval_out_path, overwrite=True)

                eval_labels = epochs_eval.get_data(picks=joint_chs)
                eval_labels_path = eval_epo_dir / f"sub-{subject}_ses-{session}_labels.npy"
                np.save(eval_labels_path, eval_labels)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", module="mne_bids")

    parser = argparse.ArgumentParser(description="MOBI EEG Processing: 15min train + 5min eval, normalized joints")
    parser.add_argument('--input_directory', type=str, required=True, help='MOBI BIDS dataset root')
    parser.add_argument('--output_dir', type=str, required=True, help='Combined output folder (with train/ and eval/ subfolders)')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
