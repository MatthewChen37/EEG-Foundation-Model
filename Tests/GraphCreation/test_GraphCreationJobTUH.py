import argparse
import os
from GraphCreation.GraphCreationJobTUH import _process_subject

TEST_SUBJECT = "/home/hice1/mchen439/bmed-sp-wang/BrainData/eegdata/TUH-Processed/aaaaaaaa"


def test_process_subject_128Hz():
    test_args = argparse.Namespace(
        input_directory=TEST_SUBJECT, version="128Hz")
    relevant_bands = _process_subject(test_args, TEST_SUBJECT)

    assert relevant_bands is not None
    assert 'delta' in relevant_bands
    assert 'theta' in relevant_bands
    assert 'alpha' in relevant_bands
    assert 'beta' in relevant_bands
    assert 'gamma' in relevant_bands
    assert 'high_freq' in relevant_bands

    assert relevant_bands['delta'].shape[2] == 240
    assert relevant_bands['theta'].shape[2] == 240
    assert relevant_bands['alpha'].shape[2] == 480
    assert relevant_bands['beta'].shape[2] == 960
    assert relevant_bands['gamma'].shape[2] == 1920
    assert relevant_bands['high_freq'].shape[2] == 3840

    assert os.path.exists(os.path.join(
        TEST_SUBJECT, f"wavelet_decompositions_v{test_args.version}"))
    assert os.path.exists(os.path.join(
        TEST_SUBJECT, "s0012015-01tcpar-t000_epo_delta_band_epoch_10.pt"))
    assert os.path.exists(os.path.join(
        TEST_SUBJECT, "s0012015-01tcpar-t000_epo_theta_band_epoch_10.pt"))
    assert os.path.exists(os.path.join(
        TEST_SUBJECT, "s0012015-01tcpar-t000_epo_alpha_band_epoch_10.pt"))
    assert os.path.exists(os.path.join(
        TEST_SUBJECT, "s0012015-01tcpar-t000_epo_beta_band_epoch_10.pt"))
    assert os.path.exists(os.path.join(
        TEST_SUBJECT, "s0012015-01tcpar-t000_epo_gamma_band_epoch_10.pt"))
    assert os.path.exists(os.path.join(
        TEST_SUBJECT, "s0012015-01tcpar-t000_epo_high_freq_band_epoch_10.pt"))

    assert os.path.exists(os.path.join(
        TEST_SUBJECT, "s0012015-01tcpar-t000_epo_delta_band_epoch_28.pt"))
    assert os.path.exists(os.path.join(
        TEST_SUBJECT, "s0012015-01tcpar-t000_epo_theta_band_epoch_28.pt"))
    assert os.path.exists(os.path.join(
        TEST_SUBJECT, "s0012015-01tcpar-t000_epo_alpha_band_epoch_28.pt"))
    assert os.path.exists(os.path.join(
        TEST_SUBJECT, "s0012015-01tcpar-t000_epo_beta_band_epoch_28.pt"))
    assert os.path.exists(os.path.join(
        TEST_SUBJECT, "s0012015-01tcpar-t000_epo_gamma_band_epoch_28.pt"))
    assert os.path.exists(os.path.join(
        TEST_SUBJECT, "s0012015-01tcpar-t000_epo_high_freq_band_epoch_28.pt"))

    assert os.path.exists(os.path.join(
        TEST_SUBJECT, f"graphs_v{test_args.version}"))
