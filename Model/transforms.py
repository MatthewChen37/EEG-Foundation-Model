import torch
import numpy as np
from numbers import Real
from torch.fft import fft, ifft
from sklearn.utils import check_random_state

'''
NONE OF THESE ARE REALLY USED
'''

'''
Based on: https://github.com/SPOClab-ca/dn3/blob/master/dn3/transforms/batch.py
'''
class BatchTransform(object):

    def __init__(self):
        """
        Batch transforms are operations that are performed on trial tensors after being accumulated into batches via the
        :meth:`__call__` method. Ideally this is implemented with pytorch operations for ease of execution graph
        integration.
        """

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, *x):
        """
        Modifies a batch of tensors.

        Parameters
        ----------
        x : torch.Tensor, tuple
            A batch of trial instance tensor. If initialized with `only_trial_data=False`, then this includes batches
            of all other loaded tensors as well.
        training: bool
                  Indicates whether this is a training batch or otherwise, allowing for alternate behaviour during
                  evaluation.

        Returns
        -------
        x : torch.Tensor, tuple
            The modified trial tensor batch, or tensors if not `only_trial_data`
        """
        raise NotImplementedError()


class RandomTemporalCrop(BatchTransform):

    def __init__(self, max_crop_frac=0.25, temporal_axis=-1):
        """
        Uniformly crops the time-dimensions of a batch.

        Parameters
        ----------
        max_crop_frac: float
                       The is the maximum fraction to crop off of the trial.
        """
        super(RandomTemporalCrop, self).__init__(only_trial_data=True)
        assert 0 < max_crop_frac < 1
        self.max_crop_frac = max_crop_frac
        self.temporal_axis = temporal_axis

    def __call__(self, x):

        trial_len = x.shape[self.temporal_axis]
        crop_len = np.random.randint(int((1 - self.max_crop_frac) * trial_len), trial_len)
        offset = np.random.randint(0, trial_len - crop_len)

        return x[..., offset:offset + crop_len]

class RandomGaussianNoise(BatchTransform):
    def __init__(self):
        """
        Adds Gaussian Noise to each batch by calculating the mean signal
        and std of each batch
        """
        super(RandomGaussianNoise, self).__init__()

    def __call__(self, x):
        mean = torch.mean(x, dim=(0, 1), keepdim=True)
        std = torch.std(x, dim=(0, 1), keepdim=True)

        noise = torch.normal(mean, std)
        return x + noise

class RandomFTSurrogate(BatchTransform):
    '''
    Inspired by https://arxiv.org/pdf/1806.08675 we essentially add noise in the Fourier domain and want to ensure that embeddings 
    of an EEG patch are close to their FT surrogates in latent space.
    '''
    def __init__(self, phase_noise_magnitude=0.4, random_state=42):
        """
        Adds Noise in the Fourier domain to each sample in batch 
        """
        super(RandomFTSurrogate, self).__init__()
        self.phase_noise_magnitude = phase_noise_magnitude
        self.random_state = random_state
        self._new_random_fft_phase = {0: self._new_random_fft_phase_even, 1: self._new_random_fft_phase_odd}

        assert (isinstance(self.phase_noise_magnitude, Real) and 0 <= self.phase_noise_magnitude <= 1
        ), f"eps must be a float between 0 and 1. Got {self.phase_noise_magnitude}."

    '''
    Copied and modified from https://github.com/braindecode/braindecode//blob/master/braindecode/augmentation/functional.py#L112-L175
    '''
    def __call__(self, x):
        f = fft(x.double(), dim=-1)
        device = x.device

        n = f.shape[-1]
        random_phase = self._new_random_fft_phase[n % 2](
            f.shape[0],
            1,
            n,
            device=device,
            random_state=self.random_state,
        )
        # Note no channel independence
        f_shifted = f * torch.exp(self.phase_noise_magnitude * random_phase)
        shifted = ifft(f_shifted, dim=-1)
        transformed_X = shifted.real.float()
        return transformed_X

    def _new_random_fft_phase_odd(self, batch_size, c, n, device, random_state):
        rng = check_random_state(random_state)
        random_phase = torch.from_numpy(
            2j * np.pi * rng.random((batch_size, c, (n - 1) // 2))
        ).to(device)
        return torch.cat(
            [
                torch.zeros((batch_size, c, 1), device=device),
                random_phase,
                -torch.flip(random_phase, [-1]),
            ],
            dim=-1,
        )

    def _new_random_fft_phase_even(self, batch_size, c, n, device, random_state):
        rng = check_random_state(random_state)
        random_phase = torch.from_numpy(
            2j * np.pi * rng.random((batch_size, c, n // 2 - 1))
        ).to(device)
        return torch.cat(
            [
                torch.zeros((batch_size, c, 1), device=device),
                random_phase,
                torch.zeros((batch_size, c, 1), device=device),
                -torch.flip(random_phase, [-1]),
            ],
            dim=-1,
        )


if __name__ == "__main__":
    transform1 = RandomGaussianNoise()
    transform2 = RandomFTSurrogate(phase_noise_magnitude=1.0, random_state=1)

    x = torch.randn(4, 2, 10)

    transformed1_x = transform1(x)

    print("Transformed1_x shape:", transformed1_x.shape)

    assert not torch.equal(x, transformed1_x), "Transform 1 is equal! It did nothing."

    transformed2_x = transform2(x)

    print("Transformed2_x shape:", transformed2_x.shape)

    assert not torch.equal(x, transformed2_x), "Transform 2 is equal! It did nothing."

    print("Tests passed!")