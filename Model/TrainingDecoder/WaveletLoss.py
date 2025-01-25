import ptwt
import torch
import torch.nn.functional as F
import gc


def WaveletLoss(original, wavelet_reconstructions, band_coeffs):
	'''
	Compute wavelet loss

	Parameters:
		original: torch.Tensor
			Original signal in shape (batch_size, num_channels, num_samples)
		wavelet_reconstructions: dict
			Predicted Wavelet coefficients, each entry in 
			the dictionary is a tensor of shape (batch_size, num_channels, num_samples)
		band_coeffs: dict
			Dictionary weighing the loss for each band.
	'''

	dbt_dwpt = ptwt.WaveletPacket(original,
							   wavelet='db4',
							   mode='reflect',
							   maxlevel=6,
							   axis=-1)
	
	og_dict = {
		'delta': dbt_dwpt['aaaaaa'],
		'theta': dbt_dwpt['aaaaad'],
		'alpha': dbt_dwpt['aaaad'],
		'beta': dbt_dwpt['aaad'],
		'gamma': dbt_dwpt['aad'],
		'other': dbt_dwpt['ad'],
		'high': dbt_dwpt['d']
	}


	loss = 0
	for band in wavelet_reconstructions.keys():
		loss += band_coeffs[band] * F.mse_loss(_std_norm(wavelet_reconstructions[band]), _std_norm(og_dict[band]))

	del dbt_dwpt
	del og_dict
	gc.collect()

	return loss

def _std_norm(x):
	mean = torch.mean(x, dim=(0, 1, 2), keepdim=True)
	std = torch.std(x, dim=(0, 1, 2), keepdim=True)
	x = (x - mean) / std
	return x



if __name__ == "__main__":
	x = torch.randn(4, 19, 15360)

	loss = WaveletLoss(x, dict(), dict())