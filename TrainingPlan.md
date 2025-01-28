# Training Plan 

Current state-of-the-art self supervised learning for learning deep signal representations, specifically [wav2vec 2.0](https://arxiv.org/abs/2006.11477) employ a self supervised loss:

$$L = L_{m} + \alpha L_{d}$$

where $L_{m}$ represents the loss from the contrastive task where some of the input features are masked and $L_{d}$ represents the codebook diversity loss. This loss is similarly employed in [BENDR](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2021.653659/full#B5). We want to argue that adding a decoder that decodes the embedded features back to reconstruct the original signal, essentially demonstrating that signal information is preserved in the embedding, helps improve explainability and will improve loss. In other words:

$$L_{MENDR} = L_{m} + \alpha L_{d} + \beta L_{recon}$$

$L_{recon}$ represents the reconstruction loss of the reconstructed signal from the decoder. We will first try MSE; however, other techniques do exist for comparing the original vs reconstructed biosignal. $\beta$ represents a tuneable hyperparameter similar to $\alpha$ that is adjusted to determine how much we want to penalize reconstruction loss.

1. Pretraining
	- Pretraining Parameter Documentation: 
		- MENDR Encoder:
			1. Currently all preset in MENDREncoder.py MultiWaveletEncoder
		- MENDR Contextualizer:
   			1. `in_features`: The contextualizer’s input features. Currently, they are equal to `encoder_h.` Implicitly, the transformers’ input layers are `in_features` * 3, as per BENDR.
   			2. `dropout`: Dropout rate for contextualizer.
   			3. `position_encoder`: A seed for setting parameters of position encoder in contextualizer.
   			4. `epochs`: Epochs in the sense of MNE -- not training epochs! Segment time series into equal-length segments.
		- Training Parameters:
			1. `mask_span`: The number of time points masked in a mask. If random start time point + mask_span >= total # of time points, the number of time points in the mask is truncated to a length of the total - start time point.
			2. `mask_rate`: For each time point, there is a `mask_rate` probability that the time point is the start of the masked sequence. Note that mask overlaps are allowed.
   			3. `multi_gpu`: Use multiple GPUs for training.
      			4. `temp`: Temperature factor in self-supervised loss function (denoted as Kappa in BENDR paper).
			5. `encoder_grad_frac`: Scale gradient by this fraction during the learning step. 
			6. `num_negatives`: Number of distractors/negatives in contrastive task.
   			7. `enc_feat_l2`:  Similar to wav2vec, see equation 2 in wav2vec paper alpha parameter, want to encourage diversity in features in “BENDR codebook” and prevent a single feature’s value from becoming too big.
			8. `max_crop_frac`: Maximum fraction to crop data in RandomTemporalCrop transform.
			11. `train_frac`: Percentage of dataset to use for training (<1).
			12. `val_frac`: Percentage of dataset to use for validation (<1>). Note that `train_frac` + `val_frac` <= 1. 


	To find the optimal pretraining parameters, we will first try training on 10% of the HBN dataset. The HBN dataset has ~100,000 graphs/data points so we will train on around 10,000 data points. We perform this initial pre-pretraining as a hyperparameter search and it will hopefully inform us about which hyperparameters are optimal. We will perform a simple grid search of various parameters in order to determine which hyperparameters are best and evaluate how well the hyperparameters are based on the training loss over 3 epochs as well as a validation of unseen data that represents 1% of the HBN dataset (approx 1000) validation graphs/datapoints. 
2. Downstream tasks
	- TBD
