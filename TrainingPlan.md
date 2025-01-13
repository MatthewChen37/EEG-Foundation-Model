# Training Plan 

1. Pretraining
	- Pretraining Parameter Documentation: 
		- Spatial-Temporal Embedder:
			1. `seq_len`: Input sequence length of TimesNet Block Module (Total # of time points).
			2. `pred_len` is the output sequence length of the TimesNet Block Module. Currently, it is equal to the sequence length, so the Block module can “learn” a temporal signal embedding technique similar to positional embedding techniques.
			3.  `top_k`: Top K frequencies will be selected in the Block Module.
			4. `d_model`: Number of EEG channels in original data. Used in TimesNet Block Module.
			5. `d_ff`: Feedforward dimension of Inception Block in TimesNet.
			6. `num_kernels`: Number of kernels in the convolution of Inception Block in TimesNet Block Module.
			7. `num_heads`: Number of heads in multi-head attention layer in GAT.
			8. `ste_dropout`: Dropout rate for GAT.
		- BENDR Encoder:
			1. `encoder_h`: The encoder’s hidden dimension is designed to create “BENDR” features.
			2. `enc_width`: Encoder width, i.e. the kernel size of convolution layers.
			3. `enc_downsample` is the encoder downsample, which is the kernel’s stride in the convolution layers. In BENDR, `enc_width` = `enc_downsample`.
			4. `enc_dropout`: Encoder dropout rate.
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
2. Downstream tasks
	- TBD
