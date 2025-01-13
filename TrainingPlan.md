# Training Plan 

1. Pretraining
	- Pretraining Parameter Documentation:
		-Spatial-Temporal Embedder:
			1. `seq_len`: Input sequence length of TimesNet Block Module (Total # of time points).
	  		2. `pred_len`: Output sequence length of TimesNet Block Module. Currently, this is equal to the sequence length so the Block module can “learn” a temporal signal embedding, similar to positional embedding techniques.
	    		3. `top_k`: Top K frequencies will be selected in the Block Module.
	      		4. `d_model`: Number of EEG channels in original data. Used in TimesNet Block Module.
	        	5. `d_ff`: Feedforward dimension of Inception Block in TimesNet.
	         	6. `num_kernels`: Number of kernels in the convolution of Inception Block in TimesNet Block Module.
	          	7. `num_heads`: Number of heads in multi-head attention layer in GAT.
	           	8. `ste_dropout`: Dropout rate for GAT.
		-BENDR Encoder:
	           	10. `encoder_h`: The hidden dimension of the encoder is designed to create "BENDR” features.
	           	11. `enc_width`: Encoder width, i.e. the kernel size of convolution layers.
	           	12. `enc_downsample`: Encoder downsample, i.e. the stride of the kernel in the convolution layers. In BENDR `enc_width` = `enc_downsample`.
	           	13. `enc_dropout`: Encoder dropout rate.
   		-MENDR Contextualizer:
   			14. `in_features`: Input features of contextualizer. Currently equal to `encoder_h.` Implicitly, the input layers of the transformers are `in_features` * 3 as per BENDR.
   			15. `dropout`: Dropout rate for contextualizer.
   			16. `position_encoder`: A seed for setting parameters of position encoder in contextualizer.
   			17. `epochs`: Epochs in the sense of MNE -- not training epochs! Segment time series into equal-length segments.
   		-Training Parameters:
   			18: `mask_span`: The number of time points masked in a mask. If random start time point + mask_span >= total # of time points, the number of time points in the mask is truncated to a length of the total - start time point.
   			19. `mask_rate`: For each time point, there is a `mask_rate` probability that the time point is the start of the masked sequence. Note that mask overlaps are allowed.
2. Downstream tasks
	- TBD
