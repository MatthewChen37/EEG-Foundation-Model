# Training Plan 

Current state-of-the-art self supervised learning for learning deep signal representations, specifically [wav2vec 2.0](https://arxiv.org/abs/2006.11477) employ a self supervised loss:

$$L = L_{m} + \alpha L_{d}$$

where $L_{m}$ represents the loss from the contrastive task where some of the input features are masked and $L_{d}$ represents the codebook diversity loss. This loss is similarly employed in [BENDR](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2021.653659/full#B5). We want to argue that adding a decoder that decodes the embedded features back to reconstruct the original signal, essentially demonstrating that signal information is preserved in the embedding, helps improve explainability and will improve loss. In other words:

$$L_{MENDR} = L_{m} + \alpha L_{d} + \beta L_{recon}$$

$L_{recon}$ represents the reconstruction loss of the reconstructed signal from the decoder. We will first try MSE; however, other techniques do exist for comparing the original vs reconstructed biosignal. $\beta$ represents a tuneable hyperparameter similar to $\alpha$ that is adjusted to determine how much we want to penalize reconstruction loss.

1. Pretraining
	- Pretraining Parameter Documentation: 
		- MENDR Encoder:
		- MENDR Contextualizer:
		- Training Parameters:

2. Downstream tasks
	- TBD
