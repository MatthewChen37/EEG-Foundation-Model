# Next Steps 

[MENDR Paper](https://arxiv.org/abs/2508.04956)

[AAAI Reviews](https://openreview.net/forum?id=2UYLqRJuWl)

## Discussion

Although MENDR was not accepted to AAAI, we plan on continuing to work on the paper and resubmitting it elsewhere. Here is a list of the main criticisms of MENDR:

1. MENDR is a "collage of ideas" --> MENDR needs to be a coherent idea.
2. Better presentation of figures in paper.
3. Tiny/Large model confusion.
4. TODO: Reproduce competitors papers.
5. Why not just MENDR Tiny since it performed better?
6. Need ablation studies.
7. Split downstream into trian/val/test splits.

## Direction

MENDR has arguably been one of the most difficult projects I have undertaken and I have sank thousands of hours since the start of 2025 to the project. I've realized in research that there exists the issue of doing "too much" in a single project. MENDR is a good example of this. There are several contributions in MENDR that it is impossible to perform experiments, ablation studies, etc. to justify all the contributions. Instead, I've decided to split the project into two parts as follows:

    1. Developing generalizable EEG representations
        - Novel Wavelet + GNN Transformer architecture for learning EEG representations
        - "Sanitizing" the EEG representations of cofounding variables such as Subject, Session, Sex, Age during pretraining
        - Evaluating on downstream tasks and visualizing the representations

    2. MENDR and the Manifold Transformer
        - EEG Manifold Transformer architecture
        - Riemmanian Self Supervised Learing (RSL) for learning EEG manifolds
        - Evaluating on downstream tasks and visualizing the manifolds using novel manifold visualization techniques

I consider each contribution to be a separate project as, for example, the [CBraMod](https://arxiv.org/abs/2412.07236) paper has similar contributions to `1`. `2` is extremely novel and thus automatically qualifies as a separate project. A principle I am experimenting with is the "rule of three" in research, where a properly scoped work should typically have three contributions.

### Elaborating on `1`:

- Very little research has been done on preprocessing the EEG data into Wavelets before passing them into an EEG foundation model. By decomposing the EEG data into Wavelets, we break the data into a series of frequency bands which correspond to widely studied EEG phenomena such as the alpha, beta, and gamma bands in neuroscience. Thus, the decomposition provides interpretability that could help a neuroscientist understand the underlying phenomena from the lense of known human brain rythms. A GNN is also used because intuitively, EEG electrodes are connected in a graph-like structure. Thus, a GNN is a natural choice for learning the EEG representations, and could provide explainability into how the model learns with respect to how the electrodes are spatially connected.

- A major problem in EEG decoding is the extreme variability in the EEG data; different subjects have different EEG dynamics and different electrode locations. [A whole challenge from NeurIPS was held solely focusing on this problem](https://www.codabench.org/competitions/9975/). Based on the result of the challenge, it seems like the problem is still unsolved. [Based on literature, the largest source of variability is between subjects, followed by the electrode systems.](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2017.00150/full). While digging through the [LaBraM paper](https://arxiv.org/abs/2405.18765), I found that the authors based their Vector-Quantized Variational Autoencoder (VQVAE) on the original [VQ-VAE paper](https://arxiv.org/abs/1711.00937). A particularly interesting finding was that they were able to train the VQVAE to learn subject-independent audio waveform representations. [A cool demo can be found here](https://avdnoord.github.io/homepage/vqvae/). The authors follow a traditional encoder-decoder architecture, but they condition the decoder on the subject id (i.e. only pass the subject id to the decoder). Given how similar the audio modality is to the EEG modality, I have a strong suspicion that this is a promising direction to explore to also generate subject-independent EEG representations. We would essentially be "sanitizing" the subject-to-subject variance by passing the subject id to the decoder and only passing the EEG to the encoder. Further interesting directions include also condition on other aspects like session, sex, age, etc.

- The evaluate how good the sanitization is, we should expect to see in a UMAP visualization that similar events across different subjects are clustered together. Then, we further evaluate it on known downstream tasks like TUAB/TUEV.

### Elaborating on `2`:

- Note that we probably WON'T use the Wavelets for the manifold learning.

- Formally implement the EEG Manifold Transformer architecture by using the attention module from (MAtt)[https://arxiv.org/abs/2210.01986]. Redo the implementation by building on top of SPDLearn rather than implementing it myself. Will also contribute to SPDLearn in the meantime because Manifold learning is something I want to focus on in the future.

- Perform Riemannian Self Supervised Learning (RSL) on the manifolds. In the RSL, we propose a learnable SPD mask and add a small lemma for why it works. Examine how well it performs on downstream tasks like TUAB/TUEV. Try two different ways - only penalizing on the eigenvalues or penalizing on both the eigenvalues and eigenvectors.

- Include explainability into the model by visualizing how the manifold is transformed -- [I want to use this](https://github.com/thibaultdesurrel/riemannien_dimension_reduction). Also UMAP + Elliptical Visualizations too -- [see this](https://en.wikipedia.org/wiki/QR_algorithm#Finding_eigenvalues_versus_finding_eigenvectors). From the MENDR paper, you'll notice there are a lot of "embryo-like" UMAP visualizations for TUAB. I want to analyze how many the golden ratio (which is the shape of embryos) has any connection to UMAP because there doesn't seem to be any known connection.


### Actionable Next Steps

1. Ensure all data is on Georgia Tech's PACE.
2. Try reproducing BIOT/LaBraM/CbraMod on TUAB/TUEV.
3. Break this repo into two repos.