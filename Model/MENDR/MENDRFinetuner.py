def MENDRFinetuner(BaseModelTrainer):
    def __init__(self, encoder, contextualizer, config, **kwargs):
        # Initialize temperature as a trainable parameter
        self.temp1 = torch.nn.Parameter(torch.tensor(0, requires_grad=True))
        self.temp2 = torch.nn.Parameter(torch.tensor(0, requires_grad=True))
        self.contrastive_finetune_loss = nn.CrossEntropyLoss()

        if config.multi_gpu:
            encoder = nn.DataParallel(encoder)
            contextualizer = nn.DataParallel(contextualizer)
            self.temp1 = nn.DataParallel(self.temp1)
            self.temp2 = nn.DataParallel(self.temp2)

        super(MENDRFinetuner, self).__init__(encoder=encoder, contextualizer=contextualizer,
			temp1=self.temp1, temp2=self.temp2, contrastive_finetune_loss=self.contrastive_finetune_loss, lr=config.learning_rate,
			l2_weight_decay=config.l2_weight_decay, metrics=dict(), ckpt_dir=config.ckpt_dir, **kwargs)

    def forward(self, data):
        relevant_bands = [data[band] for band in BANDS]
        inputs = dict(zip(BANDS, relevant_bands))


    

