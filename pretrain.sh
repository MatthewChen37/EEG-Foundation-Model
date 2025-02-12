python pretrain.py --input_dir=/home/hice1/mchen439/scratch/eegfoundationmodeldata \
 --input_dir_2=/home/hice1/mchen439/scratch/eegfoundationmodeldata2 \
 --train_frac=0.8 \
 --val_frac=0.2 \
 --learning_rate=0.001\
 --batch_size=256 \
 --training_epochs=5 \
 --ckpt_dir=./checkpoint