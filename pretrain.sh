python pretrain.py --input_dir=/home/hice1/mchen439/scratch/eegfoundationmodeldata \
 --input_dir2=/home/hice1/mchen439/scratch/eegfoundationmodeldata2 \
 --train_frac=0.9 \
 --val_frac=0.1 \
 --learning_rate=0.0001 \
 --batch_size=256 \
 --training_epochs=2 \
 --ckpt_dir=./checkpoint