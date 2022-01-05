### run virtual environment 
# deactivate current Virtual environment
deactivate

# run virtual environment for ebay_challenge
conda activate ebay


### read from raw input .tsv file, process data and write output to .csv file 
## python data_processing.py path/to/.tsv

python data_processing.py data/eBay_ML_Challenge_Dataset_2021/eBay_ML_Challenge_Dataset_2021_train.tsv


## train and evaluate model
CUDA_VISIBLE_DEVICES=1,2,3 python model_train.py --epoch 200 --batch_size 2048 --train_file 'data/processed_data/trainSet8.csv.gz'

# test previous dataset without training model again : include --test True 
CUDA_VISIBLE_DEVICES=1,2,3 python model_train.py --epoch 200 --batch_size 2048 --train_file 'data/processed_data/trainSet7.csv.gz' --test True

## generate prediction
python predict.py --train_dir train/TrainSet8 