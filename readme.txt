### run virtual environment 
# deactivate current Virtual environment
deactivate

# run virtual environment for ebay_challenge
conda activate ebay


### read from raw input .tsv file, process data and write output to .csv file 
## python data_processing.py path/to/.tsv

python data_processing.py data/eBay_ML_Challenge_Dataset_2021/eBay_ML_Challenge_Dataset_2021_train.tsv


## train and evaluate model
CUDA_VISIBLE_DEVICES=0,1,2,3 python model_train.py --epoch 200 --batch_size 20000 --train_file 'data/processed_data/trainSet11.csv.gz'

# test previous dataset without training model again : include --test True 
CUDA_VISIBLE_DEVICES=0,1,2,3 python model_train_dual_input_rank5.py --epoch 300 --batch_size 40000 --train_file 'data/processed_data/trainSet7.csv.gz' --test True

## generate prediction
python predict.py --train_dir train/TrainSet8 


####### Dual input model
# input is split into 2 , includings numeric important feature and onehot not important features
# important features are defined in model_train_dual_input file: featureA=['']

CUDA_VISIBLE_DEVICES=2,3 python model_train_dual_input.py --epoch 200 --batch_size 40960 --train_file 'data/processed_data/trainSet10.csv.gz' --test True
# predict
python predict_dual_input.py --trainSetID 10



####### USE Dual input autoencoder

# train autoencoder (--train_autoencoder True) -> train Main_model (--test False) -> Predict Quiz (--submit True):

CUDA_VISIBLE_DEVICES=0,1,2,3  python model_train_dual_input_autoencoder.py --epoch 200 --batch_size 40000 --train_file 'data/processed_data/trainSet10.csv.gz'  --trainSetID 10 --train_autoencoder True --submit True

# to use saved autoencoder -> remove:  --train_autoencoder 
# to use saved MainModel -> add:       --test True 
# to predict quiz file -> add:         --submit True

e.g., asummed you have autoencoder and main model trained and saved. Run test without create submit file. Add --test True
CUDA_VISIBLE_DEVICES=0,1,2,3  python model_train_dual_input_autoencoder.py --epoch 200 --batch_size 40000 --train_file 'data/processed_data/trainSet10.csv.gz'  --trainSetID 10 --test True

# if you want to create submit file without retrain all the models (models have been saved). add --test True --submit True
CUDA_VISIBLE_DEVICES=0,1,2,3  python model_train_dual_input_autoencoder.py --epoch 200 --batch_size 40000 --train_file 'data/processed_data/trainSet10.csv.gz'  --trainSetID 10 --test True --submit True
