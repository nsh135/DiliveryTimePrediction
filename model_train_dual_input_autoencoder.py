import pandas as pd
import sys
import argparse, importlib
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import concatenate
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
from misc import loss, read_tsv
import os
import tensorflow.keras.backend as K
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import pickle

physical_devices = tf.config.list_physical_devices('GPU') 
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)


## read only 5 rows
nrows = None
use_xgboost = False
MIXED_INPUT = True # build a mixed input model to separared important features
# important feature and onehot encoding features
# featureA = ['shipping_fee','carrier_min_estimate','carrier_max_estimate','item_price','quantity','weight','distance']

def prepare_store_places(train_file):
    #create train directory
    global train_path 
    train_file_name =   train_file.split('/')[-1].split('.')[0]
    train_path = "train/"+train_file_name + '/'  
    Path(train_path+ 'ckps').mkdir(parents=True, exist_ok=True)
    Path(train_path+ 'logs').mkdir(parents=True, exist_ok=True)
    Path(train_path+ 'saved_model').mkdir(parents=True, exist_ok=True)
    Path(train_path+ 'saved_model/autoencoder').mkdir(parents=True, exist_ok=True)


#Read data from tsv , only read 10 rows 
def csv2df(csv_file):
	"""
	csv_file: data input (.csv)
	return: pandas data frame 
	"""
	df = pd.read_csv(csv_file, sep=',', nrows= nrows, compression='gzip')
	return df

def autoencoder(input_dim, encoding_dim, X_train):
    norm_layer = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
    norm_layer.adapt(X_train.to_numpy())
    # This is our input image
    input_img = Input(shape=(input_dim,))
    # h0 = norm_layer(input_img)
    h1 =  Dense(512, activation='relu')(input_img)
    # h1 = BatchNormalization()(h1)
    h2 =  Dense(256, activation='relu')(h1)
    # h2 = BatchNormalization()(h2)
    h3 =  Dense(128, activation='relu')(h2)
    # h3 = BatchNormalization()(h3)
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu', name='encoded')(h3)
    h4 =  Dense(128, activation='relu')(encoded)
    # h4 = BatchNormalization()(h4)
    h5 =  Dense(256, activation='relu')(h4)
    # h5 = BatchNormalization()(h5)
    h6 =  Dense(512, activation='relu')(h5)
    # h6 = BatchNormalization()(h6)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(input_dim, activation='linear')(h6)
    # decoded = norm_layer(decoded)
    # This model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredLogarithmicError(
            reduction="sum_over_batch_size", name="mean_squared_logarithmic_error"))
    print("Autoencoder: ", autoencoder.summary())
    return autoencoder

def autoencoder_train(train_data,embedding_dim, n_epoch=200, batch_size=1000):# fit the keras model on the dataset
    """return encoder model"""
    
    input_dim = len(train_data.columns)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min', restore_best_weights=True)
        mcp_save = tf.keras.callbacks.ModelCheckpoint(train_path+'ckps/autoencoder.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        model = autoencoder(input_dim= input_dim, encoding_dim=embedding_dim, X_train=train_data)

    history  = model.fit(train_data, train_data, epochs=n_epoch, batch_size=batch_size, verbose=2, callbacks=[earlyStopping, mcp_save], validation_split=0.20)
    model.save(train_path+'saved_model/autoencoder')

    ##save training loss to figure
    ax = pd.DataFrame(data=history.history).plot(figsize=(15, 7))
    ax.grid()
    _ = ax.set(title="Training loss and Val loss", xlabel="Epochs")
    _ = ax.legend(["Training loss", "Val loss"])
    fig = ax.get_figure()
    fig.savefig(train_path + 'training_autoencoder_loss.jpg')
    plt.close(fig) 
    print("---Done training autoencoder")
    return model


def multi_input_model( X_train,X_embeddings, featureA):
    """
    featureA: list important feature names for input A
    X_train (df) : training dataframe
    """
    strategy = tf.distribute.MirroredStrategy()
    print('---Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        #define a custom loss 
        def ebay_loss(y_true, y_pred):
            const_early = 0.4
            const_late = 0.6
            mask_early =  tf.cast(K.less(y_true, y_pred) , tf.float32)
            mask_late = tf.cast(K.less(y_pred, y_true) , tf.float32)
            return K.mean(const_early* mask_early*(y_pred- y_true) + const_late * mask_late*(y_true-y_pred) ) 
         
         # define the keras model
        if X_train is not None: #if train_data is not none-> add normalization layer
            #add normalization layer 
            norm_layerA = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
            norm_layerA.adapt(X_train[featureA].to_numpy())
            norm_layerB = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
            norm_layerB.adapt(X_embeddings)

        inputA_dim  = len(featureA)  
        inputB_dim = X_embeddings.shape[1]
        print("---DEBUG: inputA dim : ", inputA_dim)
        print("---DEBUG: inputB_dim  : ", inputB_dim)
        # define two sets of inputs
        inputA = Input(shape=(inputA_dim,))
        inputB = Input(shape=(inputB_dim,))

        # the first branch operates on the first input
        x = norm_layerA(inputA)
        x = Dense(128,kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),bias_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l2(1e-5), activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dense(64,kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),bias_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l2(1e-5), activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dense(32,kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),bias_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l2(1e-5), activation="relu")(x)
        x = Model(inputs=inputA, outputs=x)
        # the second branch opreates on the second input
        y = norm_layerB(inputB)  
        y = Dense(128,kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),bias_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l2(1e-5), activation="relu")(y)
        y = BatchNormalization()(y)
        y = Dense(64,kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),bias_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l2(1e-5), activation="relu")(y)
        y = BatchNormalization()(y)
        y = Dense(32,kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),bias_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l2(1e-5), activation="relu")(y)
        y = Model(inputs=inputB, outputs=y)
        # combine the output of the two branches
        combined = concatenate([x.output, y.output])
        # apply a FC layer and then a regression prediction on the
        # combined outputs
        z = Dense(128,kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),bias_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l2(1e-5), activation="relu")(combined)
        z = BatchNormalization()(z)
        z = Dense(64,kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),bias_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l2(1e-5),activation="relu")(z)
        z = BatchNormalization()(z)
        z = Dense(32,kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),bias_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l2(1e-5),activation="relu")(z)
        z = BatchNormalization()(z)
        z = Dense(1, activation="relu")(z)
        # our model will accept the inputs of the two branches and
        # then output a single value
        model = Model(inputs=[x.input, y.input], outputs=z)
        model.compile(loss=ebay_loss,
                        optimizer=tf.keras.optimizers.Adam(0.0005))
    return model

def model_train(model,X_train,X_embeddings,y_train, n_epoch, batch_size):# fit the keras model on the dataset
    """Train DNN model 
    
    Args:
        model ([type]): linear regression model
        n_epoch (int): num of epochs
        batch_size: batch size

    Returns:
        model : keras model
    """
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=0, mode='min', restore_best_weights=True)
    mcp_save = tf.keras.callbacks.ModelCheckpoint(train_path+'ckps/ckpt.hdf5', save_best_only=True, monitor='val_loss', mode='min')

    train_data = [X_train[featureA], X_embeddings ]

    y_train = y_train.to_numpy().reshape((-1))
    history  = model.fit(train_data, y_train, epochs=n_epoch, batch_size=batch_size, verbose=2, callbacks=[earlyStopping, mcp_save], validation_split=0.20)
    model.save(train_path+'saved_model')
    
    ##save training loss to figure
    ax = pd.DataFrame(data=history.history).plot(figsize=(15, 7))
    ax.grid()
    _ = ax.set(title="Training loss and Val loss", xlabel="Epochs")
    _ = ax.legend(["Training loss", "Val loss"])
    fig = ax.get_figure()
    fig.savefig(train_path + 'training_loss.jpg')
    plt.close(fig) 
    return model



def evaluate(model,encoder_model,  X_test, y_test): # evaluate the keras model
    """
    evaluate model based on mean sqr error
    """
    X_test_embeddings = encoder_model.predict(X_test)
    test_data = [X_test[featureA], X_test_embeddings ]
  
    y_pred = np.rint(model.predict(test_data, batch_size=12800)[:,0])
    y_test = y_test.to_numpy().reshape((-1))
    print("y_test", y_test)
    print("y_pred", y_pred)

    print("Test with python loss")
    LOSS = loss(y_test, y_pred)
    print('Evaluated result:', y_pred)
    print("LOSS:", LOSS)
    np.savetxt(train_path+'logs/LOSS.out', [LOSS] ) 
    return y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training a DNN model')
    parser.add_argument('--train_file', dest='train_file', type=str, help='Path to trianing file (.csv)',default='data/processed_data/trainSet1.csv' )
    parser.add_argument('--test', dest='test', type=bool, help='testing only, load model from saved model', default=False)
    parser.add_argument('--epoch', dest='n_epoch', type=int, help='number of epochs, default 50', default=50)
    parser.add_argument('--batch_size', dest='batch_size', type=int, help='batch_size, default 1024', default=1024)
    parser.add_argument('--trainSetID', dest='trainSetID', type=str, help='train set id ', default='10')
    parser.add_argument('--train_autoencoder', dest='train_autoencoder', type=bool, help='training autoencoder?', default=False)
    parser.add_argument('--submit', dest='submit', type=bool, help='generate submit file?', default=False)
    args = parser.parse_args()
    
    #prepare training folder
    prepare_store_places(args.train_file)

    ## paramaters
    BATCH_SIZE =  args.batch_size
    N_EPOCH =  args.n_epoch
    
    print("**************Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    #get data file 
    print("Reading file..")
    df = csv2df(args.train_file)
    print("shape: ", df.shape)
    df = df.dropna()
    print("shape after drop NA: ", df.shape)
    

    #prepare dataset   
    # split to train and test 
    print("splitting data ... ")
    train, test = train_test_split(df, test_size=0.34, shuffle=True, random_state=10)
    X_train = train.loc[:, train.columns != 'target_from_order_placement']
    y_train = train[['target_from_order_placement']]
    X_test = test.loc[:, test.columns != 'target_from_order_placement']
    y_test = test[['target_from_order_placement']]

    global featureA 
    featureA = X_train.columns

    # show data samples
    print("data samples and data types !!!!")
    print('\n',X_test.head())
    print('\n',y_test.head())
    print('\n',X_test.head().dtypes)
    print('\n',y_test.head().dtypes)
    
    ### read quiz 
    quiz_cache = 'data/processed_data/x_record_number_quiz_convert_{}.pkl'.format(args.trainSetID)
    if os.path.exists(quiz_cache):
        with open(quiz_cache, 'rb') as handle:
            x,record_number = pickle.load(handle)
    else:
        convert = importlib.import_module('preprocessing.convert'+args.trainSetID)
        df_quiz = read_tsv("data/eBay_ML_Challenge_Dataset_2021/eBay_ML_Challenge_Dataset_2021_quiz.tsv", nrows=nrows)
        x, _, record_number = convert.process(df_quiz, no_target=True, record_number=True)
        with open(quiz_cache, 'wb') as handle:
            pickle.dump((x,record_number), handle)

    big_X_train = pd.concat([ X_train, x],axis=0) #concatenate X_train and quiz
    # scaler = StandardScaler()
    # big_X_train = pd.DataFrame(scaler.fit_transform(big_X_train))
    # big_X_train= (big_X_train-big_X_train.mean())/big_X_train.std()
    print("---big_X_train :",big_X_train.shape )
    # print("--- Need top uncomment and change Xtest to x")
    # Get auto encoder 
    if args.train_autoencoder:
        autoencoder = autoencoder_train(pd.concat([ X_train.head(50000), x.head(150000)],axis=0),embedding_dim=32, n_epoch=300, batch_size=4000)
    else: 
        autoencoder = tf.keras.models.load_model(train_path + "ckps/" + 'autoencoder.hdf5', compile=False)

    # get embedding from encoder
    print("---Get X_embeddings from X_train")
    encoder = Model(autoencoder.input, autoencoder.get_layer('encoded').output)
    
    ## get main model
    if args.test:
        print("Testing only, loading model from {}saved_model".format(train_path))
        #load from final model
        model = tf.keras.models.load_model(train_path + "saved_model", compile=False)
        #load from checkpoint
        # model = tf.keras.models.load_model(train_path + "ckps/" + 'ckpt.hdf5', compile=False)
    else:
        big_X_embeddings = encoder.predict(big_X_train,  batch_size = 200000)
        X_embeddings = encoder.predict(X_train,  batch_size = 200000)
        print("---big_X_embeddings.shape", big_X_embeddings.shape)
        print("---X_embeddings.shape", X_embeddings.shape)
            # Build a new DNN model
        model = multi_input_model( X_train=big_X_train, X_embeddings = big_X_embeddings, featureA = featureA)
        # training model 
        print("Start training a new model.")
        print(model.summary())
        # training model
        model_train(model,X_train,X_embeddings, y_train, N_EPOCH, BATCH_SIZE)
    
    print("Start evaluation Deep NN model ..... ")
    y_pred = evaluate(model, encoder, X_test, y_test) 
    print("Done Evaluation.")
   
    if args.submit:
        # load data and process
        convert = importlib.import_module('preprocessing.convert'+args.trainSetID)
        df = read_tsv("data/eBay_ML_Challenge_Dataset_2021/eBay_ML_Challenge_Dataset_2021_quiz.tsv", nrows=nrows)
        x, _, record_number = convert.process(df, no_target=True, record_number=True)
        print("Test data shape:", x.shape)
        x_embeddings = encoder.predict(x,  batch_size = 200000)
        x_quiz = [x[featureA], x_embeddings ] 
        # predict in days
        print("predicting ...")
        y_dnn_pred = model.predict(x_quiz, batch_size = 20000)[:,0].reshape((-1))
        days = np.rint(y_dnn_pred)
        print("!!!!Predicted values: ", days)
        # convert days to date
        print("converting days to dates ... ")
        delivery_dates = convert.delivery_days_to_date(df, days)
        prediction_df = pd.concat([record_number,delivery_dates], axis=1)
        print(prediction_df.head())
        # write file according to requirements
        print("writing output ... ")
        prediction_df.to_csv("submissions/xgboost_DNN.tsv.gz".format(args.trainSetID), sep='\t', index=False, header=False, compression='gzip')

    

