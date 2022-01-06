import pandas as pd
import sys
import argparse
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

physical_devices = tf.config.list_physical_devices('GPU') 
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)


## read only 5 rows
nrows = None
MIXED_INPUT = False # build a mixed input model to separared important features


def prepare_store_places(train_file):
    #create train directory
    global train_path 
    train_file_name =   train_file.split('/')[-1].split('.')[0]
    train_path = "train/"+train_file_name + '/'  
    Path(train_path+ 'ckps').mkdir(parents=True, exist_ok=True)
    Path(train_path+ 'logs').mkdir(parents=True, exist_ok=True)
    Path(train_path+ 'saved_model').mkdir(parents=True, exist_ok=True)


#Read data from tsv , only read 10 rows 
def csv2df(csv_file):
	"""
	csv_file: data input (.csv)
	return: pandas data frame 
	"""
	df = pd.read_csv(csv_file, sep=',', nrows= nrows, compression='gzip')
	return df

def DNN_model(input_dim, n_layers,X_train=None):
    """[summary]

    Args:
        input_dim (int): input dimentions
        n_layers (int): DNN model's number of hidden layers
        X_train: for normalization layer to calculate mean and std (optional)
                if X_train is None: there will not be a normalization layer,
                data should be normalized before fed into the model
        
    Returns:
        Keras model: new model
    """    
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        #define a custom loss 
        def ebay_loss(y_true, y_pred):
            const_early = 0.4
            const_late = 0.6
            mask_early =  tf.cast(K.less(y_true, y_pred) , tf.float32)
            mask_late = tf.cast(K.less(y_pred, y_true) , tf.float32)
            return const_early*K.mean( mask_early*(y_pred- y_true) ) + const_late * K.mean(mask_late*(y_true-y_pred) ) 

        # define the keras model
        model = Sequential()
        if X_train is not None: #if train_data is not none-> add normalization layer
            #add normalization layer 
            norm_layer = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
            norm_layer.adapt(X_train.to_numpy())
            print("Normalization layer mean and var:",norm_layer.mean.numpy())
            model.add(norm_layer)
    
        model.add(Dense(500, input_dim=input_dim, activation='relu'))
        for i in range(n_layers-1):
            model.add(Dense(300, activation='relu'))
        model.add(Dense(1,activation = 'relu')) #single linear output
        
        # compile the keras model
        if True: # if using custom loss
            model.compile(loss=ebay_loss,
                        optimizer=tf.keras.optimizers.Adam(0.001))
        else:  #if not using custom loss
            model.compile(loss='mean_squared_error',
                        optimizer=tf.keras.optimizers.Adam(0.001))
    return model

def multi_input_model( X_train, featureA):
    """
    featureA: list important feature names for input A
    X_train (df) : training dataframe
    """
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        #define a custom loss 
        def ebay_loss(y_true, y_pred):
            const_early = 0.4
            const_late = 0.6
            mask_early =  tf.cast(K.less(y_true, y_pred) , tf.float32)
            mask_late = tf.cast(K.less(y_pred, y_true) , tf.float32)
            return const_early*K.mean( mask_early*(y_pred- y_true) ) + const_late * K.mean(mask_late*(y_true-y_pred) ) 
         
         # define the keras model
        if X_train is not None: #if train_data is not none-> add normalization layer
            #add normalization layer 
            norm_layerA = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
            norm_layerA.adapt(X_train[featureA].to_numpy())
            norm_layerB = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
            norm_layerB.adapt(X_train[X_train.columns.difference(featureA)].to_numpy())


        inputA_dim  = len(featureA)  
        inputB_dim = len(X_train.columns) -inputA_dim 
        print("DEBUG: inputA dim : ", inputA_dim)
        print("DEBUG: inputB_dim  : ", inputB_dim)
        # define two sets of inputs
        inputA = Input(shape=(inputA_dim,))
        inputB = Input(shape=(inputB_dim,))
        # the first branch operates on the first input
        x = norm_layerA(inputA)
        x = Dense(1024, activation="relu")(x)
        x = Dense(512, activation="relu")(x)
        x = Dense(256, activation="relu")(x)
        x = Model(inputs=inputA, outputs=x)
        # the second branch opreates on the second input
        y = norm_layerA(inputB)
        y = Dense(1024, activation="relu")(y)
        y = Dense(512, activation="relu")(y)
        y = Dense(256, activation="relu")(y)
        y = Model(inputs=inputB, outputs=y)
        # combine the output of the two branches
        combined = concatenate([x.output, y.output])
        # apply a FC layer and then a regression prediction on the
        # combined outputs
        z = Dense(128, activation="relu")(combined)
        z = Dense(64, activation="relu")(z)
        z = Dense(1, activation="relu")(z)
        # our model will accept the inputs of the two branches and
        # then output a single value
        model = Model(inputs=[x.input, y.input], outputs=z)
        model.compile(loss=ebay_loss,
                        optimizer=tf.keras.optimizers.Adam(0.001))
    return model

def model_train(model,X_train,y_train, n_epoch, batch_size):# fit the keras model on the dataset
    """Train DNN model 
    
    Args:
        model ([type]): linear regression model
        n_epoch (int): num of epochs
        batch_size: batch size

    Returns:
        model : keras model
    """
    y_train = y_train.to_numpy().reshape((-1))
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min', restore_best_weights=True)
    mcp_save = tf.keras.callbacks.ModelCheckpoint(train_path+'ckps/ckpt.hdf5', save_best_only=True, monitor='val_loss', mode='min')

    history  = model.fit(X_train, y_train, epochs=n_epoch, batch_size=batch_size, verbose=2, callbacks=[earlyStopping, mcp_save], validation_split=0.16)
    
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

def evaluate(model, X_test, y_test): # evaluate the keras model
    """
    evaluate model based on mean sqr error
    """
    y_pred = np.rint(model.predict(X_test, batch_size=128000)[:,0])
    y_test = y_test.to_numpy()
    LOSS = loss(y_test, y_pred)
    print('Evaluated result:', y_pred)
    print("LOSS:", LOSS)
    np.savetxt(train_path+'logs/LOSS.out', LOSS ) 
    np.savetxt(train_path+'logs/y_pred.out', y_pred ) 
    return y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training a DNN model')
    parser.add_argument('--train_file', dest='train_file', type=str, help='Path to trianing file (.csv)',default='data/processed_data/trainSet1.csv' )
    parser.add_argument('--test', dest='test', type=bool, help='testing only, load model from saved model', default=False)
    parser.add_argument('--epoch', dest='n_epoch', type=int, help='number of epochs, default 50', default=50)
    parser.add_argument('--batch_size', dest='batch_size', type=int, help='batch_size, default 1024', default=1024)
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
    
    # onehot encoding is done in data_processing

    #prepare dataset   
    # split to train and test 
    print("splitting data ... ")
    train, test = train_test_split(df, test_size=0.17, shuffle=True, random_state=10)
    X_train = train.loc[:, train.columns != 'target_from_order_placement']
    y_train = train[['target_from_order_placement']]
    X_test = test.loc[:, test.columns != 'target_from_order_placement']
    y_test = test[['target_from_order_placement']]

    # show data samples
    print("data samples and data types !!!!")
    print('\n',X_test.head())
    print('\n',y_test.head())
    print('\n',X_test.head().dtypes)
    print('\n',y_test.head().dtypes)
    
    # important feature and onehot encoding features
    featureA = list(X_train.columns)
   

    # get the model
    if args.test:
        print("Testing only, loading model from {}saved_model".format(train_path))
        model = tf.keras.models.load_model(train_path + "saved_model", compile=False)
        # model = tf.keras.models.load_model(train_path + "ckps/" + '.mdl_wts.hdf5', compile=False)
    else:
        # Build a new DNN model
        if MIXED_INPUT: #build model with separated emphasized feature inputs 
            # Build a new model for mixed inputs
            model = multi_input_model( X_train=X_train, featureA = featureA)
        else:  # build normal DNN
            model = DNN_model(input_dim= len(X_train.columns)-1, n_layers=5, X_train=X_train)

        # training model 
        print("Start training a new model.")
        print(model.summary())
        model_train(model,X_train,y_train, N_EPOCH, BATCH_SIZE)
    
    print("Data types after onehot")
    print('\n',X_test.head().dtypes)
    #evaluation 
    print("Start evaluation ..... ")
    y_pred = evaluate(model, X_test, y_test) 
    print("Done Evaluation.")

    #make corrolation matrix to analize
    if not  args.test:
        y_pred_df = pd.DataFrame(y_pred, columns=['y_pred'])
        print("y_test", y_test['target_from_order_placement'].to_numpy())
        print("y_pred", y_pred)
        err_df = pd.DataFrame(y_pred-y_test['target_from_order_placement'].to_numpy(), columns=['error'])
        X_test = pd.concat([X_test, y_test['target_from_order_placement'], y_pred_df,err_df], axis=1) 
        corr = X_test.corr()
        corr.to_csv(train_path + 'logs/corr_matrix.csv', index=False)

