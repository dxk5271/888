import streamlit as st
import pandas as pd
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import scipy.io as sio
from scipy.io import loadmat
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Add
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPool1D, ZeroPadding1D, LSTM, Bidirectional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


st.title("Capstone 888")

option = st.selectbox(
     'Select a Model',  
    ('Neural Network', 'Logistic Regression', 'KNN'))

if st.button('Run Model'):
    with st.spinner('Wait for it...'):
        test = pd.read_parquet('ori_test_0218.parquet')
        train = pd.read_parquet('capstone888/ori_train_0218.parquet')
        class_count_0, class_count_1 = train['target'].value_counts()

# Separate class
        class_0 = train[train['target'] == 0]
        class_1 = train[train['target'] == 1]

        class_0_under = class_0.sample(class_count_1)

        train = pd.concat([class_0_under, class_1], axis=0)


        X_train = train.loc[:, train.columns != 'target']
        y_train = train.loc[:, train.columns == 'target']

        X_test = test.loc[:, test.columns != 'target']
        y_test = test.loc[:, test.columns == 'target']


        scaler = StandardScaler()
        # fit scaler on data
        scaler.fit(X_train)
        # apply transform
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)


        #X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

        #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

        print(len(X_train))
        print(len(X_test))
        print(len(X_val))


        #X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)





        
        model = keras.Sequential([
        layers.Dense(input_dim = 73, units = 32, activation = "relu"),
        layers.Dense(units= 64, activation = "relu"),
        layers.Dropout(0.2),
        layers.Dense(units= 256, activation = "relu"),
        layers.Dropout(0.2),
        layers.Dense(units= 128, activation = "relu"),
        layers.Dropout(0.3),
        layers.Dense(units= 32, activation = "relu"),
        layers.Dense(units=1, activation = "sigmoid")])
        model.summary()
        
        
        # Compile the model with categorical crossentropy loss function and the Adam optimizer
        model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
        # Train the model using the fit() method
        history = model.fit(X_train, y_train, epochs=25, batch_size=128, validation_data=(X_test, y_test))




        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss']) 
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.show()

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='lower right')
        plt.show()

        # Evaluate the model
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f'Test loss: {test_loss} - Test accuracy: {test_acc}')






   


