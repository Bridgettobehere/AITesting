# this will be able to take any datasheet from NASDAQ and try to predict the stock price of that company
# dependencies
from __future__ import print_function
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, Callback
from sklearn.cross_validation import  train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
import requests
import pandas_datareader as web
import datetime

# HYPER PARAMETERS
look_back = 7
epochs = 1000
batch_size = 32

# helper for logging the history
class AccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

def get_data(stock_symbol):
    start = datetime.datetime(2008, 1, 1)
    end = datetime.datetime.now()
    print("Downloading historical quotes for " + str(stock_symbol) + "...")
    data = web.DataReader(stock_symbol, 'yahoo', start, end)
    print("Done")
    # save this data just incase
    data.to_csv('data/' + str(stock_symbol) + ".csv")
    prices = data.Close.values.astype('float32')

    # reshape to column vector
    prices = prices.reshape(len(prices), 1)
    return prices

def process_datasheet(file_path):
    data = pd.read_csv(file_path)
    # MARK: rearrange the data into a format that we will use in our model

    # reverse the data so that the first values come first (because 2008 came before 2018)
    data = data.iloc[::-1]

    # get the stored historical prices
    prices = data.close.values.astype('float32')
    # reshape so that it turns into a column vector
    prices = prices.reshape(len(prices), 1)
    return prices

def training_split(data, split):
    trainingSize = int(len(data) * float(split))
    testingSize = len(data - trainingSize)

    trainingSet, testingSet = data[0:trainingSize, :], data[trainingSize:len(data), :]

    return trainingSet, testingSet

# convert an array of values into a time series dataset
# in form
#                     X                     Y
# t-look_back+1, t-look_back+2, ..., t     t+1

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for index in range(len(dataset)-look_back-1):
        a = dataset[index:(index+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[index + look_back, 0])
    return np.array(dataX), np.array(dataY)

def train_reshape(trainX, testX):
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    return trainX, testX

def create_model(look_back):
    model = Sequential()

    # add the first Bidirectional LSTM layer
    model.add(Bidirectional(LSTM(4, return_sequences=True), input_shape=(look_back, 1)))
    model.add(Dropout(0.2))

    # add the second LSTM block
    model.add(Bidirectional(LSTM(8, return_sequences=True)))
    model.add(Dropout(0.2))

    # add the final LSTM block
    model.add(Bidirectional(LSTM(4, return_sequences=False)))

    # add the dense layer to output the number from our feature representations
    model.add(Dense(units=1))

    # compile the model with mse as the loss function and adamoptimizer as the optimizer
    model.compile(loss='mse', optimizer='adam')
    return model


def plot(prices, predictions, stock_symbol, prediction_title):
    plt.plot(prices, label="Actual")
    plt.plot(predictions, label=prediction_title)
    plt.title(str(stock_symbol) + " Actual Price vs. Predicted Price")
    plt.ylabel("Price (USD)")
    plt.xlabel("Time")
    plt.legend()
    plt.show()


def main():
    symbol = str(input("Stock Symbol (e.g AAPL): "))

    prices = get_data(symbol)

    # normalize our data
    normalizer = MinMaxScaler(feature_range=(0,1))
    prices = normalizer.fit_transform(prices)

    training, testing = training_split(prices, 0.60)

    trainX, trainY = create_dataset(training, look_back)
    testX, testY = create_dataset(testing, look_back)

    trainX, testX = train_reshape(trainX, testX)

    model = create_model(look_back)
    checkpoint = ModelCheckpoint("model_checkpoints/" + symbol + "_checkpoint.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    # --- debugging ------
    history = AccuracyHistory()
    # --------------------
    # train the model
    callbacks = [checkpoint, history]
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    # save the model to use later on
    model.save('saved_models/' + symbol + ".h5")

    trainPredictions = model.predict(trainX)
    testPredictions = model.predict(testX)

    trainPredictions = normalizer.inverse_transform(trainPredictions)
    trainY = normalizer.inverse_transform([trainY])
    testPredictions = normalizer.inverse_transform(testPredictions)
    testY = normalizer.inverse_transform([testY])

    trainingScore = math.sqrt(mean_squared_error(trainY[0], trainPredictions[:, 0]))
    testingScore = math.sqrt(mean_squared_error(testY[0], testPredictions[:, 0]))

    print("Training Score: %.5f RMSE" % (trainingScore))
    print("Testing Score: %.5f RMSE" % (testingScore))

    # unnormalize our prices for plotting
    prices = normalizer.inverse_transform(prices)
    trainingPlot = np.empty_like(prices)
    trainingPlot[:, :] = np.nan
    trainingPlot[look_back:len(trainPredictions)+look_back, :] = trainPredictions


    testingPlot = np.empty_like(prices)
    testingPlot[:, :] = np.nan
    testingPlot[len(trainPredictions)+(look_back*2)+1:len(prices)-1, :] = testPredictions

    plot(prices, trainingPlot, symbol, "Training Prediction")
    plot(prices, testingPlot, symbol, "Testing Prediction")


if __name__ == '__main__':
    main()
