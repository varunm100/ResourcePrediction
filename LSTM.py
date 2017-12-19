from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy as np
from numpy import array
import h5py

XValPredict = [8.0,8.1,8.2,8.3,8.4,8.5,8.6]
XVal = []
YVal = []
MainSequence = []
NumEpochs = 1000
XValTemp = []

def UpdateData():
	FileData = open('LSTMData.txt', 'r').readlines()
	for SingleLine in FileData:
		XVal.append(float(SingleLine.split()[0]))
		YVal.append(np.log(float(SingleLine.split()[1])))

def GenerateSequence():
	for i in range(len(XVal)):
		TempArr = []
		TempArr.append(XVal[i])
		TempArr.append(YVal[i])
		MainSequence.append(TempArr)

def ParseTrain():
	seq = MainSequence
	seq = array(seq)
	X, y = seq[:, 0], seq[:, 1]
	X = X.reshape((len(X), 1, 1))
	XValTemp = X
	return X,y

def ParseXVal():
	SeqCounter = []
	for i in range(len(XValPredict)):
		TempArr = []
		TempArr.append(XValPredict[i])
		TempArr.append(0.0)
		SeqCounter.append(TempArr)
	SeqCounter = array(SeqCounter)
	X, y = SeqCounter[:, 0], SeqCounter[:, 1]
	X = X.reshape((len(X), 1, 1))
	print(X)
	return X

def TrainModelASave():
	model = Sequential()
	model.add(LSTM(16, input_shape=(1,1)))
	model.add(Dense(1, activation='relu'))
	model.compile(loss='mse', optimizer='adam')
	X,y = ParseTrain()
	model.fit(X, y, epochs=NumEpochs, shuffle=False, verbose=0)
	model.save('RNNModels/model4.h5')
	print('Model Saved!')

def LoadModelAPredict():
	model = load_model('RNNModels/model3.h5')
	y = model.predict(ParseXVal(), verbose=0)
	print(np.exp(y))

UpdateData()
GenerateSequence()
#TrainModelASave()
LoadModelAPredict()