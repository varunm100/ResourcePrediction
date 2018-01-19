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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import h5py
import glob

XValPredict = [214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230]
#XValPredict = [218]
XVal = []
YVal = []
XValPredictStatic = XValPredict
XValStatic = []
YValStatic = []
MainSequence = []
XValTemp = []
NumEpochs = 2000
NumHiddenNuerons = 20
BatchSize = 5

def UpdateData():
	FileData = open('LSTMData.txt', 'r').readlines()
	for SingleLine in FileData:
		XVal.append(float(SingleLine.split()[0]))
		YVal.append(np.log(float(SingleLine.split()[1])))
		XValStatic.append(float(SingleLine.split()[0]))
		YValStatic.append(float(SingleLine.split()[1]))

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
	XValPredict[:] = [(i - 134)*0.1 for i in XValPredict]
	SeqCounter = []
	for i in range(len(XValPredict)):
		TempArr = []
		TempArr.append(XValPredict[i])
		TempArr.append(0.0)
		SeqCounter.append(TempArr)
	SeqCounter = array(SeqCounter)
	X, y = SeqCounter[:, 0], SeqCounter[:, 1]
	X = X.reshape((len(X), 1, 1))
	return X

def TrainModelASave():
	model = Sequential()
	model.add(LSTM(NumHiddenNuerons, return_sequences=True, input_shape=(1,1), activation='relu'))
	model.add(LSTM(NumHiddenNuerons, return_sequences=True, activation='relu'))
	model.add(LSTM(NumHiddenNuerons, activation='relu'))
	#model.add(Dropout(1, activation='relu'))
	model.add(Dense(1, activation='relu'))
	model.compile(loss='logcosh', optimizer='adam')
	X,y = ParseTrain()
	model.fit(X, y, epochs=NumEpochs, batch_size = BatchSize,verbose=0)
	model.save('RNNModels/model11.h5')
	print('Model Saved!')

def LoadModelAPredict():
	model = load_model('RNNModels/model9.h5')
	y = model.predict(ParseXVal(), verbose=0)
	print(np.exp(y))

def LoadAllModels():
	FileNamePath = glob.glob('RNNModels/*.h5')
	XValFormatted = ParseXVal()
	k = 0
	for Path in FileNamePath:
		k+=1
		print('-------------------')
		model = load_model(Path)
		print(Path)
		ModelResults = np.exp(model.predict(XValFormatted, verbose=0))
		print(ModelResults)
		print(model.summary())
		print('NumLayers: ' + str(len(model.layers)))
		DeltaYd = np.ravel(ModelResults).tolist()
		plt.subplot(2,5,k)
		plt.plot(XValStatic+XValPredictStatic, YValStatic+DeltaYd, color='blue', linewidth=3.0)
		plt.plot(XValPredictStatic, DeltaYd, color='red', linewidth=3.0)
		plt.subplot(2,5,k).set_title(str(Path))

	bluePath = mpatches.Patch(color='blue', label='Original')
	redPatch = mpatches.Patch(color='red', label='Predicted')
	plt.legend(handles=[bluePath, redPatch],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.show()

UpdateData()
# GenerateSequence()
# TrainModelASave()
# LoadModelAPredict()
LoadAllModels()