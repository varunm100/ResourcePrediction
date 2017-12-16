import pandas as pd
from pandas import datetime
import numpy as np
import datetime as dt
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

PredictN = 7

XVal = []
YVal = []

DeltaXVal = []
DeltaYVal = []
def UpdateData():
	FileData = open('NumData.txt', 'r').readlines()
	for SingleLine in FileData:
		XVal.append(float(SingleLine.split()[0]))
		YVal.append(float(SingleLine.split()[1]))
def GetPrediction():
	NumResources = YVal
	start = dt.datetime.strptime("1 Sep 17", "%d %b %y")
	daterange = pd.date_range(start, periods=len(XVal))
	table = {"NumResources": NumResources, "Date": daterange}
	data = pd.DataFrame(table)
	data.set_index("Date", inplace=True)

	order = (1, 2, 1)
	model = ARIMA(data, order)
	model = model.fit(trend='nc',disp=0)
	#print(model.summary())
	#print (data)
	StartDate = datetime(2017, 11, 10)
	EndDate = datetime(2017,11, 18)
	# FinalForcast = model.predict(start = StartDate, end = EndDate)
	# print(FinalForcast)
	FinalForcast = model.forecast()
	return FinalForcast[0]
def PredictNSteps(NSteps):
	for i in range(0,NSteps):
		CurrentPrediction = GetPrediction()
		print(str(XVal[-1] + 1) + ' : ' + str(CurrentPrediction))
		DeltaXVal.append(XVal[-1] + 1)
		DeltaYVal.append(float(CurrentPrediction))
		XVal.append(XVal[-1] + 1)
		YVal.append(float(CurrentPrediction))

	plt.bar(XVal,YVal)
	plt.bar(DeltaXVal, DeltaYVal, color='red')
	plt.show()

def GraphCurrentData():
	plt.bar(XVal,YVal)
	plt.show()

UpdateData()
PredictNSteps(PredictN)

#-------Results-------
#216.0 : [ 96.02934496]
#217.0 : [ 94.36983635]
#218.0 : [ 92.71041697]
#219.0 : [ 91.05106275]
#220.0 : [ 89.39175642]