import numpy as np
import numpy.polynomial.polynomial as poly
from scipy.interpolate import *
from matplotlib.pyplot import *

XVal = []
YVal = []
XPredictionVals = [216, 217, 218, 219, 220]

FileData = open('NumData.txt', 'r').readlines()
for SingleLine in FileData:
	XVal.append(float(SingleLine.split()[0]))
	YVal.append(float(SingleLine.split()[1]))
XVal = np.asarray(XVal)
YVal = np.asarray(YVal)
XPredictionVals = np.asarray(XPredictionVals)
coefs = poly.polyfit(XVal, YVal, 3)
plot(XVal,YVal, 'g')
ffit = poly.polyval(XVal, coefs)
plot(XVal, ffit)
ffit = poly.Polynomial(coefs)
XVal = np.concatenate((XVal, XPredictionVals), axis=0)
plot(XVal, ffit(XVal))
for i in XPredictionVals:
	print(str(i) + ' - ' + str(ffit(i)))
show()
# ------RESULTS------
# 216 - 110.675127603
# 217 - 109.705298119
# 218 - 108.554832254
# 219 - 107.219143535
# 220 - 105.693645484