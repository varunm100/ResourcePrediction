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

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
NewYVal = smooth(YVal,2)
plot(XVal, YVal,'b-')
plot(XVal, NewYVal, 'g-', lw=2)
# plot(XVal, smooth(smooth(YVal,2),2), 'r-', lw=2)
# plot(XVal, smooth(smooth(smooth(YVal,2),2),2), 'c-', lw=2)
# plot(XVal, smooth(smooth(smooth(smooth(YVal,2),2),2),2), 'k-', lw=2)
show()

def WriteToFile():
	for i in range(0,79):
		WritingFile = open('NumData010.txt','a')
		WritingString = ''
		WritingString = str(i+134) + ' ' + str(NewYVal[i]) + '\n'
		WritingFile.write(WritingString)
		WritingFile.close() 
WriteToFile()
