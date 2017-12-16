import numpy as np
from pylab import *

PredictXVals = [216, 217, 218, 219, 220]

XVal = []
YVal = []

FileData = open('NumData.txt', 'r').readlines()

for SingleLine in FileData:
	XVal.append(float(SingleLine.split()[0]))
	YVal.append(float(SingleLine.split()[1]))

(m,b) = polyfit(XVal, YVal, 1)
print([m,b])
yp = polyval([m,b],XVal)


plot(XVal, yp, color='red')
for i in PredictXVals:
	YValTemp = 0
	YValTemp = (m*i)+b
	XVal.append(i)
	YVal.append(YValTemp)
	print('X Value: ' + str(i) + ' - Y Value: ' + str(YValTemp))
scatter(XVal, YVal)
grid(True)
xlabel('Time (By Month ID)')
ylabel('# of Resources Required')
show()
# ------RESULTS------
# 	   216 - 116
# 	   217 - 118
# 	   218 - 119
# 	   219 - 121
# 	   220 - 122