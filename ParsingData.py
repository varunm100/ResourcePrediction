import datetime
from monthdelta import monthdelta

XVal = []
YVal = []

FileData = open('NumData.txt', 'r').readlines()

IntialRelativeDate = datetime.date(2000,1,18)
i = 0
a = 0.0
for SingleLine in FileData:
	XVal.append(float(SingleLine.split()[0]))
	YVal.append(float(SingleLine.split()[1]))
	FinalWritingString = ''
	OneTimeDelta = datetime.timedelta(days=i)
	i = i + 1
	# DateCount = IntialRelativeDate + monthdelta(int(SingleLine.split()[0]))
	DateCount = IntialRelativeDate + OneTimeDelta
	DateString = DateCount.strftime("%Y-%m-%d")
	WritingFile = open('LSTMData.txt','a')
	WritingFile.write(str(a) + ' ' + SingleLine.split()[1] + '\n')
	a = a + 0.1
	WritingFile.close()

print('Done!!')