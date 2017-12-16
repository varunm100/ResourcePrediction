import logging
import warnings
import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pylab as plt

logging.getLogger('fbprophet').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

def InitilizeModelFit(NumXpredict):
    global df
    global m
    global future
    global forecast
    df = pd.read_csv('NewNumData.csv')
    df['y'] = np.log(df['y'])
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=NumXpredict)
    forecast = m.predict(future)
    #forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
    print(np.exp(forecast['yhat']).tail(NumXpredict))
    print(np.exp(forecast['yhat_lower']).tail(NumXpredict))
    print(np.exp(forecast['yhat_upper']).tail(NumXpredict))

def GraphPrediction():
    m.plot(forecast)
    m.plot_components(forecast)
    plt.show()

InitilizeModelFit(7)
GraphPrediction()