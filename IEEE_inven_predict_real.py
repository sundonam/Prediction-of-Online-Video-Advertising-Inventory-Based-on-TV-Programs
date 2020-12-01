#!/home/centos/anaconda3/bin/python3.6
#-*- coding: utf-8 -*-
import os
os.chdir('/home/ec2-user/utils/inven_data_real/')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import time
from datetime import datetime,timedelta

from pandas import read_csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

# model
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping 
from keras.layers import Dropout, Activation
from keras.models import Sequential
from keras.models import save_model, load_model

## Evaluate Model
from math import sqrt
from numpy import concatenate
from sklearn.metrics import mean_squared_error

# Time Measure
start_time = time.time()

# argument 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--index',dest='index',type=str,help="what is the parameters?")
args = parser.parse_args()
params = args.index
#program_id = params

import sys
sys.path.append('/home/centos/anaconda3/lib/python3.6/site-packages')
sys.path.append('/home/centos/anaconda3/lib/python3.6/site-packages/pandas')
sys.path.append('/home/centos/anaconda3/lib/python3.6/site-packages/numpy')


# function  
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data, index=dataset.index)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

program_id = params 

# Import Parameter
paramdata = pd.read_csv('DA_INVEN_PROG_TRG3.csv')
paramselect = paramdata[paramdata['PROGRAM_ID'] == params]

window_size = int(paramselect['WINDOW_SIZE']) 
model_neuron = int(paramselect['MODEL_NEURON']) 
model_ep = int(paramselect['MODEL_EP']) 
model_batchsize = int(paramselect['MODEL_BATCH'])  
model_dropout = float(paramselect['MODEL_DROPOUT'])
#model_dropout = 0.4
model_activation =paramselect['MODEL_ACT'] 
model_loss = paramselect['MODEL_LOSS'] 
model_optimizer = paramselect['MODEL_OPT'] 
model_ym = paramselect['YM'] 

i = window_size
j = 1488

# Start Time
start_time = time.time()

# Import Train data 
dataset = pd.read_csv(params+'.csv')
dataset['DATE'] = pd.to_datetime(dataset['DATE_TIME'],format="%Y-%m-%d %H:%M") 
dataset.sort_values(by=['DATE'], axis=0, ascending=True,inplace=True) 
dataset= dataset.set_index('DATE') 
dataset = dataset[['INVEN']] 
dataset.index.name = 'date' 
dataset_inven = dataset[['INVEN']] 

# ensure all data is float
values = dataset.values
values = values.astype('float32')

# normalize features
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# frame as supervised learning
raw_reframed = series_to_supervised(scaled, i+j, 1)

# WINDOW 
var_xx = raw_reframed.loc[:,:'var1(t-'+str(j)+')']

# Label
var_yy = raw_reframed['var1(t)']

# MERGE
raw_reframed = pd.concat([var_xx,var_yy], axis=1)
# print(raw_reframed.head())

reframed = raw_reframed
# print(reframed.head(5))

 

# DateTime
from datetime import datetime,timedelta 
current_time = datetime.utcnow() + timedelta(hours=9)
train_date_str = (current_time-timedelta(hours=2)).strftime("%Y%m%d%H")
#train_date_str = '2019052908'
s_traindate =  datetime.strptime(train_date_str, '%Y%m%d%H')

# string to datetime 
train = reframed[s_traindate:s_traindate].values              
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# load_model
#model = load_model
model = load_model('/home/ec2-user/utils/MODEL_REAL/'+program_id+'.h5')
model.fit(train_X, train_y, epochs=model_ep, batch_size=model_batchsize, verbose=2, shuffle=False) 
early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)
#history = model.fit(train_X, train_y, epochs=model_ep, batch_size=model_batchsize, verbose=2, shuffle=False)
model.save('/home/ec2-user/utils/MODEL_REAL/'+program_id+'.h5')
#model.save('/home/ec2-user/utils/model_hist/'+program_id+'_'+train_date_str+'.h5') 
print("Finished Online Training")

train_time = time.time()-start_time 

print("Start! Predict Inventory")
predict_date = (datetime.utcnow() + timedelta(hours=9)).strftime("%Y%m%d%H")
print("Making Testing Data")

#test data
from datetime import datetime,timedelta 
current_time = datetime.utcnow() + timedelta(hours=9)
test_date_str = (current_time).strftime("%Y%m%d")
test_date = test_date_str + '00'
s_test_date =  datetime.strptime(test_date, '%Y%m%d%H')

test_tdate_str = (current_time+timedelta(hours=1488)).strftime("%Y%m%d")
test_tdate = test_tdate_str + '23'
t_test_date = datetime.strptime(test_tdate,'%Y%m%d%H')

test = reframed.loc[s_test_date:t_test_date].values

# split into input and outputs
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

model = load_model('/home/ec2-user/utils/MODEL_REAL/'+program_id+'.h5')

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

dataset_actual = dataset.loc[s_test_date:t_test_date]["INVEN"]
Aarray_actual = np.array(dataset_actual)
Aarray_predicted = np.array(inv_yhat)

# calculate RMSE
rmse = sqrt(mean_squared_error(Aarray_actual, Aarray_predicted))
print('Test RMSE: %.3f' % rmse)
mape = mean_absolute_percentage_error(Aarray_actual,Aarray_predicted) 
print('Test MAPE: %.3f' % mape)
MEAN = np.mean(Aarray_predicted)
print('MEAN: %.3f' %MEAN)
STD = np.std(Aarray_predicted)
print('STD: %.3f' %STD)


result = np.stack((Aarray_actual,Aarray_predicted), axis=1)
result_pd = pd.DataFrame(result, index=dataset_actual.index, columns=['actual', 'predict'])
result_pd.reset_index(inplace=True)

result_pd['PROGRAM_ID'] = params
result_pd['YYYYMMDD'] = result_pd['date'].apply(lambda x: x.strftime("%Y%m%d"))
result_pd['HH'] = result_pd['date'].apply(lambda x: x.strftime("%H"))
result_pd['PRED_INVEN'] = result_pd['predict']
result_pd['PREDICT_DT'] = predict_date

result_pd_res = result_pd[['PROGRAM_ID', 'YYYYMMDD','HH','PRED_INVEN','PREDICT_DT']]

result_pd_res.to_csv('/home/ec2-user/utils/pred_result/'+program_id+'.csv', header=True)

print('Complete! Predict Invetory ')
print('time:',time.time()-start_time)

#print('Time to prediction of Inven .. :',time.time()-start_time)


#import pandas as pd 
#train_df = pd.DataFrame({
#    'programid':[program_id]
#    'test_sdate':[test_sdate],
#    'window_size':[window_size],
#    'train_time':[train_time]
#})
#train_df.to_csv('inven_train_S2_0521.csv', mode='a', header=False)
