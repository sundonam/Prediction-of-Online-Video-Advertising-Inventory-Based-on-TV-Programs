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

# 시간측정
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

# 파라미터 정보 가져오기
paramdata = pd.read_csv('DA_INVEN_PROG_TRG3.csv')
paramselect = paramdata[paramdata['PROGRAM_ID'] == params]

window_size = int(paramselect['WINDOW_SIZE']) # WINDOW_SIZE - WINDOW_SIZE 
model_neuron = int(paramselect['MODEL_NEURON']) # MODE_NEURON 
model_ep = int(paramselect['MODEL_EP']) # 반복횟수
model_batchsize = int(paramselect['MODEL_BATCH']) # 배치횟수 
model_dropout = float(paramselect['MODEL_DROPOUT']) # 드롭아웃
#model_dropout = 0.4
model_activation =paramselect['MODEL_ACT'] # 활성화함수
model_loss = paramselect['MODEL_LOSS'] # 손실함수 
model_optimizer = paramselect['MODEL_OPT'] # 최적화함수 
model_ym = paramselect['YM'] # 예측연월

i = window_size
j = 1488

# 시작 시간
start_time = time.time()

# 학습 데이터 가져오기 
dataset = pd.read_csv(params+'.csv')
dataset['DATE'] = pd.to_datetime(dataset['DATE_TIME'],format="%Y-%m-%d %H:%M") # DATE 컬럼 추가 
dataset.sort_values(by=['DATE'], axis=0, ascending=True,inplace=True) # DATE컬럼을 기준으로 정렬
dataset= dataset.set_index('DATE') # DATE 컬럼을 인덱스로 지정 
dataset = dataset[['INVEN']] # INVEN 컬럼만 dataset에 할당 
dataset.index.name = 'date' # 인덱스 이름을 'date' 로 변경
dataset_inven = dataset[['INVEN']] 

# ensure all data is float
values = dataset.values
values = values.astype('float32')

# normalize features
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# frame as supervised learning
raw_reframed = series_to_supervised(scaled, i+j, 1)

# shitf 하는 데이터의 첫 row만 가져오기(TETS셋 데이터 포함안되게 하기위해)
# WINDOW 컬럼
var_xx = raw_reframed.loc[:,:'var1(t-'+str(j)+')']

# 라벨
var_yy = raw_reframed['var1(t)']

# MERGE
raw_reframed = pd.concat([var_xx,var_yy], axis=1)
# print(raw_reframed.head())

reframed = raw_reframed
# print(reframed.head(5))

# train기간 설정: 실시간은 2시간 전 데이터만 학습 
# 한국시간으로  

# 현재시간 구하기 (현재시간 기준 1시간 전 데이터만 학습)
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
#model = load_model('E:/00.WISE/SMR인벤예측_201905/model/J01_PR10010797.h5')
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

# 예측 기간은 2달 후 예측 (2019053000,2019073023)

#test데이터의 시작 일자 (현재일자) 
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
print('평균: %.3f' %MEAN)
STD = np.std(Aarray_predicted)
print('표준편차: %.3f' %STD)


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

###분석 소요시간 체크

#print('Time to prediction of Inven .. :',time.time()-start_time)

## 학습 및 예측 시간 저장하기 
#import pandas as pd 
#train_df = pd.DataFrame({
#    'programid':[program_id]
#    'test_sdate':[test_sdate],
#    'window_size':[window_size],
#    'train_time':[train_time]
#})
#train_df.to_csv('inven_train_S2_0521.csv', mode='a', header=False)









 



