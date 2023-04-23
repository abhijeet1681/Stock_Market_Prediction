import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yfin
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import streamlit as st
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras import models



yfin.pdr_override()
start = '2010-01-01'
end = '2023-02-06'

st.title('Stock Trend Prediction')

input = st.text_input('Enter Stock Ticker', 'AAPL')
df = pdr.get_data_yahoo(input,start,end)
df.tail()

#Describing Data
st.subheader('Data from 2010 - '+end)
st.write(df.describe())

st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close, 'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
MA100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(MA100 , 'r')
plt.plot(df.Close ,'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
MA100 = df.Close.rolling(100).mean()
MA200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(MA100 ,'r')
plt.plot(MA200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

#Spliting The data into training and testing
train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
test = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

print(train.shape)
print(test.shape)

scaler = MinMaxScaler(feature_range=(0,1))
train_array = scaler.fit_transform(train)


model = models.load_model('keras_model.h5')

past_100_days = train.tail(100)
final_df = past_100_days.append(test, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_pred = model.predict(x_test)

scaler_val = scaler.scale_
scale_factor = 1/scaler_val[0]
y_pred = y_pred*scale_factor
y_test = y_test*scale_factor

st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_pred, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
