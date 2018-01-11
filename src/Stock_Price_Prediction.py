############## Data Preprocessing ####################
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Google_Stock_Price_Train.csv')
dataset_train = dataset.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
dataset_train_scaled = sc.fit_transform(dataset_train)


# Create Data Structure with 60 timesteps and 1 output
# Will learn with 60 data and predict next date data
X_train = []
y_train = []

for i in range(60 , len(dataset)):
    X_train.append(dataset_train_scaled[i-60:i , 0])
    y_train.append(dataset_train_scaled[i])
    
# For X_train and y_train to be accepted by RNN , they should be numpy array
X_train , y_train = np.array(X_train) , np.array(y_train)


# Reshaping
# Add new predictors (dimension), which will help guess the prediction even more
X_train  =  np.reshape(X_train , (X_train.shape[0] , X_train.shape[1] , 1 ))



################## Building RNN ##################
################## LSTM ##########################
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Its a regressor as it is predicting continuos values
regressor = Sequential()


# Add LSTM layer and dropout LST
# First layer
# unit is 50 neuron per LSTM , return_sequences=True as it is not last LSTM , input_shape = last 2 dimesnion of training set
regressor.add(LSTM(units = 50 , return_sequences=True , input_shape=( X_train.shape[1] , 1)  ))
regressor.add(Dropout(rate=0.2))
# Second layer
regressor.add(LSTM(units = 50 , return_sequences=True ))
regressor.add(Dropout(rate=0.2))
# Third layer
regressor.add(LSTM(units = 50 , return_sequences=True ))
regressor.add(Dropout(rate=0.2))
# Last and fourth layer
regressor.add(LSTM(units = 1 , return_sequences=False ))
regressor.add(Dropout(rate=0.2))

# Output Layer
regressor.add(Dense(units=1))

# Compile
regressor.compile(optimizer='adam', loss='mean_squared_error')

# fitiing the training data 
regressor.fit(X_train , y_train , epochs= 100 , batch_size=32)

################## Building RNN ##################

################## Predict and Visualize next 60 days stock ##################

# Getting Real test values
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset.iloc[:, 1:2].values

# We will have to combine test and train dataset as 60th day will require from past data which is test data and not train data
dataset_total = pd.concat((dataset['Open'] , dataset_test['Open']) , axis=0)#axis = 0 vertical , 1 for horizontal

#Predict for Jan2017 , we need 60 days data prior to Jan
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60 : len(dataset_total)].values
# Converting to numpy array
inputs = inputs.reshape(-1,1)
# Features Scaling
inputs = sc.transform(inputs)

X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i , 0])
X_test = np.array(X_test)

#3D Structure
X_test  =  np.reshape(X_test , (X_test.shape[0] , X_test.shape[1] , 1 ))

predicted_stock_price = regressor.predict(X_test)

#inverse the scaling of prediction
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

################## Predict and Visualize ##################
plt.plot(real_stock_price, color='green' , label='Real Stock Price')
plt.plot(predicted_stock_price, color='red' , label='Predicted Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()