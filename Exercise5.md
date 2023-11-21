# EXERCISE 5

Load the data set “Household Electric Power Consumption”. It provides measurements of electric power consumption in one household with a one-minute sampling rate.
There are 2,075,259 measurements gathered within 4 years. Different electrical quantities and some sub-metering values are available. But we’ll only focus on three features:
-	Date: date in format dd/mm/yyyy
-	Time: time in format hh:mm:ss
-	Global_active_power: household global minute-averaged active power (in kilowatt)
predict the amount of Global_active_power 10 minutes ahead.Using  LSTM with TensorFlow Keras
## RESOURCE/REQUIREMENTS
windows operating system ,python-editor/colab, python-interpreter
## PROGRAM LOGIC
1. Load  “Household Electric Power Consumption” dataset
2. Apply LSTM Techniques
	    3. Build the model
	    4. Compile the model
## DESCRIPTION / PROCEDURE

Load the dataset using following link
https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set?resource=download
```python
# import packages
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import time
import os
from google.colab import drive
drive.mount("/content/drive")
#path = 'drive/My Drive/Colab Notebooks/'
# read the dataset into python
df = pd.read_csv('drive/My Drive/Colab Notebooks/household_power_consumption.txt', delimiter=';')
df.head()

Output:
Date	Time	Global_active_power	Global_reactive_power	Voltage	Global_intensity	Sub_metering_1	Sub_metering_2	Sub_metering_3
0	16/12/2006	17:24:00	4.216	0.418	234.840	18.400	0.000	1.000	17.0
1	16/12/2006	17:25:00	5.360	0.436	233.630	23.000	0.000	1.000	16.0
2	16/12/2006	17:26:00	5.374	0.498	233.290	23.000	0.000	2.000	17.0
3	16/12/2006	17:27:00	5.388	0.502	233.740	23.000	0.000	1.000	17.0
4	16/12/2006	17:28:00	3.666	0.528	235.680	15.800	0.000	1.000	17.0




#Preprocessing the Dataset for Time Series Analysis
%%time

df['date_time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
df = df.dropna(subset=['Global_active_power'])

df['date_time'] = pd.to_datetime(df['date_time'])

df = df.loc[:, ['date_time', 'Global_active_power']]
df.sort_values('date_time', inplace=True, ascending=True)
df = df.reset_index(drop=True)

print('Number of rows and columns after removing missing values:', df.shape)
print('The time series starts from: ', df['date_time'].min())
print('The time series ends on: ', df['date_time'].max())
df.info()
df.head(10)

Output:
date_time	Global_active_power
0	2006-12-16 17:24:00	4.216
1	2006-12-16 17:25:00	5.360
2	2006-12-16 17:26:00	5.374
3	2006-12-16 17:27:00	5.388
4	2006-12-16 17:28:00	3.666
5	2006-12-16 17:29:00	3.520
6	2006-12-16 17:30:00	3.702
7	2006-12-16 17:31:00	3.700
8	2006-12-16 17:32:00	3.668
9	2006-12-16 17:33:00	3.662

# Split into training, validation and test datasets.
# Since it's timeseries we should do it by date.
test_cutoff_date = df['date_time'].max() - timedelta(days=7)
val_cutoff_date = test_cutoff_date - timedelta(days=14)

df_test = df[df['date_time'] > test_cutoff_date]
df_val = df[(df['date_time'] > val_cutoff_date) & (df['date_time'] <= test_cutoff_date)]
df_train = df[df['date_time'] <= val_cutoff_date]

#check out the datasets
print('Test dates: {} to {}'.format(df_test['date_time'].min(), df_test['date_time'].max()))
print('Validation dates: {} to {}'.format(df_val['date_time'].min(), df_val['date_time'].max()))
print('Train dates: {} to {}'.format(df_train['date_time'].min(), df_train['date_time'].max()))

#Transforming the Dataset for TensorFlow Keras
# Goal of the model:
#  Predict Global_active_power at a specified time in the future.
#   Eg. We want to predict how much Global_active_power will be ten minutes from now.
#       We can use all the values from t-1, t-2, t-3, .... t-history_length to predict t+10

def create_ts_files(dataset, 
                    start_index, 
                    end_index, 
                    history_length, 
                    step_size, 
                    target_step, 
                    num_rows_per_file, 
                    data_folder):
    assert step_size > 0
    assert start_index >= 0
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    time_lags = sorted(range(target_step+1, target_step+history_length+1, step_size), reverse=True)
    col_names = [f'x_lag{i}' for i in time_lags] + ['y']
    start_index = start_index + history_length
    if end_index is None:
        end_index = len(dataset) - target_step
    
    rng = range(start_index, end_index)
    num_rows = len(rng)
    num_files = math.ceil(num_rows/num_rows_per_file)
    
    # for each file.
    print(f'Creating {num_files} files.')
    for i in range(num_files):
        filename = f'{data_folder}/ts_file{i}.pkl'
        if i % 10 == 0:
            print(f'{filename}')
            
        # get the start and end indices.
        ind0 = i*num_rows_per_file
        ind1 = min(ind0 + num_rows_per_file, end_index)
        data_list = []
        
        # j in the current timestep. Will need j-n to j-1 for the history. And j + target_step for the target.
        for j in range(ind0, ind1):
            indices = range(j-1, j-history_length-1, -step_size)
            data = dataset[sorted(indices) + [j+target_step]]
            
            # append data to the list.
            data_list.append(data)

        df_ts = pd.DataFrame(data=data_list, columns=col_names)
        df_ts.to_pickle(filename)
            
    return len(col_names)-1
#Transforming the Dataset for TensorFlow Keras
# Goal of the model:
#  Predict Global_active_power at a specified time in the future.
#   Eg. We want to predict how much Global_active_power will be ten minutes from now.
#       We can use all the values from t-1, t-2, t-3, .... t-history_length to predict t+10

def create_ts_files(dataset, 
                    start_index, 
                    end_index, 
                    history_length, 
                    step_size, 
                    target_step, 
                    num_rows_per_file, 
                    data_folder):
    assert step_size > 0
    assert start_index >= 0
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    time_lags = sorted(range(target_step+1, target_step+history_length+1, step_size), reverse=True)
    col_names = [f'x_lag{i}' for i in time_lags] + ['y']
    start_index = start_index + history_length
    if end_index is None:
        end_index = len(dataset) - target_step
    
    rng = range(start_index, end_index)
    num_rows = len(rng)
    num_files = math.ceil(num_rows/num_rows_per_file)
    
    # for each file.
    print(f'Creating {num_files} files.')
    for i in range(num_files):
        filename = f'{data_folder}/ts_file{i}.pkl'
        if i % 10 == 0:
            print(f'{filename}')
            
        # get the start and end indices.
        ind0 = i*num_rows_per_file
        ind1 = min(ind0 + num_rows_per_file, end_index)
        data_list = []
        
        # j in the current timestep. Will need j-n to j-1 for the history. And j + target_step for the target.
        for j in range(ind0, ind1):
            indices = range(j-1, j-history_length-1, -step_size)
            data = dataset[sorted(indices) + [j+target_step]]
            
            # append data to the list.
            data_list.append(data)

        df_ts = pd.DataFrame(data=data_list, columns=col_names)
        df_ts.to_pickle(filename)
            
    return len(col_names)-1
# So we can handle loading the data in chunks from the hard drive instead of having to load everything into memory.
# 
# The reason we want to do this is so we can do custom processing on the data that we are feeding into the LSTM.
# LSTM requires a certain shape and it is tricky to get it right.
#
class TimeSeriesLoader:
    def __init__(self, ts_folder, filename_format):
        self.ts_folder = ts_folder
        
        # find the number of files.
        i = 0
        file_found = True
        while file_found:
            filename = self.ts_folder + '/' + filename_format.format(i)
            file_found = os.path.exists(filename)
            if file_found:
                i += 1
                
        self.num_files = i
        self.files_indices = np.arange(self.num_files)
        self.shuffle_chunks()
        
    def num_chunks(self):
        return self.num_files
    
    def get_chunk(self, idx):
        assert (idx >= 0) and (idx < self.num_files)
        
        ind = self.files_indices[idx]
        filename = self.ts_folder + '/' + filename_format.format(ind)
        df_ts = pd.read_pickle(filename)
        num_records = len(df_ts.index)
        
        features = df_ts.drop('y', axis=1).values
        target = df_ts['y'].values
        
        # reshape for input into LSTM. Batch major format.
        features_batchmajor = np.array(features).reshape(num_records, -1, 1)
        return features_batchmajor, target
    
    # this shuffles the order the chunks will be outputted from get_chunk.
    def shuffle_chunks(self):
        np.random.shuffle(self.files_indices)
ts_folder = 'ts_data'
filename_format = 'ts_file{}.pkl'
tss = TimeSeriesLoader(ts_folder, filename_format)

# Create the Keras model.
# Use hyperparameter optimization if you have the time.

ts_inputs = tf.keras.Input(shape=(num_timesteps, 1))

# units=10 -> The cell and hidden states will be of dimension 10.
#             The number of parameters that need to be trained = 4*units*(units+2)
x = layers.LSTM(units=10)(ts_inputs)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation='linear')(x)
model = tf.keras.Model(inputs=ts_inputs, outputs=outputs)
# Specify the training configuration.
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mse'])
model.summary()
```
 Output:
```
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 1008, 1)]         0         
                                                                 
 lstm (LSTM)                 (None, 10)                480       
                                                                 
 dropout (Dropout)           (None, 10)                0         
                                                                 
 dense (Dense)               (None, 1)                 11        
                                                                 
=================================================================
Total params: 491
Trainable params: 491
Non-trainable params: 0
```
 

