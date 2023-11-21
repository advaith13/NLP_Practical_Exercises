# EXERCISE 6

## OBJECTIVE
Load the Indian StockMarkets dataset from the dataset package and 
save the first 14000 observations from the "NIFTY" time series as working dataset
Dataset	 which contains 15 parameters such as date,Symbol,series,prev close,open,high,low etc..
Create two matrix containing 10 sequences of 14000 observations from the previous dataset. 
The first one must be made of the original observations and 
will be the input of our neural network. 
The second one will be the output and since we want to predict the value of the stock market at time t+1 based on the value at time t, 
this matrix will be the same as the first one were all the elements are shifted from one position. 
Make sure that each sequence are coded as a row of each matrix.
## RESOURCE/REQUIREMENTS
windows operating system ,python-editor/colab, python-interpreter
## PROGRAM LOGIC
1. Load stockmarket dataset
2. Process the data
	    3. Apply neural network techniques
	    4. Separating training and validation
	    5. Predicting stock market


## DESCRIPTION / PROCEDURE
```python
#Install Required Libraries
!pip install opendatasets 
!pip install pmdarima
# To Imporing dataset from kaggle
import opendatasets as od
# For data preprocessing
import pandas as pd
import numpy as np

# To build visualizations
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats
import pylab

# Stats model to perfrom statistical analysis
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# To build ML models
from fbprophet import Prophet
from matplotlib import pyplot as plt
import pandas.util.testing as tm
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb

# using pandas library to import the dataset
# Setting the data columns to index and for the convience to perfrom data analysis
od.download('https://www.kaggle.com/rohanrao/nifty50-stock-market-data?select=ASIANPAINT.csv')
df = pd.read_csv("/content/nifty50-stock-market-data/HDFCBANK.csv")
df.set_index("Date", drop=False, inplace=True)
df.head()
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
missing_table=missing_values_table(df)
missing_table
Output:
Your selected dataframe has 15 columns.
There are 3 columns that have missing values.
	Missing Values	% of Total Values
Trades	2850	53.7
Deliverable Volume	509	9.6
%Deliverble	509	9.6
df.drop(['Trades','Deliverable Volume','%Deliverble'],axis=1,inplace=True)
# Using the pandas library

plt.rcParams['figure.figsize'] = (20, 5)
plt.title('VWAP over time')
plt.ylabel('VWAP')
df.VWAP.plot() ;
# A pro technique would be to use plotly for interactive visual and time selectors

fig = px.line(df, x='Date', y='VWAP', title='Time Series with Selectors')

fig.update_xaxes(
    rangeslider_visible=False,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
fig.show()
# Splitting the date column to extract year, month, week and day
df.Date = pd.to_datetime(df.Date, format="%Y-%m-%d")
df["month"] = df.Date.dt.month
df["week"] = df.Date.dt.week
df["day"] = df.Date.dt.day
df["Year"]= df.Date.dt.year
df["day_of_week"] = df.Date.dt.dayofweek

#Below code block is used to plot the graph
fig = go.Figure([go.Scatter(x=df.loc[df["Year"] == 2020].Date,y=df.loc[df["Year"] == 2020].VWAP)])
fig.update_layout(
    autosize=False,
    width=1000,
    height=500,
    template='simple_white',
    title='HDFC Volume in 2020'
)
fig.update_xaxes(title="Date")
fig.update_yaxes(title="Volume")
fig.show()
df.reset_index(drop=True, inplace=True)

# Function to caclulate the moving averages based on number of days
def moving_average(DataFrame, window_size):
  numbers = DataFrame.High
  i=0
  moving_averages=[]
  while i < len(numbers) - window_size + 1:
      this_window = numbers[i : i + window_size]

      window_average = sum(this_window) / window_size
      moving_averages.append(window_average)
      i += 1
  return moving_averages

moving_averages_50 = moving_average(df, 50)
moving_averages_100 = moving_average(df, 100)
moving_averages_200 = moving_average(df, 200)

series1 = pd.Series(moving_averages_50, name="50daysMA")
series2 = pd.Series(moving_averages_100, name="100daysMA")
series3 = pd.Series(moving_averages_200, name="200daysMA")
df = pd.concat([df, series1, series2, series3], axis=1)
df['50daysMA'] = df['50daysMA'].shift(50)
df['100daysMA'] = df['50daysMA'].shift(100)
df['200daysMA'] = df['50daysMA'].shift(200)

df.set_index("Date", drop=False, inplace=True)
df.loc[df["Year"] == 2020][['VWAP', '50daysMA','100daysMA','200daysMA']].plot();
fig = go.Figure()
fig.add_trace(go.Scatter(
         x=df.loc[df["Year"] == 2020].Date,
         y=df.loc[df["Year"] == 2020].VWAP,
         name='Open',
    line=dict(color='blue'),
    opacity=0.8))

fig.add_trace(go.Scatter(
         x=df.loc[df["Year"] == 2020].Date,
         y=df.loc[df["Year"] == 2020]['50daysMA'],
         name='Close',
    line=dict(color='red'),
    opacity=0.8))

fig.add_trace(go.Scatter(
         x=df.loc[df["Year"] == 2020].Date,
         y=df.loc[df["Year"] == 2020]['100daysMA'],
         name='Close',
    line=dict(color='green'),
    opacity=0.8))
fig.add_trace(go.Scatter(
         x=df.loc[df["Year"] == 2020].Date,
         y=df.loc[df["Year"] == 2020]['200daysMA'],
         name='Close',
    line=dict(color='black'),
    opacity=0.8))

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
scipy.stats.probplot(df.VWAP,plot=pylab)
pylab.show()
cols_plot = ['Open', 'Close', 'High','Low']
axes = df[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Daily trade')
ax=df[['Volume']].plot(stacked=True)
ax.set_title('Volume over years',fontsize= 30)
ax.set_xlabel('Year',fontsize = 20)
ax.set_ylabel('Volume',fontsize = 20)
figure(figsize=(8, 6), dpi=80)
plt.show();
df.reset_index(drop=True, inplace=True)
lag_features = ["High", "Low", "Volume", "Turnover"]
window1 = 3
window2 = 7
window3 = 30

df_rolled_3d = df[lag_features].rolling(window=window1, min_periods=0)
df_rolled_7d = df[lag_features].rolling(window=window2, min_periods=0)
df_rolled_30d = df[lag_features].rolling(window=window3, min_periods=0)

df_mean_3d = df_rolled_3d.mean().shift(1).reset_index().astype(np.float32)
df_mean_7d = df_rolled_7d.mean().shift(1).reset_index().astype(np.float32)
df_mean_30d = df_rolled_30d.mean().shift(1).reset_index().astype(np.float32)

df_std_3d = df_rolled_3d.std().shift(1).reset_index().astype(np.float32)
df_std_7d = df_rolled_7d.std().shift(1).reset_index().astype(np.float32)
df_std_30d = df_rolled_30d.std().shift(1).reset_index().astype(np.float32)

for feature in lag_features:
    df[f"{feature}_mean_lag{window1}"] = df_mean_3d[feature]
    df[f"{feature}_mean_lag{window2}"] = df_mean_7d[feature]
    df[f"{feature}_mean_lag{window3}"] = df_mean_30d[feature]
    
    df[f"{feature}_std_lag{window1}"] = df_std_3d[feature]
    df[f"{feature}_std_lag{window2}"] = df_std_7d[feature]
    df[f"{feature}_std_lag{window3}"] = df_std_30d[feature]

df.fillna(df.mean(), inplace=True)

df.set_index("Date", drop=False, inplace=True)
df.reset_index(drop=True, inplace=True)
lag_features = ["High", "Low", "Volume", "Turnover"]
window1 = 3
window2 = 7
window3 = 30

df_rolled_3d = df[lag_features].rolling(window=window1, min_periods=0)
df_rolled_7d = df[lag_features].rolling(window=window2, min_periods=0)
df_rolled_30d = df[lag_features].rolling(window=window3, min_periods=0)

df_mean_3d = df_rolled_3d.mean().shift(1).reset_index().astype(np.float32)
df_mean_7d = df_rolled_7d.mean().shift(1).reset_index().astype(np.float32)
df_mean_30d = df_rolled_30d.mean().shift(1).reset_index().astype(np.float32)

df_std_3d = df_rolled_3d.std().shift(1).reset_index().astype(np.float32)
df_std_7d = df_rolled_7d.std().shift(1).reset_index().astype(np.float32)
df_std_30d = df_rolled_30d.std().shift(1).reset_index().astype(np.float32)

for feature in lag_features:
    df[f"{feature}_mean_lag{window1}"] = df_mean_3d[feature]
    df[f"{feature}_mean_lag{window2}"] = df_mean_7d[feature]
    df[f"{feature}_mean_lag{window3}"] = df_mean_30d[feature]
    
    df[f"{feature}_std_lag{window1}"] = df_std_3d[feature]
    df[f"{feature}_std_lag{window2}"] = df_std_7d[feature]
    df[f"{feature}_std_lag{window3}"] = df_std_30d[feature]

df.fillna(df.mean(), inplace=True)

df.set_index("Date", drop=False, inplace=True)
plt.rcParams.update({'figure.figsize': (10,10)})
y = df['VWAP'].to_frame()

# Multiplicative Decomposition 
result_mul = seasonal_decompose(y, model='multiplicative',period = 52)

# Additive Decomposition
result_add = seasonal_decompose(y, model='additive',period = 52)

# Plot
plt.rcParams.update({'figure.figsize': (10,10)})
result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
result_add.plot().suptitle('Additive Decompose', fontsize=22)
plt.show()
df['vwap_diff']=df['VWAP']-df['VWAP'].shift(1)
fig = go.Figure([go.Scatter(x=df.index,y=df.VWAP)])
fig.update_layout(
    autosize=False,
    width=1000,
    height=500,
    template='simple_white',
    title='VWAP over time ')
fig.show()
fig = go.Figure([go.Scatter(x=df.index,y=df.vwap_diff)])
fig.update_layout(
    autosize=False,
    width=1000,
    height=500,
    template='simple_white',
    title='difference VWAP over time ')
fig.show()
#Splitting training and valid
# Splitting training and testing set
df_train = df[df.Date < "2019"]
df_valid = df[df.Date >= "2019"]
# We put together all the features we caculated
exogenous_features = ["High_mean_lag3", "High_std_lag3", "Low_mean_lag3", "Low_std_lag3",
                      "Volume_mean_lag3", "Volume_std_lag3", "Turnover_mean_lag3",
                      "Turnover_std_lag3", 
                      "High_mean_lag7", "High_std_lag7", "Low_mean_lag7", "Low_std_lag7",
                      "Volume_mean_lag7", "Volume_std_lag7", "Turnover_mean_lag7",
                      "Turnover_std_lag7", 
                      "High_mean_lag30", "High_std_lag30", "Low_mean_lag30", "Low_std_lag30",
                      "Volume_mean_lag30", "Volume_std_lag30", "Turnover_mean_lag30",
                      "Turnover_std_lag30", 
                      "month", "week", "day", "day_of_week"]


# Standard steps for passing the data to auto_arima
model = auto_arima(df_train.VWAP, exogenous=df_train[exogenous_features], trace=True, error_action="ignore", suppress_warnings=True)
model.fit(df_train.VWAP, exogenous=df_train[exogenous_features])

forecast = model.predict(n_periods=len(df_valid), exogenous=df_valid[exogenous_features])
df_valid["Forecast_ARIMAX"] = forecast

model.summary()
```
Output:
```
Sample:	0	HQIC 	47941.542
	- 4729		
Covariance Type:	Opg		
	coef	std err	z	P>|z|	[0.025	0.975]
High_mean_lag3	1.1573	5.2e-25	2.22e+24	0.000	1.157	1.157
High_std_lag3	-0.0997	5.51e-27	-1.81e+25	0.000	-0.100	-0.100
Low_mean_lag3	0.0977	5.08e-25	1.92e+23	0.000	0.098	0.098
Low_std_lag3	-0.7024	3.32e-27	-2.12e+26	0.000	-0.702	-0.702
Volume_mean_lag3	-5.438e-07	1.46e-22	-3.72e+15	0.000	-5.44e-07	-5.44e-07
Volume_std_lag3	2.199e-06	1.02e-22	2.15e+16	0.000	2.2e-06	2.2e-06
Turnover_mean_lag3	2.811e-14	1.51e-14	1.859	0.063	-1.52e-15	5.77e-14
Turnover_std_lag3	-2.707e-14	1.23e-14	-2.206	0.027	-5.11e-14	-3.02e-15
High_mean_lag7	-0.3722	5.12e-25	-7.27e+23	0.000	-0.372	-0.372
High_std_lag7	0.1481	1.91e-26	7.74e+24	0.000	0.148	0.148
Low_mean_lag7	0.0888	5.03e-25	1.77e+23	0.000	0.089	0.089
Low_std_lag7	0.0384	1.21e-26	3.17e+24	0.000	0.038	0.038
Volume_mean_lag7	4.38e-06	1.39e-22	3.15e+16	0.000	4.38e-06	4.38e-06
Volume_std_lag7	-3.229e-06	1.62e-22	-1.99e+16	0.000	-3.23e-06	-3.23e-06
Turnover_mean_lag7	-9.32e-14	4.09e-14	-2.276	0.023	-1.73e-13	-1.29e-14
Turnover_std_lag7	4.446e-14	2.02e-14	2.199	0.028	4.84e-15	8.41e-14
High_mean_lag30	-0.0042	4.4e-25	-9.64e+21	0.000	-0.004	-0.004
High_std_lag30	0.2002	1.23e-26	1.62e+25	0.000	0.200	0.200
Low_mean_lag30	0.0301	4.41e-25	6.83e+22	0.000	0.030	0.030
Low_std_lag30	-0.2418	1.25e-26	-1.93e+25	0.000	-0.242	-0.242
Volume_mean_lag30	-2.252e-06	3.54e-22	-6.37e+15	0.000	-2.25e-06	-2.25e-06
Volume_std_lag30	-2.07e-06	7.63e-23	-2.72e+16	0.000	-2.07e-06	-2.07e-06
Turnover_mean_lag30	6.098e-14	3.2e-14	1.904	0.057	-1.79e-15	1.24e-13
Turnover_std_lag30	7.918e-15	1.33e-14	0.594	0.552	-1.82e-14	3.4e-14
Month	0.3117	8.6e-28	3.63e+26	0.000	0.312	0.312
Week	-0.0208	3.49e-27	-5.97e+24	0.000	-0.021	-0.021
Day	0.0698	1.56e-27	4.48e+25	0.000	0.070	0.070
day_of_week	-0.2180	1.09e-27	-2e+26	0.000	-0.218	-0.218
ar.L1	1.0792	6e-26	1.8e+25	0.000	1.079	1.079
ar.L2	-0.5326	1.83e-26	-2.91e+25	0.000	-0.533	-0.533
ma.L1	-0.6761	6.19e-26	-1.09e+25	0.000	-0.676	-0.676
ma.L2	0.1182	1.64e-26	7.19e+24	0.000	0.118	0.118
ma.L3	0.1432	1.3e-26	1.1e+25	0.000	0.143	0.143
sigma2	1279.8905	4.3e-28	2.98e+30	0.000	1279.890	1279.890
Ljung-Box (L1) (Q):	7.93	Jarque-Bera (JB): 	566322666.61
Prob(Q):	0.00	Prob(JB): 	0.00
Heteroskedasticity (H):	0.36	Skew: 	-32.75
Prob(H) (two-sided):	0.00	Kurtosis: 	1697.06
```
Executed Code:
https://colab.research.google.com/drive/1X3rPLHwhQBlzV8CqPxFShPu3-9OWoc83#scrollTo=WNslBGk2W56o



