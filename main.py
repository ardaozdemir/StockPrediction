import pandas as pd
from fbprophet import Prophet
import datetime as dt
import pandas_datareader as web
import numpy as np
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from matplotlib import pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'

#Getting the stock name
company = "TSLA"
predict_day = 30

#Initializing dates
start = dt.datetime(2016, 1, 1)
end = dt.datetime.now()

#Getting stock prices from yahoo
data = web.DataReader(company, 'yahoo', start, end)

#FB prophet Prediction
data.to_csv('test.csv')

df = pd.read_csv("test.csv")

columns = ["Date", "Close"]
ndf = pd.DataFrame(df, columns=columns)

#Renaming dataframe for FBProphet
prophet_df = ndf.rename(columns={"Date": "ds", "Close":"y"})

#FBProphet Predictions
m = Prophet()
m.fit(prophet_df)

prediction_days = 30   #Predict n days into the future

future = m.make_future_dataframe(periods=prediction_days)
forecast = m.predict(future)

#Saving results
forecast.to_csv('forecast.csv')

#Getting real and predicted prices
val = forecast['yhat'].values
today = data['Close'].values
prophet_prediction = val[-prediction_days:]

#Getting today's price
today_price = today[len(today)-1]

##Linear regression calculations

df_new = data[["Close"]]

df_new["Prediction"] = df_new[["Close"]].shift(-prediction_days)

#Creating independent set

x = np.array(df_new.drop(["Prediction"], 1))

#Remove last n rows from x
x = x[:-prediction_days]

#Creating dependent data set
y = np.array(df_new["Prediction"])

#Remove last n rows from y
y = y[:-prediction_days]

#Split %75 training %25 testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

#Creating and training the linear regression model
lr = LinearRegression()
lr.fit(x_train, y_train)

#Creating and training the support vector regression model
svr = SVR(kernel="rbf", C=1e3, gamma=0.1)
svr.fit(x_train, y_train)

#Creating and training random forest regressor model
rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)

#Creating and training gradient boosting regressor model
gbr = GradientBoostingRegressor()
gbr.fit(x_train, y_train)

#Creating and training kernel ridge regressor model
kr = KernelRidge()
kr.fit(x_train, y_train)

#Set x_forecast equal to last 30 rows of the "Close" prices
x_forecast = np.array(df_new.drop(["Prediction"], 1))[-prediction_days:]

#Printing linear regression model results for n days
lr_prediction = lr.predict(x_forecast)
print(lr_prediction)

#Printing support vector regression model results for n days
svr_prediction = svr.predict(x_forecast)
print(svr_prediction)

#Printing random forest regression results for n days
rfr_prediction = rfr.predict(x_forecast)
print(rfr_prediction)

#Printing gradient boosting regression results for n days
gbr_prediction = gbr.predict(x_forecast)
print(gbr_prediction)

#Printing kernel ridge regression results for n days
kr_prediction = kr.predict(x_forecast)
print(kr_prediction)

#Converting array to the numpy array to get the average
prophet_prediction = np.array(prophet_prediction)
lr_prediction = np.array(lr_prediction)
svr_prediction = np.array(svr_prediction)
rfr_prediction = np.array(rfr_prediction)
gbr_prediction = np.array(gbr_prediction)
kr_prediction = np.array(kr_prediction)

#Calculating the avarage for last 30 days
average = (prophet_prediction + lr_prediction + svr_prediction + rfr_prediction + gbr_prediction + kr_prediction)/6.0

#Converting it back to normal array
np.asarray(average)
print(average)

#Getting predicted price from different regression models
average_prediction = average[len(average)-1-prediction_days+predict_day]

print("___________________________________________________")

#Checking for buying or selling check
if(today_price < average_prediction):
    print("You should buy or keep your " + company + " share ")

else:
    print("You should sell your " + company + " share ")


#Showing todays and future price
print("Current price: " + str(today_price))
print("Price prediction after " + str(predict_day) + " days: " + str(average_prediction))

#Getting dates today to 30 days later
date_list = pd.date_range(end=dt.datetime.today() + dt.timedelta(days=30), periods=30).to_pydatetime().tolist()

#Plotting the graph using MatPlotLib
plt.plot_date(date_list,average, linestyle="solid")
plt.gcf().autofmt_xdate()
plt.xlabel("Date")
plt.ylabel("Stock price")
plt.show()