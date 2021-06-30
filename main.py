import pandas as pd
import plotly.express as px
from fbprophet import Prophet
import datetime as dt
import pandas_datareader as web
import plotly.io as pio
import numpy as np
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


pd.options.mode.chained_assignment = None  # default='warn'

#Getting the stock name
company = "TRY=X"
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

#Visualizing data
#px.line(forecast, x='ds', y='yhat')

#Saving results
forecast.to_csv('forecast.csv')

#Getting real and predicted prices
val = forecast['yhat'].values
today = data['Close'].values

#Getting today's price and predicted price
today_price = today[len(today)-1]
predicted_price_prophet = val[len(val)-1-prediction_days+predict_day]

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
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25)

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

#Getting predicted price from different regression models
predicted_price_linear_regression = lr_prediction[len(lr_prediction)-1-prediction_days+predict_day]
predicted_price_sv_regression = svr_prediction[len(svr_prediction)-1-prediction_days+predict_day]
predicted_price_rfr_regression = rfr_prediction[len(rfr_prediction)-1-prediction_days+predict_day]
predicted_price_gbr_regression = gbr_prediction[len(gbr_prediction)-1-prediction_days+predict_day]

print("___________________________________________________")

#Checking for buying or selling FBProphet
if(today_price < predicted_price_prophet):
    print("You should buy or keep your " + company + " share (FB Prophet Result)")

else:
    print("You should sell your " + company + " share (FB Prophet Result)")

#Checking for buying or selling Linear Regression
if(today_price < predicted_price_linear_regression):
    print("You should buy or keep your " + company + " share (Linear Regression Result)")

else:
    print("You should sell your " + company + " share (Linear Regression Result)")

#Checking for buying or selling Support Vector Regression
if(today_price < predicted_price_sv_regression):
    print("You should buy or keep your " + company + " share (Support Vector Regression Result)")

else:
    print("You should sell your " + company + " share (Support Vector Regression Result)")

#Checking for buying or selling Random Forest Regression
if(today_price < predicted_price_rfr_regression):
    print("You should buy or keep your " + company + " share (Random Forest Regression Result)")

else:
    print("You should sell your " + company + " share (Random Forest Regression Result)")

#Checking for buying or selling Gradient Boosting Regression
if(today_price < predicted_price_gbr_regression):
    print("You should buy or keep your " + company + " share (Gradient Boosting Regression Result)")

else:
    print("You should sell your " + company + " share (Gradient Boosting Regression Result)")


print("___________________________________________________")

print("Current price: " + str(today_price))
print("Price prediction after " + str(predict_day) + " days (FBProphet Result): " + str(predicted_price_prophet))
print("Price prediction after " + str(predict_day) + " days (Linear Regression Result): " + str(predicted_price_linear_regression))
print("Price prediction after " + str(predict_day) + " days (Support Vector Regression Result): " + str(predicted_price_sv_regression))
print("Price prediction after " + str(predict_day) + " days (Random Forest Regression Result): " + str(predicted_price_rfr_regression))
print("Price prediction after " + str(predict_day) + " days (Gradient Boosting Regression Result): " + str(predicted_price_gbr_regression))
