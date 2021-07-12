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
import yfinance as yf
import pandas_datareader.data as pdr
import matplotlib.ticker as ticker

pd.options.mode.chained_assignment = None  # default='warn'

yf.pdr_override()

#Getting the stock name
company = "AAPL"
predict_day = 30

#Initializing dates
start = dt.datetime(2016, 1, 1)
end = dt.datetime.now()

#Getting stock prices from yahoo
#data = web.DataReader(company, 'yahoo', start, end)
data = pdr.get_data_yahoo(company, data_source='yahoo', start=start, end=end)

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
svr = SVR(kernel="rbf", C=1e3, gamma=0.00001)
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
print("Linear regression:")
print(lr_prediction)

#Printing support vector regression model results for n days
svr_prediction = svr.predict(x_forecast)
#print("Support vector regression:")
#print(svr_prediction)

#Printing random forest regression results for n days
rfr_prediction = rfr.predict(x_forecast)
print("Random forest regression:")
print(rfr_prediction)

#Printing gradient boosting regression results for n days
gbr_prediction = gbr.predict(x_forecast)
print("Gradient boosting regression:")
print(gbr_prediction)

#Printing kernel ridge regression results for n days
kr_prediction = kr.predict(x_forecast)
print("Kernel ridge regression:")
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
#print(average)

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

for i in range(len(date_list)):
    date_list[i] = date_list[i].strftime('%d-%m')

#This function creates the new array with intersections
def new_prediction(lr_prediction, rfr_prediction, gbr_prediction, kr_prediction):
    new_predicts = []
    length = len(lr_prediction)
    for i in range(length):
        compare = []
        compare.append(lr_prediction[i])
        compare.append(rfr_prediction[i])
        compare.append(gbr_prediction[i])
        compare.append(kr_prediction[i])

        new_predicts.append(minDiffPairs(compare, 4))

    return new_predicts

#This function finds the closest elements in array
def minDiffPairs(arr, n):
    if n <= 1: return

    # Sorting array
    arr.sort()

    #Comparing the differences
    minDiff = arr[1] - arr[0]
    for i in range(2, n):
        minDiff = min(minDiff, arr[i] - arr[i - 1])

    #Finding the numbers with the less difference
    for i in range(1, n):
        if (arr[i] - arr[i - 1]) == minDiff:
            return arr[i - 1]

#Getting new data with intersections
new_predicts = new_prediction(lr_prediction, rfr_prediction, gbr_prediction, kr_prediction)

#Plotting the graph using MatPlotLib
#plt.plot_date(date_list,average, linestyle="solid")
#plt.gcf().autofmt_xdate()
#plt.xlabel("Date")
#plt.ylabel("Stock price")
#plt.show()

#Drawing multiple graphs in one figure
#figure, axis = plt.subplots(1, 2)

fig, ax = plt.subplots()

ax.plot_date(date_list, new_predicts, linestyle="solid", color='r')
ax.plot_date(date_list, lr_prediction, linestyle="solid", color='g')
#plt.plot_date(date_list, svr_prediction, linestyle="solid", color='b')
ax.plot_date(date_list, rfr_prediction, linestyle="solid", color='c')
ax.plot_date(date_list, gbr_prediction, linestyle="solid", color='y')
ax.plot_date(date_list, kr_prediction, linestyle="solid", color='k')
#plt.plot_date(date_list, average, linestyle="solid", color='m')
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.title("Predictions")
ax.xaxis.set_major_locator(ticker.MultipleLocator(3))

# To load the display window
plt.show()
#FB Prophet Graph
#axis[0, 0].plot(date_list, prophet_prediction)
#axis[0, 0].set_title("FB Prophet")

#Linear Regression Graph
#axis[0, 1].plot(date_list, lr_prediction)
#axis[0, 1].set_title("Linear Regression")

#Support Vector Regression Graph
#axis[1, 0].plot(date_list, svr_prediction)
#axis[1, 0].set_title("Support Vector Regression")

#Random Forest Regression Graph
#axis[1, 1].plot(date_list, rfr_prediction)
#axis[1, 1].set_title("Random Forest Regression")

#Gradient Boosting Regression Graph
#axis[2, 0].plot(date_list, gbr_prediction)
#axis[2, 0].set_title("Gradient Boosting Regression")

#Kernel Ridge Regression Graph
#axis[2, 1].plot(date_list, kr_prediction)
#axis[2, 1].set_title("Kernel Ridge Regression")

#plt.show()