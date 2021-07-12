import pandas as pd
from fbprophet import Prophet
import datetime as dt
import pandas_datareader as web
import numpy as np
import tkinter as tk, tkinter
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
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

pd.options.mode.chained_assignment = None  # default='warn'
global avg, dates

def predict_stock():
    yf.pdr_override()

    # Getting the stock name
    company = str(stock_name.get())
    predict_day = int(prediction_day.get())

    # Initializing dates
    start = dt.datetime(2016, 1, 1)
    end = dt.datetime.now()

    # Getting stock prices from yahoo
    #data = web.DataReader(company, 'yahoo', start, end)
    data = pdr.get_data_yahoo(company, data_source='yahoo', start=start, end=end)

    # FB prophet Prediction
    data.to_csv('test.csv')

    df = pd.read_csv("test.csv")

    columns = ["Date", "Close"]
    ndf = pd.DataFrame(df, columns=columns)

    # Renaming dataframe for FBProphet
    prophet_df = ndf.rename(columns={"Date": "ds", "Close": "y"})

    # FBProphet Predictions
    m = Prophet()
    m.fit(prophet_df)

    prediction_days = 30  # Predict n days into the future

    future = m.make_future_dataframe(periods=prediction_days)
    forecast = m.predict(future)

    # Saving results
    forecast.to_csv('forecast.csv')

    # Getting real and predicted prices
    val = forecast['yhat'].values
    today = data['Close'].values
    prophet_prediction = val[-prediction_days:]

    # Getting today's price
    today_price = today[len(today) - 1]

    ##Linear regression calculations

    df_new = data[["Close"]]

    df_new["Prediction"] = df_new[["Close"]].shift(-prediction_days)

    # Creating independent set

    x = np.array(df_new.drop(["Prediction"], 1))

    # Remove last n rows from x
    x = x[:-prediction_days]

    # Creating dependent data set
    y = np.array(df_new["Prediction"])

    # Remove last n rows from y
    y = y[:-prediction_days]

    # Split %75 training %25 testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

    # Creating and training the linear regression model
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    # Creating and training the support vector regression model
    svr = SVR(kernel="rbf", C=1e3, gamma=0.1)
    svr.fit(x_train, y_train)

    # Creating and training random forest regressor model
    rfr = RandomForestRegressor()
    rfr.fit(x_train, y_train)

    # Creating and training gradient boosting regressor model
    gbr = GradientBoostingRegressor()
    gbr.fit(x_train, y_train)

    # Creating and training kernel ridge regressor model
    kr = KernelRidge()
    kr.fit(x_train, y_train)

    # Set x_forecast equal to last 30 rows of the "Close" prices
    x_forecast = np.array(df_new.drop(["Prediction"], 1))[-prediction_days:]

    # Printing linear regression model results for n days
    lr_prediction = lr.predict(x_forecast)
    print(lr_prediction)

    # Printing support vector regression model results for n days
    svr_prediction = svr.predict(x_forecast)
    print(svr_prediction)

    # Printing random forest regression results for n days
    rfr_prediction = rfr.predict(x_forecast)
    print(rfr_prediction)

    # Printing gradient boosting regression results for n days
    gbr_prediction = gbr.predict(x_forecast)
    print(gbr_prediction)

    # Printing kernel ridge regression results for n days
    kr_prediction = kr.predict(x_forecast)
    print(kr_prediction)

    # Converting array to the numpy array to get the average
    prophet_prediction = np.array(prophet_prediction)
    lr_prediction = np.array(lr_prediction)
    svr_prediction = np.array(svr_prediction)
    rfr_prediction = np.array(rfr_prediction)
    gbr_prediction = np.array(gbr_prediction)
    kr_prediction = np.array(kr_prediction)

    # Calculating the avarage for last 30 days
    average = (
                          prophet_prediction + lr_prediction + svr_prediction + rfr_prediction + gbr_prediction + kr_prediction) / 6.0

    # Converting it back to normal array
    np.asarray(average)
    print(average)

    # Getting predicted price from different regression models
    average_prediction = average[len(average) - 1 - prediction_days + predict_day]

    # Getting dates today to 30 days later
    date_list = pd.date_range(end=dt.datetime.today() + dt.timedelta(days=30), periods=30).to_pydatetime().tolist()

    return average_prediction, average, date_list

def plot_prediction(average, date_list):

    plt.plot_date(date_list, average, linestyle="solid")
    plt.gcf().autofmt_xdate()
    plt.xlabel("Date")
    plt.ylabel("Stock price")
    plt.show()

def display():
    predicted_price, avg, dates = predict_stock()

    #price_display = tk.Text(master=root, height=1, width=30)
    #price_display.grid(row=7, column=0)

    #price_display.insert(tk.END, "Predicted price: " + str(int(predicted_price)) + "$")

    prediction_day_label = Label(root, text="Predicted price: " + str(int(predicted_price)) + "$", font=('Helvetica', 12, 'bold'), bg='#038b9e', fg="orange")

    prediction_day_label.grid(row=7, column=0)

    window = Tk()
    window.wm_title("Stock Prediction Graph")
    window.minsize(1400, 400)

    fig = Figure(figsize=(5,4), dpi=100)
    fig.add_subplot(111).plot_date(dates, avg, linestyle="solid")

    canvas2 = FigureCanvasTkAgg(fig, master=window)
    canvas2.draw()
    canvas2.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(canvas2, window)
    toolbar.update()
    canvas2.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)


#Creating the window
root = tk.Tk()
root.title("Stock Price Advisor")
canvas = tk.Canvas(root, width=600, height=400)
canvas.grid(columnspan=1, rowspan=10)

bg = Image.open("background.png")
bg = ImageTk.PhotoImage(bg)
background = Label(root, image=bg)
background.place(x=0, y=0, relwidth=1, relheight=1)

#Logo display
logo = Image.open("stock.png")
logo = ImageTk.PhotoImage(logo)
logo_label = tk.Label(image=logo)
logo_label.image = logo
logo_label.grid(row=0, column=0)

#Program name
program_name = tk.Label(root, text="Stock Price Advisor", font=('Helvetica', 24, 'bold'), bg='#038b9e', fg="white")
program_name.grid(row=1, column=0)

#Text boxes
stock_name_label = Label(root, text="Stock name:", font=('Helvetica', 12, 'bold'), bg='#038b9e', fg="white")
stock_name_label.grid(row=2, column=0)

stock_name = Entry(root, width=20)
stock_name.grid(row=3, column=0)

prediction_day_label = Label(root, text="Prediction Day (Max=30):", font=('Helvetica', 12, 'bold'), bg='#038b9e', fg="white")
prediction_day_label.grid(row=4, column=0)

prediction_day = Entry(root, width=20)
prediction_day.grid(row=5, column=0)

#Buttons

send_button = Button(root, text="Get advise", bg="blue", fg="white", font=('Helvetica', 12, 'bold'), command=display)
send_button.grid(row=6, column=0, padx=10, pady=10)

#Configuring rows and columns
root.rowconfigure(7, weight=1)
root.columnconfigure(7, weight=1)


root.mainloop()
