import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk

#Creating the window
root = tk.Tk()
root.title("Stock Price Advisor")
canvas = tk.Canvas(root, width=600, height=800)
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

send_button = Button(root, text="Get advise", bg="blue", fg="white", font=('Helvetica', 12, 'bold'))
send_button.grid(row=6, column=0, padx=10, pady=10)

#Configuring rows and columns
root.rowconfigure(7, weight=1)
root.columnconfigure(7, weight=1)


root.mainloop()
