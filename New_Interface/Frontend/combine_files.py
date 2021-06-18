import os
import random
import shutil
import time
from collections import Iterable
from itertools import chain
from tkinter import *
from tkinter.filedialog import askopenfilename
import numpy as np

from ttkthemes import themed_tk as tk
from tkinter import ttk, messagebox
from PIL import ImageTk
from PIL import Image
import New_Interface.Frontend.user_dashboard as user_dashboard
import pandas as pd

class CombineFiles:
    def __init__(self,window, dashboard_selection):
        self.window = window
        self.window.geometry("1366x720+0+0")
        self.window.title("Combine Data Dashboard")
        self.window.resizable(False, False)
        self.admin_dashboard_frame = ImageTk.PhotoImage \
            (file='images\\user_frame.png')

        self.reg_frame = Frame(self.window, bg="#ffffff", width=1300, height=680)
        self.reg_frame.place(x=30, y=30)

        self.txt = "Combine Data"
        self.heading = Label(self.reg_frame, text=self.txt, font=('yu gothic ui', 30, "bold"), bg="white",
                             fg='black',
                             bd=5,
                             relief=FLAT)
        self.heading.place(x=350, y=0, width=600)

        self.cred_frame = LabelFrame(self.reg_frame, text="", bg="white", fg="#4f4e4d", height=140,
                                     width=350, borderwidth=2.4,
                                     font=("yu gothic ui", 13, "bold"))
        self.cred_frame.config(highlightbackground="red")
        self.cred_frame.place(x=100, y=100)

        self.selected_data_label = Label(self.cred_frame, text="Selected Data: ", bg="white", fg="#4f4e4d",
                                    font=("yu gothic ui", 13, "bold"))
        self.selected_data_label.place(x=15, y=10)

        self.lb = Listbox(self.window, width=50, height=3)
        self.lb.place(x=150, y=170)
        self.listbox_object = dashboard_selection
        for item in self.listbox_object:
            self.file_name = os.path.basename(item)
            self.lb.insert(END,self.file_name)

        read_file(self)

def read_file(self):
    read_file_and_columns = []
    for file in self.listbox_object:
        if file.endswith('.XPT'):
            file = pd.read_sas(file)
            file = file.columns.to_numpy().tolist()
            read_file_and_columns.append(file)
            check_similarity(read_file_and_columns)
            read_file_and_columns.clear()
        elif file.endswith('.CSV'):
            file = pd.read_sas(file)
            read_file_and_columns.append(file.columns)
        elif file.endswith('.XLSX'):
            file = pd.read_excel(file, index_col=0)
            read_file_and_columns.append(file.columns)
        else:
            messagebox.showerror("Program not optimized", "The program is not optimized for this filename type: "
                                                          "\n {}".format(file))

def check_similarity(read_file_and_columns):
    values = []
    similar_values = []
    if not values:
        for elements in read_file_and_columns:
            values.append(elements)
    else:
        for elements in read_file_and_columns:
            if elements in values:
                similar_values.append(elements)
            else:
                values.append(elements)
    print(similar_values)











def win():
    window = tk.ThemedTk()
    window.get_themes()
    window.set_theme("arc")
    CombineFiles(window)
    window.mainloop()


if __name__ == '__main__':
    win()