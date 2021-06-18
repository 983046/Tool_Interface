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
    def __init__(self, window, dashboard_selection):
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
            self.lb.insert(END, self.file_name)

        read_file(self)


def read_file(self):
    read_file_and_columns = []
    Columns_list = []
    for file in self.listbox_object:
        if file.endswith('.XPT'):
            file = pd.read_sas(file)
            Columns_list.append(ile.columns)
            #file = file.columns.to_numpy().astype('str').tolist()
            read_file_and_columns.append(file)

        elif file.endswith('.CSV'):
            file = pd.read_sas(file)
            read_file_and_columns.append(file.columns)
            Columns_list.append(ile.columns)
        elif file.endswith('.XLSX'):
            file = pd.read_excel(file, index_col=0)
            read_file_and_columns.append(file.columns)
            Columns_list.append(ile.columns)
        else:
            messagebox.showerror("Program not optimized", "The program is not optimized for this filename type: "
                                                          "\n {}".format(file))
    check_similarity(read_file_and_columns)
    #here calll the function with Columns_list

values = []
similar_values = []

def check_similarity(read_file_and_columns):
    rows=[]
    columns = []
    for i in read_file_and_columns:
        for j in i:
            columns.append(j)
        rows.append(columns)
    np_array = np.array(rows)
    common = common_Field(np_array)
    print(np_array)
    print(common)

def common_Field(datasetList):
    """
    Take a list of columns
    Example A = ['1','2','3'] ; B = ['10','20','3'];C = ['2','3','333']; DatasetList = [A,B,C]
    :param DatasetList: A list of dataset's columns

    :return: the  common field
    """
    u, c = np.unique(datasetList, return_counts=True)
    dup = u[c == datasetList.shape[0]]
    return dup


def win():
    window = tk.ThemedTk()
    window.get_themes()
    window.set_theme("arc")
    CombineFiles(window)
    window.mainloop()


if __name__ == '__main__':
    win()
