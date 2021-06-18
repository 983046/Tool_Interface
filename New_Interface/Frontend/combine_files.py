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
from New_Interface.Frontend.user_dashboard import UserDashboard
import pandas as pd

class CombineFiles(UserDashboard):
    def __init__(self, window, dashboard_selection):
        self.window = window
        self.window.geometry("1366x720+0+0")
        self.window.title("Combine Data Dashboard")
        self.window.resizable(False, False)
        self.admin_dashboard_frame = ImageTk.PhotoImage \
            (file='images\\user_frame.png')
        self.image_panel = Label(self.window, image=self.admin_dashboard_frame)
        self.image_panel.pack(fill='both', expand='yes')

        # ============================Welcome Dashboard==============================
        self.txt = "Welcome to Combine"
        self.heading = Label(self.window, text=self.txt, font=('yu gothic ui', 20, "bold"), bg="white",
                             fg='black',
                             relief=FLAT)
        self.heading.place(x=570, y=43)

        # ============================Date and time==============================
        self.date_time_image = Label(self.window, bg="white")
        self.date_time = Label(self.window)
        self.date_time.place(x=80, y=45)
        self.time_running()

        # ============================Exit button===============================
        self.exit = ImageTk.PhotoImage \
            (file='images\\exit_button.png')
        self.exit_button = Button(self.window, image=self.exit,
                                  font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                  , borderwidth=0, background="white", cursor="hand2", command=self.click_exit)
        self.exit_button.place(x=1260, y=55)

        self.lb = Listbox(self.window, width=50, height=3)
        self.lb.place(x=150, y=170)
        self.listbox_object = dashboard_selection
        for item in self.listbox_object:
            self.file_name = os.path.basename(item)
            self.lb.insert(END, self.file_name)

        self.common_values = read_file(self)
        self.lb_common = Listbox(self.window, width=50, height=3)
        self.lb_common.place(x=600, y=170)
        for self.item in self.common_values:
            self.lb_common.insert(END, self.item)


def read_file(self):
    read_file_and_columns = []
    #Columns_list = []
    for file in self.listbox_object:
        if file.endswith('.XPT'):
            file = pd.read_sas(file).columns
            file = file.to_numpy().astype('str').tolist()
            read_file_and_columns.append(file)
        elif file.endswith('.CSV'):
            file = pd.read_sas(file)
            read_file_and_columns.append(file.columns)
        elif file.endswith('.XLSX'):
            file = pd.read_excel(file, index_col=0)
            read_file_and_columns.append(file.columns)
        else:
            messagebox.showerror("Program not optimized", "The program is not optimized for this filename type: "
                                                          "\n {}".format(file))

    return check_similarity(read_file_and_columns)



def check_similarity(read_file_and_columns):
    rows = []
    columns = []
    for i in read_file_and_columns:
        for j in i:
            columns.append(j)
        rows.append(columns)
    np_array = np.array(rows)
    common = common_field(np_array)
    print(np_array)
    print(common)
    return common


def common_field(dataset_list):
    """
    Take a list of columns
    Example A = ['1','2','3'] ; B = ['10','20','3'];C = ['2','3','333']; DatasetList = [A,B,C]
    :param DatasetList: A list of dataset's columns

    :return: the  common field
    """
    for i , data in enumerate(dataset_list,0):
        if i ==0 :
            flatted= data
        else:
            flatted = np.concatenate((flatted,data))

    u, c = np.unique(dataset_list, return_counts=True)
    dup = u[c == dataset_list.shape[0]]
    print(dup)
    return dup


def win():
    window = tk.ThemedTk()
    window.get_themes()
    window.set_theme("arc")
    CombineFiles(window)
    window.mainloop()


if __name__ == '__main__':
    win()
