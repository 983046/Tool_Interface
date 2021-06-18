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
import pandas as pd

from New_Interface.Frontend.user_dashboard import UserDashboard


class ConcatenateFiles(UserDashboard):
    def __init__(self, window, dashboard_selection):
        self.window = window
        self.window.geometry("1366x720+0+0")
        self.window.title("Concatenate Data Dashboard")
        self.window.resizable(False, False)
        self.admin_dashboard_frame = ImageTk.PhotoImage \
            (file='images\\user_frame.png')
        self.image_panel = Label(self.window, image=self.admin_dashboard_frame)
        self.image_panel.pack(fill='both', expand='yes')

        # ============================Welcome Dashboard==============================
        self.txt = "Welcome to Concatenate"
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

        self.set_frame()

        self.lb_selection = Listbox(self.window, width=50, height=3)
        self.lb_selection.place(x=150, y=250)
        self.listbox_object = dashboard_selection
        for item in self.listbox_object:
            self.file_name = os.path.basename(item)
            self.lb_selection.insert(END, self.file_name)

        self.common_values = self.extract_common_features()
        self.lb_common = Listbox(self.window, width=50, height=3)
        self.lb_common.place(x=580, y=250)
        for self.item in self.common_values:
            self.lb_common.insert(END, self.item)

        self.concatenate_image = ImageTk.PhotoImage \
            (file='images\\concatenate_file_button_red.png')
        self.concatenate_files_button_red = Button(self.window, image=self.concatenate_image,
                                                   font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                                   activebackground="white"
                                                   , borderwidth=0, background="white", cursor="hand2",
                                                   command=self.click_concatenate_files)
        self.concatenate_files_button_red.place(x=1000, y=325)

    def set_frame(self):
        add_frame = Frame(self.window)
        add_frame.place(x=46, y=115)

        self.add_dashboard_frame = ImageTk.PhotoImage \
            (file='images\\concatenate_frame.png')
        self.add_panel = Label(add_frame, image=self.add_dashboard_frame, bg="white")
        self.add_panel.pack(fill='both', expand='yes')

        self.add = ImageTk.PhotoImage \
            (file='images\\add_button_red.png')
        self.add_button = Button(self.window, image=self.add,
                                 font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                 , borderwidth=0, background="white", cursor="hand2", command=self.click_add)
        self.add_button.place(x=622, y=542)

    def click_concatenate_files(self):
        selected = self.lb_common.curselection()
        selected_files = self.extract_common_features()
        one_element = ''
        for index in selected[::-1]:
            one_element = selected_files[index]

        print(one_element)
        files = self.read_selected_files()
        merged_dataset = pd
        for i, dataset in enumerate(files):
            if i == 0:
                merged_dataset = dataset
            else:
                merged_dataset = pd.merge(merged_dataset,
                                          dataset, on=one_element)
        messagebox.showinfo("Merged Data", "Data was merged on: \n {}".format(one_element))

        # todo Need to do something with the data, (i.e. save)


    def extract_common_features(self):
        files = self.read_selected_files()
        read_file_and_columns = []
        for file in files:
            file = file.columns
            read_file_and_columns.append(file)

        return self.common_field(read_file_and_columns)

    def common_field(self, dataset_list):
        """
        Take a list of columns
        Example A = ['1','2','3'] ; B = ['10','20','3'];C = ['2','3','333']; DatasetList = [A,B,C]
        :param dataset_list:
        :param DatasetList: A list of dataset's columns

        :return: the  common field
        """
        for i, data in enumerate(dataset_list, 0):
            if i == 0:
                flatted = data
            else:
                flatted = np.concatenate((flatted, data))

        u, c = np.unique(flatted, return_counts=True)
        dup = u[c == len(dataset_list)]
        return dup


def win():
    window = tk.ThemedTk()
    window.get_themes()
    window.set_theme("arc")
    ConcatenateFiles(window)
    window.mainloop()


if __name__ == '__main__':
    win()
