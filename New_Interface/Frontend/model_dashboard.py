import os

import time
from tkinter import *
from tkinter.filedialog import askopenfilename

from ttkthemes import themed_tk as tk
from tkinter import ttk, messagebox
from PIL import ImageTk
import pandas as pd
from pandastable import Table, TableModel
import numpy as np
from New_Interface.Frontend.run_model import RunModel
from New_Interface.Frontend.user_dashboard import UserDashboard
FOLDER_URL = r'C:\Users\marci\OneDrive\Other\Desktop\Shared\Tool_Interface\New_Interface\Frontend\joined_files'

class ModelDashboard(UserDashboard, RunModel):
    def __init__(self, window):
        self.window = window
        self.window.title("Concatenate Data Dashboard")
        self.admin_dashboard_frame = ImageTk.PhotoImage \
            (file='images\\user_frame.png')
        self.image_panel = Label(self.window, image=self.admin_dashboard_frame)
        self.image_panel.pack(fill='both', expand='yes')

        # ============================Welcome Dashboard==============================
        self.txt = "Welcome to features"
        self.heading = Label(self.window, text=self.txt, font=('yu gothic ui', 20, "bold"), bg="white",
                             fg='black',
                             relief=FLAT)
        self.heading.place(x=570, y=43)

        # ============================Date and time==============================
        self.date_time_image = Label(self.window, bg="white")
        self.date_time = Label(self.window)
        self.date_time.place(x=80, y=45)
        self.time_running()

        self.set_frame()

    def set_frame(self):
        self.files = self.read_folder(FOLDER_URL)
        if len(self.files) != 0:
            self.chosen_file = StringVar(self.window)
            self.combo_chosen_file = OptionMenu(self.window, self.chosen_file,
                                          *self.files)
            self.combo_chosen_file.place(x=120, y=250)

        self.next = ImageTk.PhotoImage \
            (file='images\\apply_button_red.png')
        self.next_button = Button(self.window, image=self.next,
                                   font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                   , borderwidth=0, background="white", cursor="hand2", command=self.label_section)
        self.next_button.place(x=250, y=220)

        # self.model_value = ['SVM', 'Regression']
        # self.chosen_model_value = StringVar(self.window)
        # combo_model_value = OptionMenu(self.window, self.chosen_model_value,
        #                         *self.model_value)
        # combo_model_value.place(x=407, y=250)
        #
        # self.applied_pca = ['Yes', 'No']
        # self.chosen_applied_pca= StringVar(self.window)
        # combo_applied_pca = OptionMenu(self.window, self.chosen_applied_pca,
        #                         *self.applied_pca)
        # combo_applied_pca.place(x=578, y=250)
        #
        # self.apply = ImageTk.PhotoImage \
        #     (file='images\\apply_button_red.png')
        # self.apply_button = Button(self.window, image=self.apply,
        #                                 font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
        #                                 , borderwidth=0, background="white", cursor="hand2")
        # self.apply_button.place(x=800, y=220)

    def label_section(self):
        file_name = self.chosen_file.get()
        file_url = FOLDER_URL + '\\' + file_name
        read_file = self.read_single_file(file_url)
        file_columns = read_file.columns
        if len(file_columns) != 0:
            self.label = StringVar(self.window)
            self.combo_label = OptionMenu(self.window, self.label, *file_columns)
            self.combo_label.place(x=220 , y=250)
        self.next_button.destroy()
        self.combo_chosen_file.configure(state="disabled")

        self.next = ImageTk.PhotoImage \
            (file='images\\apply_button_red.png')
        self.next_button = Button(self.window, image=self.next,
                                   font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                   , borderwidth=0, background="white", cursor="hand2", command=self.deeper_label)
        self.next_button.place(x=350, y=220)

    def deeper_label(self):
        self.combo_label.configure(state="disabled")
        self.next_button.destroy()

        self.cb = IntVar()
        checkbox = Checkbutton(self.window, text="Deeper labeling", variable=self.cb, onvalue=1, offvalue=0, command=self.isChecked)
        checkbox.place(x=350, y=250)

        # self.specific_value = read_file[label]
        # self.chosen_specific_value = StringVar(self.window)
        # self.combo_specific_value = OptionMenu(self.window, self.chosen_specific_value, *self.specific_value)
        # self.combo_specific_value.configure(state="disabled")
        # self.combo_specific_value.place(x=500, y=250)

        self.specific_value = StringVar()
        self.inFileTxt = Entry(self.window, textvariable=self.specific_value)
        self.inFileTxt.configure(state="disabled")
        self.inFileTxt.place(x=500, y=250)

        self.model_value = ['SVM', 'Regression']
        self.chosen_model_value = StringVar(self.window)
        combo_model_value = OptionMenu(self.window, self.chosen_model_value, *self.model_value)
        combo_model_value.place(x=120, y=400)

        self.applied_pca = ['Yes', 'No']
        self.chosen_applied_pca= StringVar(self.window)
        combo_applied_pca = OptionMenu(self.window, self.chosen_applied_pca,
                                *self.applied_pca)
        combo_applied_pca.place(x=250, y=400)

        self.apply = ImageTk.PhotoImage \
            (file='images\\apply_button_red.png')
        self.apply_button = Button(self.window, image=self.apply,
                                        font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                        , borderwidth=0, background="white", cursor="hand2", command=self.chosen_model)
        self.apply_button.place(x=350, y=400)


    def isChecked(self):
        if self.cb.get() == 1:
            self.inFileTxt.configure(state="normal")
        elif self.cb.get() == 0:
            self.inFileTxt.configure(state="disabled")
        else:
            messagebox.showerror('PythonGuides', 'Something went wrong!')



    def chosen_model(self):
        label = self.label.get()
        deeper_label = self.inFileTxt.get()

        model = self.chosen_model_value.get()
        pca = self.chosen_applied_pca.get()
        file_name = self.chosen_file.get()
        file_url = FOLDER_URL + '\\' + file_name
        read_file = self.read_single_file(file_url)
        features = None

        if self.cb.get() == 1:
            features, chosen_label = self.feature_deeper_label(read_file, label, deeper_label)
        else:
            feature, chosen_label = self.feature_label(read_file, label)

        if pca == 'Yes':
            if model == 'SVM':
                self.pca_svm(features, chosen_label)
            elif model == 'Regression':
                self.pca_regression()
        else:
            if model == 'SVM':
                self.svm
            elif model == 'Regression':
                self.regression()


def win():
    window = tk.ThemedTk()
    window.get_themes()
    window.set_theme("arc")
    ModelDashboard(window)
    window.mainloop()


if __name__ == '__main__':
    win()
