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

class ExtractionDashboard(UserDashboard, RunModel):
    def __init__(self, window):
        self.window = window
        windowWidth = self.window.winfo_reqwidth()
        windowHeight = self.window.winfo_reqheight()
        positionRight = int(self.window.winfo_screenwidth() / 6 - windowWidth / 2)
        positionDown = int(self.window.winfo_screenheight() / 5 - windowHeight / 2)
        self.window.geometry("+{}+{}".format(positionRight, positionDown))
        self.window.title("Dashboard")
        self.window.resizable(False, False)
        self.admin_dashboard_frame = ImageTk.PhotoImage \
            (file='images\\user_frame.png')
        self.image_panel = Label(self.window, image=self.admin_dashboard_frame)
        self.image_panel.pack(fill='both', expand='yes')

        self.set_frame()

    def set_frame(self):
        add_frame = Frame(self.window)
        add_frame.place(x=35, y=159)

        self.model_dashboard_frame = ImageTk.PhotoImage \
            (file='images\\extraction_frame.png')
        self.model_panel = Label(add_frame, image=self.model_dashboard_frame, bg="white")
        self.model_panel.pack(fill='both', expand='yes')

        self.next_file = ImageTk.PhotoImage \
            (file='images\\next_button_red.png')
        self.next_file_button_red = Button(self.window, image=self.next_file,
                                           font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                           activebackground="white"
                                           , borderwidth=0, background="white", cursor="hand2", command=self.label_section)
        self.next_file_button_red.place(x=85, y=350)

        self.feature = ImageTk.PhotoImage \
            (file='images\\feature_button_red.png')
        self.feature_button_blue = Button(self.window, image=self.feature,
                                              font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                              activebackground="white"
                                              , borderwidth=0, background="white", cursor="hand2",
                                              command=self.click_next_file)
        self.feature_button_blue.place(x=150, y=24)


        self.add = ImageTk.PhotoImage \
            (file='images\\add_button_red.png')
        self.add_button = Button(self.window, image=self.add,
                                 font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                 , borderwidth=0, background="white", cursor="hand2", command=self.click_add)
        self.add_button.place(x=22, y=24)

        self.extract = ImageTk.PhotoImage \
            (file='images\\extract_button_blue.png')
        self.extract_button_red = Button(self.window, image=self.extract,
                                        font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                        activebackground="white"
                                        , borderwidth=0, background="white", cursor="hand2", command=self.run_model_frame)
        self.extract_button_red.place(x=278, y=24)

        self.selected_shape = ImageTk.PhotoImage \
            (file='images\\selected_shape.png')
        self.selected_shape_red = Button(self.window, image=self.selected_shape,
                                         font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                         , borderwidth=0, background="white", cursor="hand2")
        self.selected_shape_red.place(x=313, y=120)

        # self.model = ImageTk.PhotoImage \
        #     (file='images\\model_button_blue.png')
        # self.model_button_red = Button(self.window, image=self.model,
        #                                 font=("yu gothic ui", 13, "bold"), relief=FLAT,
        #                                 activebackground="white"
        #                                 , borderwidth=0, background="white", cursor="hand2", command=self.run_model_frame)
        # self.model_button_red.place(x=278, y=24)

        self.files = self.read_folder(FOLDER_URL)
        if len(self.files) != 0:
            self.chosen_file = StringVar(self.window)
            self.combo_chosen_file = OptionMenu(self.window, self.chosen_file,
                                          *self.files)
            self.combo_chosen_file.configure(width=11)
            self.combo_chosen_file.place(x=123, y=280)

        self.next_frame = ImageTk.PhotoImage \
            (file='images\\next_button_red.png')
        self.next_frame_button_red = Button(self.window, image=self.next_frame,
                                           font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                           activebackground="white"
                                           , borderwidth=0, background="white", cursor="hand2", command=self.pressed_model_frame)
        self.next_frame_button_red.configure(state="disabled")
        self.next_frame_button_red.place(x=1000, y=450)

    def label_section(self):
        file_name = self.chosen_file.get()
        file_url = FOLDER_URL + '\\' + file_name
        read_file = self.read_single_file(file_url)
        file_columns = read_file.columns
        if len(file_columns) != 0:
            self.label = StringVar(self.window)
            self.combo_label = OptionMenu(self.window, self.label, *file_columns)
            self.combo_label.configure(width=21)
            self.combo_label.place(x=307, y=443)
        self.combo_chosen_file.configure(state="disabled")

        self.cb = IntVar()
        checkbox = Checkbutton(self.window, text="Deeper label?", variable=self.cb, onvalue=1, offvalue=0, command=self.isChecked)
        checkbox.place(x=578, y=445)

        self.specific_value = StringVar()
        self.inFileTxt = Entry(self.window, textvariable=self.specific_value)
        self.inFileTxt.configure(state="disabled",width='27')
        self.inFileTxt.place(x=705, y=448)

        self.next_frame_button_red.configure(state="active")

    def pressed_model_frame(self):
        label = self.label.get()
        if label == '':
            messagebox.showinfo("Label not selected", "Need to select a label")
        else:
            pass
            #todo run different Model_Frame here.

    def run_model_frame(self):
        win = Toplevel()
        from New_Interface.Frontend import extraction_dashboard
        extraction_dashboard.ExtractionDashboard(win)
        self.window.withdraw()
        win.deiconify()

    def deeper_label(self):
        # self.specific_value = read_file[label]
        # self.chosen_specific_value = StringVar(self.window)
        # self.combo_specific_value = OptionMenu(self.window, self.chosen_specific_value, *self.specific_value)
        # self.combo_specific_value.configure(state="disabled")
        # self.combo_specific_value.place(x=500, y=250)

        self.model_value = ['SVM', 'Regression', 'MLPRegressor']
        self.chosen_model_value = StringVar(self.window)
        combo_model_value = OptionMenu(self.window, self.chosen_model_value, *self.model_value)
        combo_model_value.place(x=120, y=380)

        self.applied_pca = ['Yes', 'No']
        self.chosen_applied_pca= StringVar(self.window)
        combo_applied_pca = OptionMenu(self.window, self.chosen_applied_pca,
                                *self.applied_pca)
        combo_applied_pca.place(x=350, y=380)

        self.apply = ImageTk.PhotoImage \
            (file='images\\apply_button_red.png')
        self.apply_button = Button(self.window, image=self.apply,
                                        font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                        , borderwidth=0, background="white", cursor="hand2", command=self.chosen_model)
        self.apply_button.place(x=600, y=380)

        self.explanation = ImageTk.PhotoImage \
            (file='images\\explanation_button_red.png')
        self.explanation_button = Button(self.window, image=self.explanation,
                                   font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                   , borderwidth=0, background="white", cursor="hand2", command=self.get_importance)
        self.explanation_button.place(x=800, y=380)

    def isChecked(self):
        if self.cb.get() == 1:
            self.inFileTxt.configure(state="normal")
        elif self.cb.get() == 0:
            self.inFileTxt.configure(state="disabled")
        else:
            messagebox.showerror('PythonGuides', 'Something went wrong!')

    def chosen_model(self):
        label = self.label.get()

        model = self.chosen_model_value.get()
        pca = self.chosen_applied_pca.get()
        file_name = self.chosen_file.get()
        file_url = FOLDER_URL + '\\' + file_name
        read_file = self.read_single_file(file_url)

        if self.cb.get() == 1:
            deeper_label = self.inFileTxt.get()
            features, chosen_label = self.feature_deeper_label(read_file, label, deeper_label)
        else:
            features, chosen_label = self.feature_label(read_file, label)

        if pca == 'Yes':
            if model == 'SVM':
                self.pca_svm(features, chosen_label)
            elif model == 'Regression':
                self.pca_regression(features, chosen_label)
            elif model == 'MLPRegressor':
                pass
        else:
            if model == 'SVM':
                #todo this is not supported
                self.training_type, X_train = self.svm(features,chosen_label)
            elif model == 'Regression':
                self.training_type, self.X_train = self.regression(features, chosen_label)
            elif model == 'MLPRegressor':
                self.MLPRegression(features, chosen_label)



    def get_importance(self):
        self.importance_plot(self.training_type, self.X_train)


    def click_add(self):
        pass



def win():
    window = tk.ThemedTk()
    window.get_themes()
    window.set_theme("arc")
    ExtractionDashboard(window)
    window.mainloop()


if __name__ == '__main__':
    win()




