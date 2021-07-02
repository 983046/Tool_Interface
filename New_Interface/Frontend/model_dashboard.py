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
        self.window.title("Concatenate Model Dashboard")
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
        add_frame = Frame(self.window)
        add_frame.place(x=48, y=116)

        self.next = ImageTk.PhotoImage \
            (file='images\\apply_button_red.png')
        self.next_button = Button(self.window, image=self.next,
                                   font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                   , borderwidth=0, background="white", cursor="hand2", command=self.label_section)
        self.next_button.place(x=250, y=250)

        self.next_feature = ImageTk.PhotoImage \
            (file='images\\feature_button_red.png')
        self.next_feature_button_blue = Button(self.window, image=self.next_feature,
                                              font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                              activebackground="white"
                                              , borderwidth=0, background="white", cursor="hand2",
                                              command=self.click_next_file)
        self.next_feature_button_blue.place(x=477, y=583)

        self.model_dashboard_frame = ImageTk.PhotoImage \
            (file='images\\model_frame.png')
        self.model_panel = Label(add_frame, image=self.model_dashboard_frame, bg="white")
        self.model_panel.pack(fill='both', expand='yes')

        self.add = ImageTk.PhotoImage \
            (file='images\\add_button_red.png')
        self.add_button = Button(self.window, image=self.add,
                                 font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                 , borderwidth=0, background="white", cursor="hand2", command=self.click_add)
        self.add_button.place(x=622, y=542)

        self.model = ImageTk.PhotoImage \
            (file='images\\model_button_blue.png')
        self.model_button_red = Button(self.window, image=self.model,
                                        font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                        activebackground="white"
                                        , borderwidth=0, background="white", cursor="hand2", command=self.run_model_frame)
        self.model_button_red.place(x=796, y=583)

        self.files = self.read_folder(FOLDER_URL)
        if len(self.files) != 0:
            self.chosen_file = StringVar(self.window)
            self.combo_chosen_file = OptionMenu(self.window, self.chosen_file,
                                          *self.files)
            self.combo_chosen_file.place(x=120, y=250)

    def label_section(self):
        file_name = self.chosen_file.get()
        file_url = FOLDER_URL + '\\' + file_name
        read_file = self.read_single_file(file_url)
        file_columns = read_file.columns
        if len(file_columns) != 0:
            self.label = StringVar(self.window)
            self.combo_label = OptionMenu(self.window, self.label, *file_columns)
            self.combo_label.place(x=340 , y=250)
        self.next_button.destroy()
        self.combo_chosen_file.configure(state="disabled")

        self.next = ImageTk.PhotoImage \
            (file='images\\apply_button_red.png')
        self.next_button = Button(self.window, image=self.next,
                                   font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                   , borderwidth=0, background="white", cursor="hand2", command=self.deeper_label)
        self.next_button.place(x=500, y=250)

    def run_model_frame(self):
        win = Toplevel()
        from New_Interface.Frontend import model_dashboard
        model_dashboard.ModelDashboard(win)
        self.window.withdraw()
        win.deiconify()

    def deeper_label(self):
        self.combo_label.configure(state="disabled")
        self.next_button.destroy()

        self.cb = IntVar()
        checkbox = Checkbutton(self.window, text="Deeper labeling", variable=self.cb, onvalue=1, offvalue=0, command=self.isChecked)
        checkbox.place(x=598, y=250)

        # self.specific_value = read_file[label]
        # self.chosen_specific_value = StringVar(self.window)
        # self.combo_specific_value = OptionMenu(self.window, self.chosen_specific_value, *self.specific_value)
        # self.combo_specific_value.configure(state="disabled")
        # self.combo_specific_value.place(x=500, y=250)

        self.specific_value = StringVar()
        self.inFileTxt = Entry(self.window, textvariable=self.specific_value)
        self.inFileTxt.configure(state="disabled")
        self.inFileTxt.place(x=800, y=250)

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
        deeper_label = self.inFileTxt.get()

        model = self.chosen_model_value.get()
        pca = self.chosen_applied_pca.get()
        file_name = self.chosen_file.get()
        file_url = FOLDER_URL + '\\' + file_name
        read_file = self.read_single_file(file_url)

        if self.cb.get() == 1:
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
        add_frame = Frame(self.window)
        add_frame.place(x=46, y=115)

        self.add_dashboard_frame = ImageTk.PhotoImage \
            (file='images\\add_frame.png')
        self.add_panel = Label(add_frame, image=self.add_dashboard_frame, bg="white")
        self.add_panel.pack(fill='both', expand='yes')

        self.add = ImageTk.PhotoImage \
            (file='images\\add_button_blue.png')
        self.add_button = Button(self.window, image=self.add,
                                 font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                 , borderwidth=0, background="white", cursor="hand2", command=self.click_add)
        self.add_button.place(x=622, y=542)

        self.add_file = ImageTk.PhotoImage \
            (file='images\\add_file_button_red.png')
        self.add_file_button_red = Button(self.window, image=self.add_file,
                                          font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                          , borderwidth=0, background="white", cursor="hand2",
                                          command=self.click_add_file)
        self.add_file_button_red.place(x=200, y=225)

        # self.yscroll = Scrollbar(self.window)
        # self.yscroll.pack(side=RIGHT, fill=Y)

        # self.file_values = self.read_folder(ADDED_FILES)
        self.lb = Listbox(self.window, width=70, height=20, selectmode=MULTIPLE)
        self.lb.place(x=472, y=160)

        # self.yscroll = Scrollbar(command=self.lb.yview, orient=VERTICAL)
        # self.yscroll.place(x=900, y=160)

        self.remove_file = ImageTk.PhotoImage \
            (file='images\\remove_file_button_red.png')
        self.remove_file_button_red = Button(self.window, image=self.remove_file,
                                             font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                             , borderwidth=0, background="white", cursor="hand2",
                                             command=self.click_remove_file)
        self.remove_file_button_red.place(x=200, y=325)

        self.combine_file = ImageTk.PhotoImage \
            (file='images\\combine_file_button_red.png')
        self.combine_file_button_red = Button(self.window, image=self.combine_file,
                                              font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                              , borderwidth=0, background="white", cursor="hand2",
                                              command=self.click_combine_file)
        self.combine_file_button_red.place(x=1000, y=225)

        self.concatenate_file_user = ImageTk.PhotoImage \
            (file='images\\concatenate_file_button_red.png')
        self.concatenate_file_user_button_red = Button(self.window, image=self.concatenate_file_user,
                                                       font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                                       activebackground="white"
                                                       , borderwidth=0, background="white", cursor="hand2",
                                                       command=self.click_concatenate_file_user)
        self.concatenate_file_user_button_red.place(x=1000, y=325)

        self.next_file = ImageTk.PhotoImage \
            (file='images\\next_button_red.png')
        self.next_file_button_red = Button(self.window, image=self.next_file,
                                           font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                           activebackground="white"
                                           , borderwidth=0, background="white", cursor="hand2",
                                           command=self.click_next_file)
        self.next_file_button_red.place(x=1000, y=425)

        self.next_feature_button_blue.destroy()

        self.next_feature = ImageTk.PhotoImage \
            (file='images\\feature_button_red.png')
        self.next_feature_button_red = Button(self.window, image=self.next_feature,
                                               font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                               activebackground="white"
                                               , borderwidth=0, background="white", cursor="hand2",
                                               command=self.click_next_file)
        self.next_feature_button_red.place(x=477, y=583)

        self.model_button_red.configure(state="disabled")

        # self.model = ImageTk.PhotoImage \
        #     (file='images\\model_button_red.png')
        # self.model_button_red = Button(self.window, image=self.model,
        #                                 font=("yu gothic ui", 13, "bold"), relief=FLAT,
        #                                 activebackground="white"
        #                                 , borderwidth=0, background="white", cursor="hand2", command=self.run_model_frame)
        # self.model_button_red.place(x=796, y=583)



def win():
    window = tk.ThemedTk()
    window.get_themes()
    window.set_theme("arc")
    ModelDashboard(window)
    window.mainloop()


if __name__ == '__main__':
    win()




