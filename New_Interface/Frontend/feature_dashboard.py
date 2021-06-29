import os

import time
from tkinter import *
from tkinter.filedialog import askopenfilename

from sklearn import preprocessing
from ttkthemes import themed_tk as tk
from tkinter import ttk, messagebox
from PIL import ImageTk
import pandas as pd
from pandastable import Table, TableModel
import numpy as np

from New_Interface.Frontend.user_dashboard import UserDashboard
FOLDER_URL = r'C:\Users\marci\OneDrive\Other\Desktop\Shared\Tool_Interface\New_Interface\Frontend\joined_files'

class FeatureDashboard(UserDashboard):
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


    def apply_norm(self, dataset,file_url):
        #todo Add this into run_model.py instead.
        method = self.chosen_normalise.get()

        names = dataset.columns
        if method == 'Normalizer':
            scaler = preprocessing.Normalizer()
        elif method == 'MinMaxScaler':
            scaler = preprocessing.MinMaxScaler()
        elif method == 'StandardScaler':
            scaler = preprocessing.StandardScaler()
        else:
            return

        d = scaler.fit(dataset)
        scaled_np = d.transform(dataset)
        scaled_df = pd.DataFrame(scaled_np, columns=names)
        scaled_df.to_csv(file_url)

    def apply_pressed(self):
        choice = self.chosen_na_value.get()
        file_name = self.chosen_file.get()
        file_url = FOLDER_URL + '\\' + file_name

        read_file = self.read_single_file(file_url)

        if choice == 'mean':
            #todo working out the mean
            where_are_NaNs = np.isnan(read_file)
            read_file[where_are_NaNs] = read_file.mean(axis=1)
            read_file = read_file
            df = pd.DataFrame(read_file, columns=read_file.columns)
            df.to_csv(file_url)
        elif choice == 'zero':
            where_are_NaNs = np.isnan(read_file)
            read_file[where_are_NaNs] = 0
            read_file = read_file
            df = pd.DataFrame(read_file, columns=read_file.columns)
            df.to_csv(file_url)
        elif choice == 'one':
            where_are_NaNs = np.isnan(read_file)
            read_file[where_are_NaNs] = 1
            read_file = read_file
            df = pd.DataFrame(read_file, columns=read_file.columns)
            df.to_csv(file_url)
        elif choice == 'em':
            None

        self.apply_norm(read_file,file_url)

        self.display_dataframe_image = ImageTk.PhotoImage \
            (file='images\\edit_table_button_red.png')
        self.display_dataframe_button = Button(self.window, image=self.display_dataframe_image,
                                        font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                        , borderwidth=0, background="white", cursor="hand2", command=self.display_dataframe)
        self.display_dataframe_button.place(x=1000, y=220)

        # self.model = ImageTk.PhotoImage \
        #     (file='images\\model_button_red.png')
        # self.model_button_red = Button(self.window, image=self.model,
        #                                font=("yu gothic ui", 13, "bold"), relief=FLAT,
        #                                activebackground="white"
        #                                , borderwidth=0, background="white", cursor="hand2", command=self.run_model_frame)
        # self.model_button_red.place(x=796, y=583)

        self.model_button_red.configure(state="active")




    def set_frame(self):
        add_frame = Frame(self.window)
        add_frame.place(x=48, y=116)

        self.add = ImageTk.PhotoImage \
            (file='images\\add_button_red.png')
        self.add_button = Button(self.window, image=self.add,
                                 font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                 , borderwidth=0, background="white", cursor="hand2", command=self.click_add)
        self.add_button.place(x=622, y=542)

        self.next_feature = ImageTk.PhotoImage \
            (file='images\\feature_button_blue.png')
        self.next_feature_button_blue = Button(self.window, image=self.next_feature,
                                              font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                              activebackground="white"
                                              , borderwidth=0, background="white", cursor="hand2",
                                              command=self.run_feature_frame)
        self.next_feature_button_blue.place(x=477, y=583)

        self.feature_dashboard_frame = ImageTk.PhotoImage \
            (file='images\\feature_frame.png')
        self.feature_panel = Label(add_frame, image=self.feature_dashboard_frame, bg="white")
        self.feature_panel.pack(fill='both', expand='yes')


        self.files = self.read_folder(FOLDER_URL)
        if len(self.files) != 0:
            self.chosen_file = StringVar(self.window)
            comboLab = OptionMenu(self.window, self.chosen_file,
                                          *self.files)
            comboLab.place(x=120, y=250)

        self.na_values = ['mean', 'zero', 'one', 'em']
        if len(self.na_values) != 0:
            self.chosen_na_value = StringVar(self.window)
            comboLab = OptionMenu(self.window, self.chosen_na_value,
                                          *self.na_values)
            comboLab.place(x=407, y=250)

        self.normalise = ['Normalizer', 'MinMaxScaler', 'StandardScaler', 'em']
        self.chosen_normalise = StringVar(self.window)
        comboLab = OptionMenu(self.window, self.chosen_normalise,
                              *self.normalise)
        comboLab.place(x=578, y=250)


        self.apply = ImageTk.PhotoImage \
            (file='images\\apply_button_red.png')
        self.apply_button = Button(self.window, image=self.apply,
                                        font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                        , borderwidth=0, background="white", cursor="hand2", command=self.apply_pressed)
        self.apply_button.place(x=800, y=220)


        self.display_dataframe_image = ImageTk.PhotoImage \
            (file='images\\edit_table_button_grey.png')
        self.display_dataframe_button = Button(self.window, image=self.display_dataframe_image,
                                        font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                        , borderwidth=0, background="white", cursor="hand2", command=self.display_dataframe)
        self.display_dataframe_button.place(x=1000, y=220)

        self.model = ImageTk.PhotoImage \
            (file='images\\model_button_red.png')
        self.model_button_red = Button(self.window, image=self.model,
                                       font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                       activebackground="white"
                                       , borderwidth=0, background="white", cursor="hand2",
                                       command=self.run_model_frame)
        self.model_button_red.configure(state="disabled")
        self.model_button_red.place(x=796, y=583)

        self.continue_button = ImageTk.PhotoImage \
            (file='images\\continue_button_red.png')
        self.click_continue_button = Button(self.window, image=self.continue_button,
                                        font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                        , borderwidth=0, background="white", cursor="hand2", command=self.continue_pressed)
        self.click_continue_button.place(x=1000, y=300)

    def continue_pressed(self):
        self.model_button_red.configure(state="active")



    def run_model_frame(self):
        win = Toplevel()
        from New_Interface.Frontend import model_dashboard
        model_dashboard.ModelDashboard(win)
        self.window.withdraw()
        win.deiconify()


    def display_dataframe(self):
        file_name = self.chosen_file.get()
        file_url = FOLDER_URL + '\\' + file_name

        self.win = Toplevel()
        f = Frame(self.win)
        f.pack(fill=BOTH, expand=1)
        self.table = pt = Table(f,showtoolbar=True, showstatusbar=True)
        self.table.importCSV(file_url)
        pt.show()


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


        if str(self.model_button_red['state']) == 'active':
            self.model_button_red.configure(state="active")
        else:
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
    FeatureDashboard(window)
    window.mainloop()


if __name__ == '__main__':
    win()
