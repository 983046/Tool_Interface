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



    # def apply_norm(self, dataset,file_url):
    #     #todo Add this into run_model.py instead.
    #     method = self.chosen_normalise.get()
    #
    #     names = dataset.columns
    #     if method == 'Normalizer':
    #         scaler = preprocessing.Normalizer()
    #     elif method == 'MinMaxScaler':
    #         scaler = preprocessing.MinMaxScaler()
    #     elif method == 'StandardScaler':
    #         scaler = preprocessing.StandardScaler()
    #     else:
    #         return
    #
    #     d = scaler.fit(dataset)
    #     scaled_np = d.transform(dataset)
    #     scaled_df = pd.DataFrame(scaled_np, columns=names)
    #     scaled_df.to_csv(file_url)

    def apply_pressed(self):
        self.extract_button_red.configure(state="active")
        choice = self.chosen_na_value.get()
        file_name = self.chosen_file.get()
        file_url = FOLDER_URL + '\\' + file_name

        read_file = self.read_single_file(file_url)

        if choice == 'Mean':
            #todo working out the mean
            where_are_NaNs = np.isnan(read_file)
            read_file[where_are_NaNs] = read_file.mean(axis=1)
            read_file = read_file
            df = pd.DataFrame(read_file, columns=read_file.columns)
            df.to_csv(file_url)
        elif choice == 'Zero':
            where_are_NaNs = np.isnan(read_file)
            read_file[where_are_NaNs] = 0
            read_file = read_file
            df = pd.DataFrame(read_file, columns=read_file.columns)
            df.to_csv(file_url)
        elif choice == 'One':
            where_are_NaNs = np.isnan(read_file)
            read_file[where_are_NaNs] = 1
            read_file = read_file
            df = pd.DataFrame(read_file, columns=read_file.columns)
            df.to_csv(file_url)
        elif choice == 'Nothing':
            None

        #self.apply_norm(read_file,file_url)

        self.display_dataframe_button.configure(state="active")


        # self.model = ImageTk.PhotoImage \
        #     (file='images\\model_button_red.png')
        # self.model_button_red = Button(self.window, image=self.model,
        #                                font=("yu gothic ui", 13, "bold"), relief=FLAT,
        #                                activebackground="white"
        #                                , borderwidth=0, background="white", cursor="hand2", command=self.run_model_frame)
        # self.model_button_red.place(x=796, y=583)


    def set_frame(self):
        add_frame = Frame(self.window)
        add_frame.place(x=35, y=159)

        self.feature_dashboard_frame = ImageTk.PhotoImage \
            (file='images\\feature_frame.png')
        self.feature_panel = Label(add_frame, image=self.feature_dashboard_frame, bg="white")
        self.feature_panel.pack(fill='both', expand='yes')

        self.add = ImageTk.PhotoImage \
            (file='images\\add_button_red.png')
        self.add_button = Button(self.window, image=self.add,
                                 font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                 , borderwidth=0, background="white", cursor="hand2", command=self.click_add)
        self.add_button.place(x=22, y=24)

        self.feature = ImageTk.PhotoImage \
            (file='images\\feature_button_blue.png')
        self.feature_button_blue = Button(self.window, image=self.feature,
                                              font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                              activebackground="white"
                                              , borderwidth=0, background="white", cursor="hand2",
                                              command=self.run_feature_frame)
        self.feature_button_blue.place(x=150, y=24)

        # self.model = ImageTk.PhotoImage \
        #     (file='images\\model_button_red.png')
        # self.model_button_red = Button(self.window, image=self.model,
        #                                 font=("yu gothic ui", 13, "bold"), relief=FLAT,
        #                                 activebackground="white"
        #                                 , borderwidth=0, background="white", cursor="hand2",command=self.run_model_frame)
        # self.model_button_red.configure(state="disabled")
        # self.model_button_red.place(x=278, y=24)

        self.extract = ImageTk.PhotoImage \
            (file='images\\extract_button_red.png')
        self.extract_button_red = Button(self.window, image=self.extract,
                                        font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                        activebackground="white"
                                        , borderwidth=0, background="white", cursor="hand2", command=self.run_extract_frame)
        self.extract_button_red.configure(state="disabled")
        self.extract_button_red.place(x=278, y=24)


        self.files = self.read_folder(FOLDER_URL)
        if len(self.files) != 0:
            self.chosen_file = StringVar(self.window)
            comboLab = OptionMenu(self.window, self.chosen_file,
                                          *self.files)
            comboLab.configure(width=11)
            comboLab.place(x=123, y=280)

        self.na_values = ['Mean', 'Zero', 'One', 'Nothing']
        if len(self.na_values) != 0:
            self.chosen_na_value = StringVar(self.window)
            comboLab = OptionMenu(self.window, self.chosen_na_value,
                                          *self.na_values)
            comboLab.configure(width=21)
            comboLab.place(x=288, y=280)

        self.normalise = ['Normalizer', 'MinMaxScaler', 'StandardScaler', 'Nothing']
        self.chosen_normalise = StringVar(self.window)
        comboLab = OptionMenu(self.window, self.chosen_normalise,
                              *self.normalise)
        comboLab.configure(width=31)
        comboLab.place(x=513, y=280)


        self.apply = ImageTk.PhotoImage \
            (file='images\\apply_button_red.png')
        self.apply_button = Button(self.window, image=self.apply,
                                        font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                        , borderwidth=0, background="white", cursor="hand2", command=self.apply_pressed)
        self.apply_button.place(x=900, y=260)

        self.display_dataframe_image = ImageTk.PhotoImage \
            (file='images\\edit_table_button_red.png')
        self.display_dataframe_button = Button(self.window, image=self.display_dataframe_image,
                                        font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                        , borderwidth=0, background="white", cursor="hand2", command=self.display_dataframe)
        self.display_dataframe_button.configure(state="disabled")
        self.display_dataframe_button.place(x=1100, y=260)

        self.selected_shape = ImageTk.PhotoImage \
            (file='images\\selected_shape.png')
        self.selected_shape_red = Button(self.window, image=self.selected_shape,
                                             font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                             , borderwidth=0, background="white", cursor="hand2")
        self.selected_shape_red.place(x=188, y=120)

        self.model = ImageTk.PhotoImage \
            (file='images\\model_button_red.png')
        self.model_button_red = Button(self.window, image=self.model,
                                        font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                        activebackground="white"
                                        , borderwidth=0, background="white", cursor="hand2")
        self.model_button_red.configure(state="disabled")
        self.model_button_red.place(x=410, y=24)

    def run_extract_frame(self):
        win = Toplevel()
        from New_Interface.Frontend import extraction_dashboard
        extraction_dashboard.ExtractionDashboard(win, self.chosen_normalise.get(),self.chosen_file.get())
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
        win = Toplevel()
        from New_Interface.Frontend import user_dashboard
        user_dashboard.UserDashboard(win).set_feature_button()
        self.window.withdraw()
        win.deiconify()


def win():
    window = tk.ThemedTk()
    window.get_themes()
    window.set_theme("arc")
    FeatureDashboard(window)
    window.mainloop()


if __name__ == '__main__':
    win()



# def chnageDs(dataset,listType):
#     droped = 0
#     for i in range(len(dataset.columns)):
#             col = dataset.columns[i-droped]
#             print(i,col,listType[i],droped)
#             if listType[i] == 'cat':
#                     dataset[col] = pd.factorize(dataset[col])[0]
#             elif listType[i] == 'drop':
#                   #print('drop')
#                   dataset = dataset.drop([col], axis=1)
#                   #print(len(dataset.columns))
#                   droped = droped + 1
#             else:
#                   dataset[col] = dataset[col].astype(listType[i])
#     return dataset
# dataset[col] = pd.factorize(dataset[col])[0]


# def pca(data,n_components):
#     '''
#        tun pca on a dataset
#        @x : data
#        @n_components : how many commponent needed
#        @retun : eigen values and vectores
#     '''
#     print('pca',n_components)
#     pca_data = PCA(n_components=n_components)
#     principal_components_data = pca_data.fit_transform(data)
#      todo use this:
#     eighenValues = pca_data.explained_variance_ratio_
#     plt.plot(eighenValues[:20])
#     plt.show()
#     return (pca_data, principal_components_data)

# n_components = 'mle'
# #n_components = 4
# pcaCom = pca(x,n_components)

#todo train/test balance.
#u, c = np.unique(y_testOrg, return_counts=True)

#todo
# MLE
# Testing and training split : outputs 50% 50%
# Buttons for forest.
# PCA - Number of components.


