import os
import pickle
import tkinter

from tkinter import *
import numpy as np

from ttkthemes import themed_tk as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import ImageTk
import pandas as pd

from New_Interface.Frontend.user_dashboard import UserDashboard

#todo Solve: File location
#todo saving name have a condition on the length
SAVED_FILE_URL = r'C:\Users\marci\OneDrive\Other\Desktop\Shared\Tool_Interface\New_Interface\Frontend\joined_files'


class ConcatenateFiles(UserDashboard):
    def __init__(self, window, selection):
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
        self.set_frame(selection)


    def set_frame(self,selection):
        add_frame = Frame(self.window)
        add_frame.place(x=35, y=159)

        self.concatenate_frame = ImageTk.PhotoImage \
            (file='images\\concatenate_frame.png')
        self.add_panel = Label(add_frame, image=self.concatenate_frame, bg="white")
        self.add_panel.pack(fill='both', expand='yes')

        self.lb_selection = Listbox(self.window, width=50, height=6)
        self.lb_selection.place(x=127, y=285)
        self.listbox_object = selection
        for item in self.listbox_object:
            self.file_name = os.path.basename(item)
            self.lb_selection.insert(END, self.file_name)

        self.common_values = self.extract_common_features()
        self.lb_common = Listbox(self.window, width=64, height=6)
        self.lb_common.place(x=554, y=285)
        for self.item in self.common_values:
            self.lb_common.insert(END, self.item)

        self.concatenate_image = ImageTk.PhotoImage \
            (file='images\\concatenate_file_button_red.png')
        self.concatenate_files_button_red = Button(self.window, image=self.concatenate_image,
                                                   font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                                   activebackground="white"
                                                   , borderwidth=0, background="white", cursor="hand2",
                                                   command=self.click_concatenate_files)
        self.concatenate_files_button_red.place(x=1050, y=400)


        self.feature = ImageTk.PhotoImage \
            (file='images\\feature_button_red.png')
        self.feature_button_red = Button(self.window, image=self.feature,
                                               font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                               activebackground="white"
                                               , borderwidth=0, background="white", cursor="hand2")
        self.feature_button_red.configure(state="disabled")
        self.feature_button_red.place(x=150, y=24)

        self.extract = ImageTk.PhotoImage \
            (file='images\\extract_button_blue.png')
        self.extract_button_red = Button(self.window, image=self.extract,
                                        font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                        activebackground="white"
                                        , borderwidth=0, background="white", cursor="hand2")
        self.extract_button_red.configure(state="disabled")
        self.extract_button_red.place(x=278, y=24)

        # self.model = ImageTk.PhotoImage \
        #     (file='images\\model_button_red.png')
        # self.model_button_red = Button(self.window, image=self.model,
        #                                 font=("yu gothic ui", 13, "bold"), relief=FLAT,
        #                                 activebackground="white"
        #                                 , borderwidth=0, background="white", cursor="hand2")
        # self.model_button_red.configure(state="disabled")
        # self.model_button_red.place(x=278, y=24)

        self.add = ImageTk.PhotoImage \
            (file='images\\add_button_blue.png')
        self.add_button = Button(self.window, image=self.add,
                                 font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                 , borderwidth=0, background="white", cursor="hand2", command=self.click_add)
        self.add_button.place(x=22, y=24)

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
                # messagebox.showinfo("Merged Data", "Data was merged on: \n {}".format(one_element))

        user_input = simpledialog.askstring(title="File Name",prompt="Enter name for the file.:", parent=self.window)
        if type(user_input) == str and user_input != '':
            file_url = SAVED_FILE_URL + '\\' + user_input + '.csv'
            df = pd.DataFrame(merged_dataset, columns=merged_dataset.columns)
            df.to_csv(file_url)
            messagebox.showinfo("File name", "File saved as: \n {}".format(user_input))
        else:
            messagebox.showinfo("File name", "The file name is empty or need to be a word: \n {}".format(user_input))

            # todo Need to do something with the data, (i.e. save)

    def get_file_name(self):
        files = self.read_selected_files()
        read_file = []
        for file in files:
            read_file.append(file)

        return read_file

    def extract_common_features(self):
        files = self.read_selected_files()
        read_file_and_columns = []
        for file in files:
            file = file.columns
            read_file_and_columns.append(file)

        return self.common_field(read_file_and_columns)

    def click_add(self):
        win = Toplevel()
        from New_Interface.Frontend import user_dashboard
        user_dashboard.UserDashboard(win).set_feature_button()
        self.window.withdraw()
        win.deiconify()

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
