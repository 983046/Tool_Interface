from tkinter import messagebox

from New_Interface.Frontend.user_dashboard import UserDashboard
import numpy as np


class UserDashboard(UserDashboard):
    def __init__(self, dashboard_selection):
        self.listbox_object = dashboard_selection
        self.files = self.read_selected_files()
        for i, dataset in enumerate(self.files):
            if i == 0:
                merged_dataset = dataset
            else:
                merged_dataset = np.concatenate([merged_dataset, dataset], axis=1)
        messagebox.showinfo("Combined  Data", "Data was combined:")

    # todo Need to do something with the data, (i.e. save)
