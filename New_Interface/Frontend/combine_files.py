from tkinter import messagebox
import numpy as np
from tqdm import tk

from New_Interface.Frontend.concatenate_files import ConcatenateFiles

#todo Make combine_files with concatenate_files, make it automatic.
class CombineFiles(ConcatenateFiles):
    def __init__(self,dashboard_selection):
        self.listbox_object = dashboard_selection
        self.files = self.read_selected_files()
        for i, dataset in enumerate(self.files):
            if i == 0:
                merged_dataset = dataset
            else:
                merged_dataset = np.concatenate([merged_dataset, dataset], axis=1)
        messagebox.showinfo("Combined  Data", "Data was combined:")

    # todo Need to do something with the data, (i.e. save)


def win():
    window = tk.ThemedTk()
    window.get_themes()
    window.set_theme("arc")
    CombineFiles(window)
    window.mainloop()


if __name__ == '__main__':
    win()

