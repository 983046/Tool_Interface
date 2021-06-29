from tkinter import messagebox, simpledialog, Button, FLAT
import numpy as np
from PIL import ImageTk
from tqdm import tk
import pandas as pd

from New_Interface.Frontend.concatenate_files import ConcatenateFiles

#todo Make combine_files with concatenate_files, make it automatic.

#todo Solve: File location
SAVED_FILE_URL = r'C:\Users\marci\OneDrive\Other\Desktop\Shared\Tool_Interface\New_Interface\Frontend\joined_files'

class CombineFiles(ConcatenateFiles):
    def __init__(self,dashboard_selection):
        self.listbox_object = dashboard_selection
        self.files = self.read_selected_files()

        for i, dataset in enumerate(self.files):
            if i == 0:
                merged_dataset = dataset
            else:
                merged_dataset = np.concatenate([merged_dataset, dataset], axis=1)

        user_input = simpledialog.askstring(title="File Name", prompt="Enter name for the file.:", parent=self.window)
        if type(user_input) == str and user_input != '':
            file_url = SAVED_FILE_URL + '\\' + user_input + '.csv'
            df = pd.DataFrame(merged_dataset, columns=merged_dataset.columns)
            df.to_csv(file_url)
            self.combine_concatenate_pressed = True
            messagebox.showinfo("File name", "File saved as: \n {}".format(user_input))
        else:
            messagebox.showinfo("File name", "The file name is empty or need to be a word: \n {}".format(user_input))




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

