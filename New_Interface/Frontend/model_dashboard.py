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
from New_Interface.Frontend.user_dashboard import UserDashboard

class ModelDashboard(UserDashboard):
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
            return


def win():
    window = tk.ThemedTk()
    window.get_themes()
    window.set_theme("arc")
    ModelDashboard(window)
    window.mainloop()


if __name__ == '__main__':
    win()
