from tkinter import *
from ttkthemes import themed_tk as tk

from New_Interface.Frontend.user_dashboard import UserDashboard


class StartApp:
    def __init__(self, window):
        # todo Add login feature
        self.window = window
        win = Toplevel()
        UserDashboard(win)
        self.window.withdraw()
        win.deiconify()


def win():
    window = tk.ThemedTk()
    window.get_themes()
    window.set_theme("arc")
    StartApp(window)
    window.mainloop()


if __name__ == '__main__':
    win()
