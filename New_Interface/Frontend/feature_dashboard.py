import os

import time
from tkinter import *
from tkinter.filedialog import askopenfilename

from ttkthemes import themed_tk as tk
from tkinter import ttk, messagebox
from PIL import ImageTk
import pandas as pd
from New_Interface.Frontend.user_dashboard import UserDashboard

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

        # =======================================================================
        # ========================Starting Tree View=============================
        # =======================================================================
        self.tree_view_frame = Frame(self.window, bg="white")
        self.tree_view_frame.place(x=388, y=180, height=250, width=600)

        style = ttk.Style()
        style.configure("Treeview.Heading", font=('yu gothic ui', 10, "bold"), foreground="red")
        style.configure("Treeview", font=('yu gothic ui', 9, "bold"), foreground="#f29844")

        scroll_x = Scrollbar(self.tree_view_frame, orient=HORIZONTAL)
        scroll_y = Scrollbar(self.tree_view_frame, orient=VERTICAL)
        self.student_tree = ttk.Treeview(self.tree_view_frame,
                                         columns=(
                                             "STUDENT ID", "FNAME", "LNAME", "EMAIL", "DOB", "GENDER", "ADDRESS",
                                             "CONTACT NO", "SHIFT", "COURSE ENROLLED", "BATCH", "SECTION",
                                             "REGISTRATION DATE"),
                                         xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set)
        scroll_x.pack(side=BOTTOM, fill=X)
        scroll_y.pack(side=RIGHT, fill=Y)
        scroll_x.config(command=self.student_tree.xview)
        scroll_y.config(command=self.student_tree.yview)

        # ==========================TreeView Heading====================
        self.student_tree.heading("STUDENT ID", text="STUDENT ID")
        self.student_tree.heading("FNAME", text="FNAME")
        self.student_tree.heading("LNAME", text="LNAME")
        self.student_tree.heading("EMAIL", text="EMAIL")
        self.student_tree.heading("DOB", text="DOB")
        self.student_tree.heading("GENDER", text="GENDER")
        self.student_tree.heading("ADDRESS", text="ADDRESS")
        self.student_tree.heading("CONTACT NO", text="CONTACT NO")
        self.student_tree.heading("SHIFT", text="SHIFT")
        self.student_tree.heading("COURSE ENROLLED", text="COURSE ENROLLED")
        self.student_tree.heading("BATCH", text="BATCH")
        self.student_tree.heading("SECTION", text="SECTION")
        self.student_tree.heading("REGISTRATION DATE", text="REGISTRATION DATE")
        self.student_tree["show"] = "headings"

        # ==========================TreeView Column====================
        self.student_tree.column("STUDENT ID", width=150)
        self.student_tree.column("FNAME", width=170)
        self.student_tree.column("LNAME", width=170)
        self.student_tree.column("EMAIL", width=200)
        self.student_tree.column("ADDRESS", width=150)
        self.student_tree.column("GENDER", width=150)
        self.student_tree.column("CONTACT NO", width=150)
        self.student_tree.column("DOB", width=150)
        self.student_tree.column("SHIFT", width=150)
        self.student_tree.column("COURSE ENROLLED", width=150)
        self.student_tree.column("BATCH", width=100)
        self.student_tree.column("SECTION", width=100)
        self.student_tree.column("REGISTRATION DATE", width=150)
        self.student_tree.pack(fill=BOTH, expand=1)



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
                                              command=self.click_next_file)
        self.next_feature_button_blue.place(x=477, y=583)

        self.feature_dashboard_frame = ImageTk.PhotoImage \
            (file='images\\feature_frame.png')
        self.feature_panel = Label(add_frame, image=self.feature_dashboard_frame, bg="white")
        self.feature_panel.pack(fill='both', expand='yes')




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


def win():
    window = tk.ThemedTk()
    window.get_themes()
    window.set_theme("arc")
    FeatureDashboard(window)
    window.mainloop()


if __name__ == '__main__':
    win()
