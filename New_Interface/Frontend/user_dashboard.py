import os

import time
from tkinter import *
from tkinter.filedialog import askopenfilename

from ttkthemes import themed_tk as tk
from tkinter import ttk, messagebox
from PIL import ImageTk
import pandas as pd

#todo Validation for buttons and saving single file

file_path = []
class UserDashboard:
    def __init__(self, window):
        self.window = window
        self.window.geometry("1366x720+0+0")
        self.window.title("Dashboard")
        self.window.resizable(False, False)
        self.admin_dashboard_frame = ImageTk.PhotoImage \
            (file='images\\user_frame.png')
        self.image_panel = Label(self.window, image=self.admin_dashboard_frame)
        self.image_panel.pack(fill='both', expand='yes')

        # ============================Welcome Dashboard==============================
        self.txt = "Welcome to Dashboard"
        self.heading = Label(self.window, text=self.txt, font=('yu gothic ui', 20, "bold"), bg="white",
                             fg='black',
                             relief=FLAT)
        self.heading.place(x=570, y=43)

        # ============================Date and time==============================
        self.date_time_image = Label(self.window, bg="white")
        self.date_time = Label(self.window)
        self.date_time.place(x=80, y=45)
        self.time_running()

        # ============================Exit button===============================
        self.exit = ImageTk.PhotoImage \
            (file='images\\exit_button.png')
        self.exit_button = Button(self.window, image=self.exit,
                                  font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                  , borderwidth=0, background="white", cursor="hand2", command=self.click_exit)
        self.exit_button.place(x=1260, y=55)
        # ============================Add button===============================
        self.add = ImageTk.PhotoImage \
            (file='images\\add_button_red.png')
        self.add_button = Button(self.window, image=self.add,
                                 font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                 , borderwidth=0, background="white", cursor="hand2", command=self.click_add)
        self.add_button.place(x=622, y=542)
        self.click_add()

    def click_next_file(self):
        self.next_feature = ImageTk.PhotoImage \
            (file='images\\feature_button_red.png')
        self.next_feature_button_red = Button(self.window, image=self.next_feature,
                                                  font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                                  activebackground="white"
                                                  , borderwidth=0, background="white", cursor="hand2",
                                                  command=self.run_feature_frame())
        self.next_feature_button_red.place(x=475, y=582)

    def run_feature_frame(self):
        win = Toplevel()
        from New_Interface.Frontend import feature_dashboard
        feature_dashboard.FeatureDashboard(win)
        self.window.withdraw()
        win.deiconify()


    def click_add(self):
        add_frame = Frame(self.window)
        add_frame.place(x=48, y=116)

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
            (file='images\\combine_file_button_grey.png')
        self.combine_file_button_red = Button(self.window, image=self.combine_file,
                                              font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                              activebackground="white"
                                              , borderwidth=0, background="white", cursor="hand2")
        self.combine_file_button_red.place(x=1000, y=225)

        self.concatenate_file_user = ImageTk.PhotoImage \
            (file='images\\concatenate_file_button_grey.png')
        self.concatenate_file_user_button_red = Button(self.window, image=self.concatenate_file_user,
                                                       font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                                       activebackground="white"
                                                       , borderwidth=0, background="white", cursor="hand2")
        self.concatenate_file_user_button_red.place(x=1000, y=325)

        self.next_file = ImageTk.PhotoImage \
            (file='images\\next_button_grey.png')
        self.next_file_button_red = Button(self.window, image=self.next_file,
                                           font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                           activebackground="white"
                                           , borderwidth=0, background="white", cursor="hand2")
        self.next_file_button_red.place(x=1000, y=425)




    def click_concatenate_file_user(self):
        if len(self.get_selection()) >= 2:
            win = Toplevel()
            from New_Interface.Frontend import concatenate_files
            #todo Buttton not works.
            concatenate_files.ConcatenateFiles(win, self.get_selection())
            self.window.withdraw()
            win.deiconify()
        else:
            messagebox.showinfo("File selection", "Two or more files need to be selected")


    def click_combine_file(self):
        if len(self.get_selection()) >= 2:
            from New_Interface.Frontend import combine_files
            combine_files.CombineFiles(self.get_selection())
        else:
            messagebox.showinfo("File selection", "Two or more files need to be selected")

    def get_selection(self):
        value = []
        selected = self.lb.curselection()
        for index in selected[::-1]:
            one_element = file_path[index]
            value.append(one_element)
        return value

    def click_remove_file(self):
        try:
            selected = self.lb.curselection()
            for index in selected[::-1]:
                file_path.pop(index)
                file_name = self.lb.get(index)
                self.lb.delete(index)
                # print(file_name)
                # file_location = ADDED_FILES + '\\' + file_name
                # print(file_location)
                # if os.path.exists(file_location):
                #     os.remove(file_location)
                #
                #     self.window.deiconify()
                messagebox.showinfo("Removed File", "The removed file was: \n {}".format(file_name))

        except IndexError:
            pass

    def read_selected_files(self):
        files = []
        for file in self.listbox_object:
            if file.endswith('.XPT'):
                file = pd.read_sas(file)
                files.append(file)
            elif file.endswith('.CSV'):
                file = pd.read_sas(file)
                files.append(file)
            elif file.endswith('.XLSX'):
                file = pd.read_excel(file, index_col=0)
                files.append(file)
            else:
                messagebox.showerror("Program not optimized", "The program is not optimized for this filename type: "
                                                              "\n {}".format(file))
        return files


    def click_add_file(self):
        data = [('All files', '*.*')]
        file = askopenfilename(filetypes=data, defaultextension=data,
                               title='Please select a file:')
        if len(file) != 0:
            file_path.append(file)
            file_name = os.path.basename(file)
            # url = ADDED_FILES + '/' + file_name
            # exists = os.path.exists(url)
            # if exists:
            #     messagebox.showinfo("Selected File", "The file already exists: \n {}".format(file_name))
            # else:
            # self.window.deiconify()
            messagebox.showinfo("Selected File", "The added file is: \n {}".format(file))
            # shutil.copyfile(file, file_name)
            # shutil.move(file_name, ADDED_FILES)
            self.lb.insert(END, file_name)
        # else:
        #     self.window.deiconify()
        #     messagebox.showinfo("Selected File", "No File was selected")

        self.red_buttons()


    def red_buttons(self):
        if (self.lb.size() >= 1):
            self.next_file = ImageTk.PhotoImage \
                (file='images\\next_button_red.png')
            self.next_file_button_red = Button(self.window, image=self.next_file,
                                               font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                               activebackground="white"
                                               , borderwidth=0, background="white", cursor="hand2",
                                               command=self.click_next_file)
            self.next_file_button_red.place(x=1000, y=425)
        if (self.lb.size() >= 2):
            self.combine_file = ImageTk.PhotoImage \
                (file='images\\combine_file_button_red.png')
            self.combine_file_button_red = Button(self.window, image=self.combine_file,
                                                  font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                                  activebackground="white"
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

    def read_folder(self, url):
        values = []
        for (root, dirs, files) in os.walk(url):
            for file in files:
                fileName = os.path.basename(file)
                values.append(fileName)

        return values

    def click_exit(self):
        self.window.deiconify()
        ask = messagebox.askyesnocancel("Confirm Exit", "Are you sure you want to Exit\n User Dashboard?")
        if ask is True:
            self.window.quit()

    def time_running(self):
        """ displays the current date and time which is shown at top left corner of admin dashboard"""
        self.time = time.strftime("%H:%M:%S")
        self.date = time.strftime('%Y/%m/%d')
        concated_text = f"  {self.time} \n {self.date}"
        self.date_time.configure(text=concated_text, font=("yu gothic ui", 13, "bold"), relief=FLAT
                                 , borderwidth=0, background="white", foreground="black")
        self.date_time.after(100, self.time_running)


def win():
    window = tk.ThemedTk()
    window.get_themes()
    window.set_theme("arc")
    UserDashboard(window)
    window.mainloop()


if __name__ == '__main__':
    win()
