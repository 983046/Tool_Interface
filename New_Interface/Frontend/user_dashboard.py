import os

import time
from tkinter import *
from tkinter.filedialog import askopenfilename

from ttkthemes import themed_tk as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import ImageTk
import pandas as pd

#todo Validation for buttons and saving single file

file_path = []
SAVED_FILE_URL = r'C:\Users\marci\OneDrive\Other\Desktop\Shared\Tool_Interface\New_Interface\Frontend\joined_files'

class UserDashboard:
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

    def click_next_file(self):
        self.feature_button_red.configure(state="active")
        if len(self.get_selection()) != 0:
            boolean_selection = messagebox.askquestion("Save Selection", "Would you like to save the selection")
            if boolean_selection == 'yes':
                user_input = simpledialog.askstring(title="File Name", prompt="Enter name for the file.:")
                file_url = SAVED_FILE_URL + '\\' + user_input + '.csv'

                self.listbox_object = self.get_selection()
                file = self.read_selected_files()
                df = pd.DataFrame(file[0], columns=file[0].columns)


                df.to_csv(file_url)
                messagebox.showinfo("File name", "File saved as: \n {}".format(user_input))
            else:
                None

    def run_feature_frame(self):
        win = Toplevel()
        from New_Interface.Frontend import feature_dashboard
        feature_dashboard.FeatureDashboard(win)
        self.window.withdraw()
        win.deiconify()


    def set_frame(self):
        add_frame = Frame(self.window)
        add_frame.place(x=35, y=159)

        self.add_dashboard_frame = ImageTk.PhotoImage \
            (file='images\\add_frame.png')
        self.add_panel = Label(add_frame, image=self.add_dashboard_frame, bg="white")
        self.add_panel.pack(fill='both', expand='yes')

        self.add = ImageTk.PhotoImage \
            (file='images\\add_button_blue.png')
        self.add_button = Button(self.window, image=self.add,
                                 font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                 , borderwidth=0, background="white", cursor="hand2", command=self.set_frame)
        self.add_button.place(x=22, y=24)

        self.extract = ImageTk.PhotoImage \
            (file='images\\extract_button_red.png')
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

        self.add_file = ImageTk.PhotoImage \
            (file='images\\add_file_button_red.png')
        self.add_file_button_red = Button(self.window, image=self.add_file,
                                          font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                          , borderwidth=0, background="white", cursor="hand2",
                                          command=self.click_add_file)
        self.add_file_button_red.place(x=110, y=320)

        self.lb = Listbox(self.window, width=50, height=22, selectmode=MULTIPLE)
        self.lb.place(x=641, y=199)

        self.remove_file = ImageTk.PhotoImage \
            (file='images\\remove_file_button_red.png')
        self.remove_file_button_red = Button(self.window, image=self.remove_file,
                                             font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                             , borderwidth=0, background="white", cursor="hand2",
                                             command=self.click_remove_file)
        self.remove_file_button_red.place(x=110, y=470)

        self.combine_file = ImageTk.PhotoImage \
            (file='images\\combine_file_button_red.png')
        self.combine_file_button_red = Button(self.window, image=self.combine_file,
                                                font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                                activebackground="white"
                                                , borderwidth=0, background="white", cursor="hand2",command=self.click_combine_file)
        self.combine_file_button_red.configure(state="disabled")
        self.combine_file_button_red.place(x=1015, y=300)


        self.concatenate_file_user = ImageTk.PhotoImage \
            (file='images\\concatenate_file_button_red.png')
        self.concatenate_file_user_button_red = Button(self.window, image=self.concatenate_file_user,
                                                       font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                                       activebackground="white"
                                                       , borderwidth=0, background="white", cursor="hand2",command=self.click_concatenate_file_user)
        self.concatenate_file_user_button_red.configure(state="disabled")
        self.concatenate_file_user_button_red.place(x=1015, y=480)

        self.next_file = ImageTk.PhotoImage \
            (file='images\\next_button_red.png')
        self.next_file_button_red = Button(self.window, image=self.next_file,
                                           font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                           activebackground="white"
                                           , borderwidth=0, background="white", cursor="hand2", command=self.click_next_file)
        # self.next_file_button_red.configure(state="disabled")
        self.next_file_button_red.place(x=1100, y=600)

        self.feature = ImageTk.PhotoImage \
            (file='images\\feature_button_red.png')
        self.feature_button_red = Button(self.window, image=self.feature,
                                                  font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                                  activebackground="white"
                                                  , borderwidth=0, background="white", cursor="hand2",
                                                  command=self.run_feature_frame)
        self.feature_button_red.configure(state="disabled")
        self.feature_button_red.place(x=150, y=24)

        self.selected_shape = ImageTk.PhotoImage \
            (file='images\\selected_shape.png')
        self.selected_shape_red = Button(self.window, image=self.selected_shape,
                                             font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                             , borderwidth=0, background="white", cursor="hand2")
        self.selected_shape_red.place(x=55, y=120)

        self.model = ImageTk.PhotoImage \
            (file='images\\model_button_red.png')
        self.model_button_red = Button(self.window, image=self.model,
                                        font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                        activebackground="white"
                                        , borderwidth=0, background="white", cursor="hand2")
        self.model_button_red.configure(state="disabled")
        self.model_button_red.place(x=410, y=24)

    def click_concatenate_file_user(self):
        if len(self.get_selection()) >= 2:
            if self.check_file_type() == True:
                win = Toplevel()
                from New_Interface.Frontend import concatenate_files
                #todo Buttton not works.
                concatenate_files.ConcatenateFiles(win, self.get_selection())
                self.window.withdraw()
                win.deiconify()
            else:
                messagebox.showerror("Program not optimized", "The program is not optimized for this filename type: ")
        else:
            messagebox.showinfo("File selection", "Two or more files need to be selected")

    def check_file_type(self):
        selected = self.lb.curselection()
        for index in selected:
            one_element = file_path[index]
            if one_element.endswith('.XPT'):
                return True
            elif one_element.endswith('.csv'):
                return True
            elif one_element.endswith('.xlsx'):
                return True
            else:
                return False

    def click_combine_file(self):
        if len(self.get_selection()) >= 2:
            if self.check_file_type() == True:
                from New_Interface.Frontend import combine_files
                combine_files.CombineFiles(self.get_selection())
            else:
                messagebox.showerror("Program not optimized", "The program is not optimized for this filename type: ")
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
            elif file.endswith('.csv'):
                file = pd.read_csv(file)
                files.append(file)
            elif file.endswith('.xlsx'):
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
            self.next_file_button_red.configure(state="active")
        if (self.lb.size() >= 2):
            self.concatenate_file_user_button_red.configure(state="active")
            self.combine_file_button_red.configure(state="active")
            self.next_file_button_red.configure(state="active")

    def set_feature_button(self):
        self.feature_button_red.configure(state="active")
        #self.next_file_button_red.configure(state="active")


    def read_folder(self, url):
        values = []
        for (root, dirs, files) in os.walk(url):
            for file in files:
                fileName = os.path.basename(file)
                values.append(fileName)

        return values

    def read_single_file(self, file):
        if file.endswith('.XPT'):
            file = pd.read_sas(file)
            return file
        elif file.endswith('.csv'):
            file = pd.read_csv(file)
            return file
        elif file.endswith('.xlsx'):
            file = pd.read_excel(file, index_col=0)
            return file



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
