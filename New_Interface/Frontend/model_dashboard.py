import os

import time
from tkinter import *
from tkinter.filedialog import askopenfilename

from ttkthemes import themed_tk as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import ImageTk
import pandas as pd
from pandastable import Table, TableModel
import numpy as np
import pickle


from New_Interface.Frontend.run_model import RunModel
from New_Interface.Frontend.user_dashboard import UserDashboard
from New_Interface.Frontend.feature_dashboard import FeatureDashboard

FOLDER_URL = r'C:\Users\marci\OneDrive\Other\Desktop\Shared\Tool_Interface\New_Interface\Frontend\joined_files'
SAVED_MODEL_URL = r'C:\Users\marci\OneDrive\Other\Desktop\Shared\Tool_Interface\New_Interface\Frontend\saved_model'

class ModelDashboard(RunModel, FeatureDashboard):
    def __init__(self, window, label, deeper_label,file_name, ticked_deeper,chosen_normalise):
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
        self.label = label
        self.deeper_label = deeper_label
        self.ticked_deeper = ticked_deeper
        self.file_name = file_name
        self.chosen_normalise = chosen_normalise
        self.set_frame()

    def set_frame(self):
        self.add_frame = Frame(self.window)
        self.add_frame.place(x=35, y=159)

        self.model_dashboard_frame = ImageTk.PhotoImage \
            (file='images\\model_pca_frame.png')
        self.model_panel = Label(self.add_frame, image=self.model_dashboard_frame, bg="white")
        self.model_panel.pack(fill='both', expand='yes')

        self.feature = ImageTk.PhotoImage \
            (file='images\\feature_button_red.png')
        self.feature_button_blue = Button(self.window, image=self.feature,
                                          font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                          activebackground="white"
                                          , borderwidth=0, background="white", cursor="hand2",
                                          command=self.run_feature_frame)
        self.feature_button_blue.place(x=150, y=24)


        self.model = ImageTk.PhotoImage \
            (file='images\\model_button_blue.png')
        self.model_button_red = Button(self.window, image=self.model,
                                        font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                        activebackground="white"
                                        , borderwidth=0, background="white", cursor="hand2", command=self.run_model_frame)
        self.model_button_red.place(x=410, y=24)

        self.add = ImageTk.PhotoImage \
            (file='images\\add_button_red.png')
        self.add_button = Button(self.window, image=self.add,
                                 font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                 , borderwidth=0, background="white", cursor="hand2", command=self.click_add)
        self.add_button.place(x=22, y=24)

        self.extract = ImageTk.PhotoImage \
            (file='images\\extract_button_red.png')
        self.extract_button_red = Button(self.window, image=self.extract,
                                         font=("yu gothic ui", 13, "bold"), relief=FLAT,
                                         activebackground="white"
                                         , borderwidth=0, background="white", cursor="hand2",
                                         command=self.run_extraction_frame)
        self.extract_button_red.place(x=278, y=24)

        self.model_value = ['SVM', 'Regression', 'MLPRegressor','XGBoost','Nothing']
        self.chosen_model_value = StringVar(self.window)
        self.combo_model_value = OptionMenu(self.window, self.chosen_model_value, *self.model_value)
        self.combo_model_value.configure(width=11)
        self.combo_model_value.place(x=123, y=280)

        # self.applied_pca = ['Yes', 'No']
        # self.chosen_applied_pca= StringVar(self.window)
        # combo_applied_pca = OptionMenu(self.window, self.chosen_applied_pca,
        #                         *self.applied_pca)
        # combo_applied_pca.place(x=350, y=380)

        self.cb = IntVar()
        checkbox = Checkbutton(self.window, text="<--- Ticking the box will apply PCA", variable=self.cb, onvalue=1,
                               offvalue=0, command=self.isChecked)
        checkbox.place(x=292, y=283)

        self.apply = ImageTk.PhotoImage \
            (file='images\\apply_button_red.png')
        self.apply_button = Button(self.window, image=self.apply,
                                   font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                   , borderwidth=0, background="white", cursor="hand2", command=self.chosen_model)
        self.apply_button.place(x=1100, y=260)

        self.explanation = ImageTk.PhotoImage \
            (file='images\\explanation_button_red.png')
        self.explanation_button = Button(self.window, image=self.explanation,
                                         font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                         , borderwidth=0, background="white", cursor="hand2",
                                         command=self.get_importance)
        self.explanation_button.configure(state='disabled')
        self.explanation_button.place(x=1100, y=443)

        self.specific_value = StringVar()
        self.n_elements_model = Entry(self.window, textvariable=self.specific_value)
        self.n_elements_model.configure(state="disabled", width='27')
        self.n_elements_model.place(x=705, y=293)

        self.selected_shape = ImageTk.PhotoImage \
            (file='images\\selected_shape.png')
        self.selected_shape_red = Button(self.window, image=self.selected_shape,
                                         font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                         , borderwidth=0, background="white", cursor="hand2")
        self.selected_shape_red.place(x=445, y=120)

        self.save_model = ImageTk.PhotoImage \
            (file='images\\save_model_button_red.png')
        self.save_model_button = Button(self.window, image=self.save_model,
                                         font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                         , borderwidth=0, background="white", cursor="hand2", command=self.save_model_pressed)
        self.save_model_button.configure(state='disabled')
        self.save_model_button.place(x=1100, y=574)

        self.gbr = ImageTk.PhotoImage \
            (file='images\\gbr_button_red.png')
        self.gbr_button = Button(self.window, image=self.gbr,
                                         font=("yu gothic ui", 13, "bold"), relief=FLAT, activebackground="white"
                                         , borderwidth=0, background="white", cursor="hand2",
                                         command=self.run_gbr)
        self.gbr_button.configure(state='disabled')
        self.gbr_button.place(x=800, y=443)

    def save_model_pressed(self):
        user_input = simpledialog.askstring(title="File Name", prompt="Enter name for the file.:")
        document_name = SAVED_MODEL_URL + '\\' + user_input + '.sav'
        pickle.dump(self.training_type, open(document_name, 'wb'))

    def click_add(self):
        win = Toplevel()
        from New_Interface.Frontend import user_dashboard
        user_dashboard.UserDashboard(win).set_feature_button()
        self.window.withdraw()
        win.deiconify()

    def run_extraction_frame(self):
        win = Toplevel()
        from New_Interface.Frontend import extraction_dashboard
        extraction_dashboard.ExtractionDashboard(win,self.chosen_normalise,self.file_name)
        self.window.withdraw()
        win.deiconify()

    def run_model_frame(self):
        win = Toplevel()
        from New_Interface.Frontend import model_dashboard
        model_dashboard.ModelDashboard(win,self.label, self.deeper_label,self.file_name, self.ticked_deeper,self.chosen_normalise)
        self.window.withdraw()
        win.deiconify()

    def isChecked(self):
        if self.cb.get() == 1:
            self.n_elements_model.configure(state="normal")
        elif self.cb.get() == 0:
            self.n_elements_model.configure(state="disabled")
        else:
            messagebox.showerror('Something unexpected', 'Something went wrong!')

    def chosen_model(self):
        self.model = self.chosen_model_value.get()
        file_url = FOLDER_URL + '\\' + self.file_name
        read_file = self.read_single_file(file_url)
        self.model_ran = False

        if self.ticked_deeper == 1:
            self.features, self.chosen_label = self.feature_deeper_label(read_file, self.label, self.deeper_label)
        else:
            self.features, self.chosen_label = self.feature_label(read_file, self.label)

        if self.cb.get() == 1:
            try:
                integer_result = int(self.specific_value.get())
                if self.model == 'SVM':
                    self.training_type, X_train, self.X_test = self.pca_svm(self.features, self.chosen_label, integer_result,self.chosen_normalise)
                    self.explanation_value = ['Nothing']
                elif self.model == 'Regression':
                    self.training_type, X_train, self.X_test = self.pca_regression(self.features, self.chosen_label, integer_result,self.chosen_normalise)
                    self.explanation_value = ['Shap dependence Plot','Shap Dot Plot', 'Shap Bar Plot', 'Nothing']
                elif self.model == 'MLPRegressor':
                    self.max_iter = int(simpledialog.askstring(title="max_iter", prompt="Enter the amount of max iterations:",
                                               parent=self.window))
                    self.hidden_layer_sizes = int(simpledialog.askstring(title="hidden_layer_sizes",
                                                                         prompt="Enter the amount of "
                                                                                "hidden layers:",
                                                                         parent=self.window))

                    self.training_type, self.X_train, self.X_test = self.pca_MLPRegression(self.features, self.chosen_label,integer_result,
                                                                          self.chosen_normalise,self.max_iter,self.hidden_layer_sizes )
                    self.explanation_value = ['lime plot','Shap dependence Plot', 'Shap Dot Plot', 'Shap Bar Plot', 'Nothing']
                elif self.model == 'XGBoost':
                    self.explanation_value = ['lime plot', 'Shap dependence Plot','Shap Dot Plot', 'Shap Bar Plot', 'Nothing']
                    self.training_type, self.X_train, self.X_test = self.pca_XGBoost(self.features, self.chosen_label,integer_result,
                                                                                 self.chosen_normalise)

            except ValueError:
                messagebox.showerror('Components', 'Number of components need to be a number!')
        else:
            if self.model == 'SVM':
                self.explanation_value = ['Nothing']
                self.training_type, self.X_train, self.X_test = self.svm(self.features, self.chosen_label,self.chosen_normalise)
            elif self.model == 'Regression': #todo lime,
                self.explanation_value = ['Shap dependence Plot','Shap Dot Plot', 'Shap Bar Plot', 'Nothing']
                self.training_type, self.X_train, self.X_test = self.regression(self.features, self.chosen_label,self.chosen_normalise)
            elif self.model == 'MLPRegressor': #todo just lime
                self.max_iter = int(simpledialog.askstring(title="max_iter", prompt="Enter the amount of max iterations:",
                                                       parent=self.window))
                self.hidden_layer_sizes = int(simpledialog.askstring(title="hidden_layer_sizes",
                                                                 prompt="Enter the amount of "
                                                                        "hidden layers:",
                                                                 parent=self.window))
                self.explanation_value = ['lime plot', 'Nothing']
                self.training_type, self.X_train, self.X_test = self.MLPRegression(self.features, self.chosen_label,self.chosen_normalise,self.max_iter,self.hidden_layer_sizes)
            elif self.model == 'XGBoost':
                self.explanation_value = ['lime plot', 'Shap Bar Plot', 'Nothing']
                self.training_type, self.X_train, self.X_test = self.XGBoost(self.features, self.chosen_label,self.chosen_normalise)

        self.save_model_button.configure(state='normal')
        self.gbr_button.configure(state='normal')
        self.explanation_button.configure(state="normal")
        self.chosen_explanation_value = StringVar(self.window)
        self.combo_explanation_value = OptionMenu(self.window, self.chosen_explanation_value, *self.explanation_value)
        self.combo_explanation_value.configure(width=35)
        self.combo_explanation_value.place(x=124, y=467)

    def get_importance(self):
        explanation_type = self.chosen_explanation_value.get()
        if explanation_type == 'Nothing':
            pass
        elif explanation_type == 'Shap Dot Plot':
            self.shap_dot_plot(self.training_type, self.X_train)
        elif explanation_type == 'Shap Bar Plot':
            self.shap_bar_plot(self.training_type, self.X_train)
        elif explanation_type == 'Shap dependence Plot':
            self.shap_dependence_plot(self.training_type, self.X_train)
        elif explanation_type == 'lime plot':
            self.name = simpledialog.askstring(title="File Name", prompt="Enter Name:",
                                                       parent=self.window)
            self.lime_plot(self.training_type, self.X_train, self.X_test,self.features, self.name)

    def run_gbr(self):
        self.gradient_boosting_regression(self.features, self.chosen_label)


def win():
    window = tk.ThemedTk()
    window.get_themes()
    window.set_theme("arc")
    ModelDashboard(window)
    window.mainloop()


if __name__ == '__main__':
    win()
