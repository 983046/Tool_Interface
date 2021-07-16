import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
from matplotlib.colors import ListedColormap
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, precision_score
from sklearn.model_selection import *
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.svm import SVC
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import imgkit



imputer = SimpleImputer(strategy='median')
from sklearn.ensemble import GradientBoostingRegressor
import lime
from lime import lime_tabular
import xgboost as xgb


FILE_URL = 'C:\\Users\\marci\\OneDrive\\Other\\Desktop\\Shared\\Tool_Interface\\New_Interface\\Frontend\\HTML_file'
HTML_IMAGE_URL = 'C:\\Users\\marci\\OneDrive\\Other\\Desktop\\Shared\\Tool_Interface\\New_Interface\\Frontend\\HTML_image'
HTML_pdf = 'C:\\Users\\marci\\OneDrive\\Other\\Desktop\\Shared\\Tool_Interface\\New_Interface\\Frontend\\HTML_pdf'

class RunModel:
    def feature_label(self, read_file, label):
        features = read_file.drop(label, axis=1)
        label = read_file[label]
        return features, label

    def feature_deeper_label(self, read_file, label, deeper_label):
        data_labels = read_file
        feature_use = read_file

        # Changes all the value to 0 except if have health condition, which is changed to 1.
        data = read_file[[label]]. \
            applymap(lambda x: (0 if x != int(deeper_label) else 1))
        data_labels[label] = data[label]

        features = feature_use.drop(label, axis=1)
        labels = data_labels[label]
        print(features)
        print(labels)
        return features, labels

    def apply_pca(self, data, n_elements_model):
        pca_data = PCA(n_components=n_elements_model)
        principal_components_data = pca_data.fit_transform(data)
        eighenValues = pca_data.explained_variance_ratio_
        appPlot = tk.Tk()

        plot = plt.figure(figsize=(5, 7))
        plt.plot(eighenValues[:20])

        canvas = FigureCanvasTkAgg(plot, appPlot)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        return principal_components_data

    def covert_to_dataframe(self, principal_components_data):
        # todo Hassan code here in future. Ask question
        principal_data_Df = pd.DataFrame(data=principal_components_data,
                                         columns=['principal component 1',
                                                  'principal component 2'])
        return principal_data_Df

    def smote(self, x, y):
        sm = SMOTE(random_state=42)
        X_sm, y_sm = sm.fit_resample(x, y)

        print(f'''Shape of X before SMOTE: {x.shape} 
        Shape of X after SMOTE: {X_sm.shape}''')

        print('Balance of positive and negative classes (%):')
        print(y_sm.value_counts(normalize=True) * 100)

        X_train, X_test, y_train, y_test = train_test_split(
            X_sm, y_sm, random_state=42)

        # todo split them.
        return X_train, y_train

    def clean_testing_split(self, x, y):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
        return X_test, y_test

    def two_dim_graph_train(self, X, y, training_type, number_of_columns):
        """
        Shows the graph of passed in data and labels.
        :param X: Data
        :param y: labels
        :param training_type: Training training_type
        :param number_of_columns: Integer value of number of columns
        """

        X = X.to_numpy()
        y = y.to_numpy()

        (set_for_X, set_for_Y) = (X, y)
        (ravel_1, ravel_2) = np.meshgrid(np.arange(start=set_for_X[:, 0].min() - 1,
                                                   stop=set_for_X[:, 0].max() + 1, step=0.01),
                                         np.arange(start=set_for_X[:, 1].min() - 1,
                                                   stop=set_for_X[:, 1].max() + 1, step=0.01))
        Xpred = np.array([ravel_1.ravel(), ravel_2.ravel()] + [np.repeat(0,
                                                                         ravel_1.ravel().size) for _ in
                                                               range(number_of_columns)]).T

        pred = training_type.decision_function(Xpred).reshape(ravel_1.shape)

        plot = plt.figure(figsize=(15, 10))
        plt.contourf(
            ravel_1,
            ravel_2,
            pred,
            alpha=1.0,
            cmap='RdYlGn',
            levels=np.linspace(pred.min(), pred.max(), 100),
        )
        plt.xlim(ravel_1.min(), ravel_1.max())
        plt.ylim(ravel_2.min(), ravel_2.max())
        for (i, j) in enumerate(np.unique(set_for_Y)):
            plt.scatter(set_for_X[set_for_Y == j, 0], set_for_X[set_for_Y == j, 1],
                        c=ListedColormap(('red', 'green'))(i), label=j)
        plt.title('Graph')
        plt.xlabel('principal component 1')
        plt.ylabel('principal component 1')
        plt.tight_layout()
        plt.legend()
        # plt.show()

        appPlot = tk.Tk()

        canvas = FigureCanvasTkAgg(plot, appPlot)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def scale(self, data):
        """
        Reshape data of columns.
        :param data: Data
        :return: Reshaped data
        """

        # Scale only columns that have values greater than 1

        to_scale = [col for col in data.columns if data[col].max() > 1]
        mms = MinMaxScaler()
        scaled = mms.fit_transform(data[to_scale])
        scaled = pd.DataFrame(scaled, columns=to_scale)

        # Replace original columns with scaled ones

        for col in scaled:
            data[col] = scaled[col]

        return data

    def concatenate(self, features, labels):
        """
        Joins features and labels together.
        :param features: Feature data
        :param labels: Label data
        :return: Combined features and labels, feature columns
        """

        labels_rows = len(labels)
        labels = np.reshape(labels.to_numpy(), (labels_rows, 1))

        data = np.concatenate([features, labels], axis=1)
        data = pd.DataFrame(data)

        features_columns = features.columns
        features_labels = np.append(features_columns, 'extraction_dashboard_label')

        data.columns = features_labels

        return (data, features_columns)

    def normilzation_type(self, chosen_normalise):
        if chosen_normalise == 'Normalizer':
            return Normalizer()
        elif chosen_normalise == 'MinMaxScaler':
            return MinMaxScaler()
        elif chosen_normalise == 'StandardScaler':
            return StandardScaler()
        else:
            pass

    def normalization(self, concatenate_data_labels, features_columns, chosen_normalise):
        """
        Makes all the values to be between 0 to 1 for rows.
        :param concatenate_data_labels: Joined data and labels
        :param features_columns: Columns of features
        :return: Normalized data, feature columns
        """
        data = concatenate_data_labels.loc[:, features_columns].values
        normalizer = self.normilzation_type(chosen_normalise)
        data = normalizer.fit_transform(data)
        feat_cols = ['feature' + str(i) for i in range(data.shape[1])]
        Feature_named_column = pd.DataFrame(data, columns=feat_cols)
        return (data, Feature_named_column)

    def pca_svm(self, features, label, n_elements_model, chosen_normalise):
        concatenate_data_labels, features_columns = self.concatenate(features,
                                                                     label)
        # PCA
        x_data, feature_named_column = \
            self.normalization(concatenate_data_labels, features_columns, chosen_normalise)

        principal_components_data = self.apply_pca(x_data, n_elements_model)
        features = pd.DataFrame(principal_components_data)

        # features = self.covert_to_dataframe(principal_components_data)

        X_train, y_train = self.smote(features, label)
        X_test, y_test = self.clean_testing_split(features, label)

        training_type = SVC(kernel='rbf', gamma='auto', C=1.0,
                            decision_function_shape='ovr').fit(X_train,
                                                               y_train)
        preds = cross_val_predict(training_type, X_test, y_test, cv=10)

        print("Random forest classifier: ")
        try:
            print(f'Accuracy = {accuracy_score(y_test, preds):.2f}'
                  f'\nRecall = {recall_score(y_test, preds):.2f}\n')
        except:
            print('Precision = :', precision_score(y_test, preds, average='micro'))
            print("Recall Score : ", recall_score(y_test, preds, average='micro'))
        cm = confusion_matrix(y_test, preds)
        print(cm)

        shape = X_train.shape
        number_of_col = shape[1] - 2

        # Print
        self.two_dim_graph_train(X_train, y_train, training_type, number_of_col)

        appPlot = tk.Tk()

        plot = plt.figure(figsize=(5, 7))
        ax = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
        sns.distplot(preds, hist=False, color="b", label="Fitted Values", ax=ax)
        plt.title('Actual vs Fitted Values')
        plt.tight_layout()
        # test.show()

        canvas = FigureCanvasTkAgg(plot, appPlot)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        return training_type, X_train, X_test

    def pca_regression(self, features, label, n_elements_model, chosen_normalise):

        if chosen_normalise != "Nothing":
            concatenate_data_labels, features_columns = self.concatenate(features,
                                                                         label)
            # PCA
            features, feature_named_column = \
                self.normalization(concatenate_data_labels, features_columns, chosen_normalise)

        principal_components_data = self.apply_pca(features, n_elements_model)
        features = pd.DataFrame(principal_components_data)

        X_train, y_train = self.smote(features, label)
        X_test, y_test = self.clean_testing_split(features, label)

        training_type = RandomForestClassifier(random_state=42)
        training_type.fit(X_train, y_train)
        preds = cross_val_predict(training_type, X_test, y_test, cv=10)

        print("Random forest classifier: ")
        print("pca_MLPRegression")
        try:
            print(f'Accuracy = {accuracy_score(y_test, preds):.2f}'
                  f'\nRecall = {recall_score(y_test, preds):.2f}\n')
        except:
            print('Precision = :', precision_score(y_test, preds, average='micro'))
            print("Recall Score : ", recall_score(y_test, preds, average='micro'))
        cm = confusion_matrix(y_test, preds)
        print(cm)

        shape = X_train.shape
        number_of_col = shape[1] - 2

        # Print
        self.two_dim_graph_train(X_train, y_train, training_type, number_of_col)

        appPlot = tk.Tk()
        treePlot = tk.Tk()

        plot = plt.figure(figsize=(5, 7))
        ax = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
        sns.distplot(preds, hist=False, color="b", label="Fitted Values", ax=ax)
        plt.title('Actual vs Fitted Values')
        plt.tight_layout()
        # test.show()

        canvas = FigureCanvasTkAgg(plot, appPlot)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        Tree = training_type.estimators_[5]
        treeplot = plt.figure(figsize=(15, 10))
        tree.plot_tree(Tree, filled=True,
                       rounded=True,
                       fontsize=14);

        canvas = FigureCanvasTkAgg(treeplot, treePlot)
        canvas.draw()
        canvas.get_tk_widget().pack()

        return training_type, X_train, X_test

    def svm(self, features, label, chosen_normalise):
        if chosen_normalise != "Nothing":
            concatenate_data_labels, features_columns = self.concatenate(features,
                                                                         label)
            features, feature_named_column = \
                self.normalization(concatenate_data_labels, features_columns, chosen_normalise)

        # SMOTE
        X_train, y_train = self.smote(features, label)
        X_test, y_test = self.clean_testing_split(features, label)

        training_type = SVC(kernel='rbf', gamma='auto', C=1.0,
                            decision_function_shape='ovr').fit(X_train,
                                                               y_train)

        preds = cross_val_predict(training_type, X_test, y_test, cv=10)

        print("SVM: ")
        try:
            print(f'Accuracy = {accuracy_score(y_test, preds):.2f}'
                  f'\nRecall = {recall_score(y_test, preds):.2f}\n')
        except:
            print('Precision = :', precision_score(y_test, preds, average='micro'))
            print("Recall Score : ", recall_score(y_test, preds, average='micro'))
        cm = confusion_matrix(y_test, preds)
        print(cm)

        appPlot = tk.Tk()

        plot = plt.figure(figsize=(5, 7))
        ax = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
        sns.distplot(preds, hist=False, color="b", label="Fitted Values", ax=ax)
        plt.title('Actual vs Fitted Values')
        plt.tight_layout()
        # test.show()

        canvas = FigureCanvasTkAgg(plot, appPlot)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        return training_type, X_train, X_test

    def regression(self, features, labels, chosen_normalise):
        # x = self.scale(features)
        if chosen_normalise != "Nothing":
            concatenate_data_labels, features_columns = self.concatenate(features,
                                                                         labels)
            features, feature_named_column = \
                self.normalization(concatenate_data_labels, features_columns, chosen_normalise)

        X_train, y_train = self.smote(features, labels)
        X_test, y_test = self.clean_testing_split(features, labels)

        training_type = RandomForestClassifier(random_state=42)
        training_type.fit(X_train, y_train)
        preds = cross_val_predict(training_type, X_test, y_test, cv=10)

        print("Random forest classifier: ")
        try:
            print(f'Accuracy = {accuracy_score(y_test, preds):.2f}'
                  f'\nRecall = {recall_score(y_test, preds):.2f}\n')
        except:
            print('Precision = :', precision_score(y_test, preds, average='micro'))
            print("Recall Score : ", recall_score(y_test, preds, average='micro'))
        cm = confusion_matrix(y_test, preds)
        print(cm)

        appPlot = tk.Tk()
        treePlot = tk.Tk()

        plot = plt.figure(figsize=(5, 7))
        ax = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
        sns.distplot(preds, hist=False, color="b", label="Fitted Values", ax=ax)
        plt.title('Actual vs Fitted Values')
        plt.tight_layout()
        #test.show()

        canvas = FigureCanvasTkAgg(plot, appPlot)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        Tree = training_type.estimators_[5]
        treeplot = plt.figure(figsize=(15, 10))
        tree.plot_tree(Tree, filled=True,
                         rounded=True,
                         fontsize=14);


        canvas = FigureCanvasTkAgg(treeplot, treePlot)
        canvas.draw()
        canvas.get_tk_widget().pack()


        return training_type, X_train, X_test

    def MLPRegression(self, features, labels, chosen_normalise,max_iter,hidden_layer_sizes):
        if chosen_normalise != "Nothing":
            concatenate_data_labels, features_columns = self.concatenate(features,
                                                                         labels)
            features, feature_named_column = \
                self.normalization(concatenate_data_labels, features_columns, chosen_normalise)

        # features = self.scale(features)

        X_train, y_train = self.smote(features, labels)
        X_test, y_test = self.clean_testing_split(features, labels)

        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, max_iter=max_iter,
                            hidden_layer_sizes=hidden_layer_sizes, random_state=1)
        clf.fit(X_train, y_train)
        preds = cross_val_predict(clf, X_test, y_test, cv=10)

        print("MLPRegression")
        try:
            print(f'Accuracy = {accuracy_score(y_test, preds):.2f}'
                  f'\nRecall = {recall_score(y_test, preds):.2f}\n')
        except:
            print('Precision = :', precision_score(y_test, preds, average='micro'))
            print("Recall Score : ", recall_score(y_test, preds, average='micro'))
        cm = confusion_matrix(y_test, preds)
        print(cm)

        appPlot = tk.Tk()

        plot = plt.figure(figsize=(5, 7))
        ax = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
        sns.distplot(preds, hist=False, color="b", label="Fitted Values", ax=ax)
        plt.title('Actual vs Fitted Values')
        plt.tight_layout()
        # test.show()

        canvas = FigureCanvasTkAgg(plot, appPlot)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        return clf, X_train, X_test

    def pca_MLPRegression(self, features, labels, n_elements_model, chosen_normalise,max_iter,hidden_layer_sizes):
        # max_iter = 1000
        # hidden_layer_sizes = 15
        if chosen_normalise != "Nothing":
            concatenate_data_labels, features_columns = self.concatenate(features,
                                                                         labels)
            # PCA
            features, feature_named_column = \
                self.normalization(concatenate_data_labels, features_columns, chosen_normalise)

        principal_components_data = self.apply_pca(features, n_elements_model)
        # features = self.covert_to_dataframe(principal_components_data)
        features = pd.DataFrame(principal_components_data)
        # features = self.scale(features)

        X_train, y_train = self.smote(features, labels)
        X_test, y_test = self.clean_testing_split(features, labels)

        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, max_iter=max_iter,
                            hidden_layer_sizes=hidden_layer_sizes, random_state=1)
        clf.fit(X_train, y_train)

        preds = cross_val_predict(clf, X_test, y_test, cv=10)

        print("pca_MLPRegression")
        try:
            print(f'Accuracy = {accuracy_score(y_test, preds):.2f}'
                  f'\nRecall = {recall_score(y_test, preds):.2f}\n')
        except:
            print('Precision = :', precision_score(y_test, preds, average='micro'))
            print("Recall Score : ", recall_score(y_test, preds, average='micro'))

        cm = confusion_matrix(y_test, preds)
        print(cm)

        shape = X_train.shape
        number_of_col = shape[1] - 2

        # Print
        self.two_dim_graph_train(X_train, y_train, clf, number_of_col)

        appPlot = tk.Tk()

        plot = plt.figure(figsize=(5, 7))
        ax = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
        sns.distplot(preds, hist=False, color="b", label="Fitted Values", ax=ax)
        plt.title('Actual vs Fitted Values')
        plt.tight_layout()
        # test.show()

        canvas = FigureCanvasTkAgg(plot, appPlot)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        return clf, X_train, X_test

    def gradient_boosting_regression(self, features, labels):
        # X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2,
        #                                                     random_state=42)
        X_train, y_train = self.smote(features, labels)
        X_test, y_test = self.clean_testing_split(features, labels)
        self.perform_feature_selection(X_train, X_test, y_train, y_test)

    def perform_feature_selection(self, X_train, X_test, y_train, y_test):
        """
        Method to perform feature selection using gradient booster.
        :param X_train: Training data
        :param X_test: Testing data
        :param y_train: Training extraction_dashboard_label
        :param y_test: Testing extraction_dashboard_label
        """

        imputer.fit(X_train)
        # Transform both training data and testing data
        transformed_X = imputer.transform(X_train)
        transformed_X_test = imputer.transform(X_test)
        transformed_y = np.array(y_train).reshape((-1,))
        transformed_y_test = np.array(y_test).reshape((-1,))

        model = GradientBoostingRegressor(loss='lad',
                                          max_depth=5,
                                          max_features=None,
                                          min_samples_leaf=2,
                                          min_samples_split=6,
                                          n_estimators=350,
                                          random_state=42)

        model.fit(transformed_X, transformed_y)

        X = pd.DataFrame(X_train)
        # Extract the feature importance into a dataframe
        important_feature = pd.DataFrame({'feature': list(X.columns),
                                          'importance': model.feature_importances_})

        # Show the top 10 most important
        important_feature = important_feature.sort_values('importance', ascending=False) \
            .reset_index(drop=True)

        important_feature.loc[:9, :].plot(x='feature', y='importance',
                                          edgecolor='k',
                                          kind='barh', color='blue')
        plt.xlabel('Relative Importance')
        plt.ylabel('')
        plt.title('Feature Importances from Random Forest')
        plt.tight_layout()
        plt.show()
        plt.close()

        # Extract the names of the most important features
        most_important_features = important_feature['feature'][:10]
        index_of_features = [list(X.columns).index(x) for x in most_important_features]
        kept_important_features_X = transformed_X[:, index_of_features]
        kept_important_features_X_test = transformed_X_test[:, index_of_features]

        # Create the model with the same hyperparamters
        model_reduced = GradientBoostingRegressor(loss='lad',
                                                  max_depth=5,
                                                  max_features=None,
                                                  min_samples_leaf=2,
                                                  min_samples_split=6,
                                                  n_estimators=350,
                                                  random_state=42)

        # Fit and test on the reduced set of features
        model_reduced.fit(kept_important_features_X, transformed_y)
        model_reduced_pred = model_reduced.predict(kept_important_features_X_test)

        # Find the residuals
        residuals = abs(model_reduced_pred - transformed_y_test)

        # Exact the worst and best prediction
        wrong = kept_important_features_X_test[np.argmax(residuals), :]

        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=kept_important_features_X,
                                                           mode='regression',
                                                           training_labels=transformed_y,
                                                           feature_names=list(most_important_features))

        explanation_for_wrong = explainer.explain_instance(data_row=wrong,
                                                           predict_fn=model_reduced.predict)

        # Plot the prediction explaination
        explanation_for_wrong.as_pyplot_figure()
        plt.title('Explanation of Prediction')
        plt.xlabel('Effect on Prediction')
        plt.tight_layout()
        plt.show()
        plt.close()

    def XGBoost(self, features, labels, chosen_normalise):
        if chosen_normalise != "Nothing":
            concatenate_data_labels, features_columns = self.concatenate(features,
                                                                         labels)
            features, feature_named_column = \
                self.normalization(concatenate_data_labels, features_columns, chosen_normalise)

        X_train, y_train = self.smote(features, labels)
        X_test, y_test = self.clean_testing_split(features, labels)

        xgb_model = xgb.XGBClassifier(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                                      max_depth=5, alpha=10, n_estimators=10)
        xgb_model.fit(X_train, y_train)

        preds = cross_val_predict(xgb_model, X_test, y_test, cv=10)

        print("XGBoost")

        try:
            print(f'Accuracy = {accuracy_score(y_test, preds):.2f}'
                  f'\nRecall = {recall_score(y_test, preds):.2f}\n')
        except:
            print('Precision = :', precision_score(y_test, preds, average='micro'))
            print("Recall Score : ", recall_score(y_test, preds, average='micro'))

        cm = confusion_matrix(y_test, preds)
        print(cm)

        appPlot = tk.Tk()

        plot = plt.figure(figsize=(5, 7))
        ax = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
        sns.distplot(preds, hist=False, color="b", label="Fitted Values", ax=ax)
        plt.title('Actual vs Fitted Values')
        plt.tight_layout()
        # test.show()

        canvas = FigureCanvasTkAgg(plot, appPlot)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        return xgb_model, X_train, X_test

    def pca_XGBoost(self, features, labels, n_elements_model, chosen_normalise):
        if chosen_normalise != "Nothing":
            concatenate_data_labels, features_columns = self.concatenate(features,
                                                                         labels)
            # PCA
            features, feature_named_column = \
                self.normalization(concatenate_data_labels, features_columns, chosen_normalise)

        principal_components_data = self.apply_pca(features, n_elements_model)
        features = pd.DataFrame(principal_components_data)

        X_train, y_train = self.smote(features, labels)
        X_test, y_test = self.clean_testing_split(features, labels)

        xgb_model = xgb.XGBClassifier(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                                      max_depth=5, alpha=10, n_estimators=10)
        xgb_model.fit(X_train, y_train)

        preds = cross_val_predict(xgb_model, X_test, y_test, cv=10)

        print("pca_XGBoost")
        try:
            print(f'Accuracy = {accuracy_score(y_test, preds):.2f}'
                  f'\nRecall = {recall_score(y_test, preds):.2f}\n')
        except:
            print('Precision = :', precision_score(y_test, preds, average='micro'))
            print("Recall Score : ", recall_score(y_test, preds, average='micro'))
        cm = confusion_matrix(y_test, preds)
        print(cm)

        shape = X_train.shape
        number_of_col = shape[1] - 2

        # Print
        self.two_dim_graph_train(X_train, y_train, xgb_model, number_of_col)

        appPlot = tk.Tk()

        plot = plt.figure(figsize=(5, 7))
        ax = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
        sns.distplot(preds, hist=False, color="b", label="Fitted Values", ax=ax)
        plt.title('Actual vs Fitted Values')
        plt.tight_layout()
        # test.show()

        canvas = FigureCanvasTkAgg(plot, appPlot)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        return xgb_model, X_train, X_test

    # todo PLot is here

    def importance_plot(self, training_type, X_train):
        expShap = shap.TreeExplainer(training_type)
        shap_values = expShap.shap_values(X_train)
        shap.summary_plot(shap_values[0], X_train, plot_type='dot')
        shap.summary_plot(shap_values, X_train, plot_type='bar')
        # self.shap_plot(0, training_type, X_train)

    def shap_dot_plot(self, training_type, X_train):
        expShap = shap.TreeExplainer(training_type)
        shap_values = expShap.shap_values(X_train)
        shap.summary_plot(shap_values[0], X_train, plot_type='dot')

    def shap_bar_plot(self, training_type, X_train):
        expShap = shap.TreeExplainer(training_type)
        shap_values = expShap.shap_values(X_train)
        shap.summary_plot(shap_values, X_train, plot_type='bar')

    def shap_plot(self, j, training_type, X_train):
        explainerModel = shap.TreeExplainer(training_type)
        shap_values_Model = explainerModel.shap_values(X_train)
        # shap.force_plot(explainerModel.expected_value[j], shap_values_Model[j])

    def shap_dependence_plot(self, training_type, X_train):
        expShap = shap.TreeExplainer(training_type)
        shap_values = expShap.shap_values(X_train)
        shap.dependence_plot("SEQN", shap_values[0], X_train)

    def lime_plot(self, training_type, X_train, X_test, features, file_name):
        # columns=X_test.columns.values
        X_test = pd.DataFrame(X_test)
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=features.columns,
            class_names=['bad', 'good'],
            mode='classification'
        )

        exp = explainer.explain_instance(
            data_row=X_test.iloc[1],
            predict_fn=training_type.predict_proba
        )

        html_save = FILE_URL + '\\' + file_name + '.html'
        exp.save_to_file(html_save)

        # pdf = HTML_pdf +'\\' + file_name + '.pdf'
        image_save = HTML_IMAGE_URL +'\\' + file_name + '.jpg'

        imgkit.from_file(html_save, image_save)





