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
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import *
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC


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

    def apply_pca(self, data,n_elements_model):
        pca_data = PCA(n_components=n_elements_model)
        principal_components_data = pca_data.fit_transform(data)
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
        return X_train, X_test, y_train, y_test

    def run_svm(self, X_train, X_test, y_train):
        # todo I dont need these?
        normaliser = StandardScaler()
        normaliser.fit(X_train)
        X_train = normaliser.transform(X_train)
        X_test = normaliser.transform(X_test)

        training_type = SVC(kernel='rbf', gamma='auto', C=1.0,
                            decision_function_shape='ovr').fit(X_train,
                                                               y_train)
        return (X_test, training_type, X_train)

    def two_dim_graph_train(self,
                            X,
                            y,
                            training_type,
                            number_of_columns,
                            ):
        """
        Shows the graph of passed in data and labels.
        :param X: Data
        :param y: labels
        :param training_type: Training training_type
        :param number_of_columns: Integer value of number of columns
        """

        (set_for_X, set_for_Y) = (X, y)
        (ravel_1, ravel_2) = np.meshgrid(np.arange(start=set_for_X[:, 0].min() - 1,
                                                   stop=set_for_X[:, 0].max() + 1, step=0.01),
                                         np.arange(start=set_for_X[:, 1].min() - 1,
                                                   stop=set_for_X[:, 1].max() + 1, step=0.01))
        Xpred = np.array([ravel_1.ravel(), ravel_2.ravel()] + [np.repeat(0,
                                                                         ravel_1.ravel().size) for _ in
                                                               range(number_of_columns)]).T

        pred = training_type.decision_function(Xpred).reshape(ravel_1.shape)

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
        plt.show()

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

    def normalization(self, concatenate_data_labels, features_columns):
        """
        Makes all the values to be between 0 to 1 for rows.
        :param concatenate_data_labels: Joined data and labels
        :param features_columns: Columns of features
        :return: Normalized data, feature columns
        """

        data = concatenate_data_labels.loc[:, features_columns].values
        normalizer = StandardScaler()
        data = normalizer.fit_transform(data)
        feat_cols = ['feature' + str(i) for i in range(data.shape[1])]
        Feature_named_column = pd.DataFrame(data, columns=feat_cols)
        return (data, Feature_named_column)



    def pca_svm(self, features, label, n_elements_model):
        concatenate_data_labels, features_columns = self.concatenate(features,
                                                                     label)
        # PCA
        x_data, feature_named_column = \
            self.normalization(concatenate_data_labels, features_columns)

        principal_components_data = self.apply_pca(x_data,n_elements_model)
        features = self.covert_to_dataframe(principal_components_data)
        # todo Hassan do you think we need scale
        # features = self.scale(features)

        X_train, X_test, y_train, y_test = self.smote(features, label)

        X_test, training_type, X_train = self.run_svm(X_train, X_test, y_train)

        scores = cross_val_predict(training_type, X_test, y_test, cv=10)

        accuracy_text = 'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(),
                                                         scores.std() * 2)
        accuracy_text = 'PCA with SVM: ' + accuracy_text
        print(accuracy_text)

        # CM
        y_pred = training_type.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        print('This is SVM confusion matrix: ')
        print(cm)

        shape = X_train.shape
        number_of_col = shape[1] - 2

        # Print
        self.two_dim_graph_train(X_train, y_train, training_type, number_of_col)

    def pca_regression(self, features, label,n_elements_model):
        concatenate_data_labels, features_columns = self.concatenate(features,
                                                                     label)
        # PCA
        x_data, feature_named_column = \
            self.normalization(concatenate_data_labels, features_columns)

        principal_components_data = self.apply_pca(x_data,n_elements_model)
        features = self.covert_to_dataframe(principal_components_data)
        # todo Hassan do you think we need scale
        # features = self.scale(features)

        X_train, X_test, y_train, y_test = self.smote(features, label)

        training_type = RandomForestClassifier(random_state=42)
        training_type.fit(X_train, y_train)
        preds = cross_val_predict(training_type, X_test, y_test, cv=10)

        print("Random forest classifier: ")
        print(f'Accuracy = {accuracy_score(y_test, preds):.2f}'
              f'\nRecall = {recall_score(y_test, preds):.2f}\n')
        cm = confusion_matrix(y_test, preds)
        print(cm)

        plt.figure(figsize=(5, 7))
        ax = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
        sns.distplot(preds, hist=False, color="b", label="Fitted Values", ax=ax)
        plt.title('Actual vs Fitted Values')
        plt.tight_layout()
        plt.show()
        plt.close()

        Tree = training_type.estimators_[5]
        plt.figure(figsize=(25, 15))
        tree.plot_tree(Tree, filled=True,
                       rounded=True,
                       fontsize=14);
        plt.show()
        plt.close()

    def svm(self, features, label):
        # SMOTE
        X_train, X_test, y_train, y_test = self.smote(features, label)
        X_test, training_type, X_train = self.run_svm(X_train, X_test, y_train)

        scores = cross_val_predict(training_type, X_test, y_test, cv=10)

        accuracy_text = 'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(),
                                                         scores.std() * 2)
        accuracy_text = 'SVM: ' + accuracy_text
        print(accuracy_text)

        # CM
        y_pred = training_type.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        print('This is SVM confusion matrix: ')
        print(cm)

        plt.figure(figsize=(5, 7))
        ax = sns.distplot(y_test, hist=False, color='r',
                          label='Actual Value')
        sns.distplot(y_pred, hist=False, color='b', label='Fitted Values',
                     ax=ax)
        plt.title('Actual vs Fitted Values')
        plt.show()
        plt.close()

        return training_type, X_train

    def regression(self, features, labels):
        x = self.scale(features)
        X_train, X_test, y_train, y_test = self.smote(x, labels)

        training_type = RandomForestClassifier(random_state=42)
        training_type.fit(X_train, y_train)
        preds = cross_val_predict(training_type, X_test, y_test, cv=10)

        print("Random forest classifier: ")
        print(f'Accuracy = {accuracy_score(y_test, preds):.2f}'
              f'\nRecall = {recall_score(y_test, preds):.2f}\n')
        cm = confusion_matrix(y_test, preds)
        print(cm)

        plt.figure(figsize=(5, 7))
        ax = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
        sns.distplot(preds, hist=False, color="b", label="Fitted Values", ax=ax)
        plt.title('Actual vs Fitted Values')
        plt.tight_layout()
        plt.show()
        plt.close()

        Tree = training_type.estimators_[5]
        plt.figure(figsize=(25, 15))
        tree.plot_tree(Tree, filled=True,
                       rounded=True,
                       fontsize=14);
        plt.show()
        plt.close()

        return training_type, X_train

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

    def MLPRegression(self, features, labels):
        max_iter = 1000
        hidden_layer_sizes = 15
        features = self.scale(features)
        X_train, X_test, y_train, y_test = self.smote(features, labels)

        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, max_iter=max_iter,
                            hidden_layer_sizes=hidden_layer_sizes, random_state=1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.score(X_test, y_pred))
