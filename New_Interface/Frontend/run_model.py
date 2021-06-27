from sklearn.decomposition import PCA
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import *
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class RunModel:
    def feature_label(self, read_file, label):
        features = read_file.drop(label, axis=1)
        label = read_file[label]
        return features, label
    def feature_deeper_label(self, read_file, label, deeper_label):
        features = read_file.drop(label, axis=1)
        data_labels = read_file

        data = read_file[[label]]. \
            applymap(lambda x: (0 if x != int(deeper_label) else 1))
        data_labels[label] = data[label]

        labels = data_labels[label]

        return features, labels

    def apply_pca(self, data):
        pca_data = PCA(n_components=2)
        principal_components_data = pca_data.fit_transform(data)
        return principal_components_data

    def covert_to_dataframe(self,principal_components_data):
        #todo Hassan code here in future. Ask question
        principal_data_Df = pd.DataFrame(data=principal_components_data,
                                         columns=['principal component 1',
                                                  'principal component 2'])
        return principal_data_Df

    def smote(self, x, y):
        """
        Used for class imbalances.
        :param x: Data
        :param y: Labels
        :return: Balanced data
        """
        sm = SMOTE(random_state=42)
        X_sm, y_sm = sm.fit_resample(x, y)

        print(f'''Shape of X before SMOTE: {x.shape} 
        Shape of X after SMOTE: {X_sm.shape}''')

        print('Balance of positive and negative classes (%):')
        print(y_sm.value_counts(normalize=True) * 100)

        X_train, X_test, y_train, y_test = train_test_split(
            X_sm, y_sm, random_state=42)

        return X_train, X_test, y_train, y_test

    def run_svm(self, X_train, X_test, y_train):
        #todo I dont need these?
        normaliser = StandardScaler()
        normaliser.fit(X_train)
        X_train = normaliser.transform(X_train)
        X_test = normaliser.transform(X_test)

        training_type = SVC(kernel='rbf', gamma='auto', C=1.0,
                            decision_function_shape='ovr').fit(X_train,
                                                               y_train)
        return (X_test, training_type, X_train)

    def two_dim_graph_train(
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



    def pca_svm(self, features, label):
        principal_components_data = self.apply_pca(features)
        features = self.covert_to_dataframe(principal_components_data)
        #todo Hassan do you think we need scale
        #features = self.scale(features)

        X_train, X_test, y_train, y_test = self.smote(features, label)



        X_test, training_type, X_train = self.run_svm(X_train,X_test,y_train)

        scores = cross_val_score(training_type, X_test, y_test, cv=10,
                                 scoring='accuracy')

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



    def pca_regression(self):
        return
    def svm(self):
        return
    def regression(self):
        return

