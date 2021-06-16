"""
A tool for predication of illnesses.
Author: Marcin kapcia 983046
Version number: 1
Creation Date: 01/2021
"""

import os
import pickle
import shutil
import tkinter
import tkinter as tk
import urllib.request
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showinfo
from urllib.error import HTTPError, URLError

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bs4 import BeautifulSoup
from imblearn.over_sampling import SMOTE
from matplotlib.colors import ListedColormap
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

imputer = SimpleImputer(strategy='median')
from sklearn.ensemble import GradientBoostingRegressor
import lime
import lime.lime_tabular

FILE_URL = r'unknown_file'
LABORATORY_URL = r'laboratory'
DEMOGRAPHIC_URL = r'demographic'
MEDICAL_URL = r'medical'
SAV_URL = r'sav_files'
PKL_URL = r'pkl_files'

NO_RETURN = "No_Return"
RETURN = "RETURN"


def combine_data(
        laboratory_input,
        demographic_input,
        demographic_bool,
        medical_input,
        health_condition,
        code_value,
):
    """
    Joins data together on SEQN, considering if demographic data is included.
    Edits the joined data for the specific code value.
    :param laboratory_input: Laboratory data
    :param demographic_input: Demographic data
    :param demographic_bool: Include demographic data or not
    :param medical_input: Medical Data
    :param health_condition: Health condition data
    :param code_value: Inputted number for health condition
    :return: Data as a whole, data features, data labels
    """

    laboratory = LABORATORY_URL + '\\' + str(laboratory_input)
    demographic = DEMOGRAPHIC_URL + '\\' + str(demographic_input)
    medical = MEDICAL_URL + '\\' + str(medical_input)

    # Checks if demographic file is included or not.
    if demographic_bool != '0':
        if demographic.endswith('.XPT'):
            demographic_data = pd.read_sas(demographic)
        else:
            demographic_data = pd.read_csv(demographic)
        if medical.endswith('.XPT'):
            medical_conditions_data = pd.read_sas(medical)
        else:
            medical_conditions_data = pd.read_csv(medical)

        # Makes dataframe only with SEQN and chosen health condition
        medical_conditions_data = medical_conditions_data[['SEQN',
                                                           health_condition]]
        print(laboratory)
        if laboratory.endswith('.XPT'):
            data = pd.read_sas(laboratory)
        else:
            data = pd.read_csv(laboratory)

        data = data.dropna()
        medical_conditions_data = \
            medical_conditions_data.dropna(subset=[health_condition])

        merged_data = pd.merge(demographic_data,
                               medical_conditions_data, on='SEQN')
        combined_information_data = pd.merge(merged_data, data,
                                             on='SEQN')
    else:

        if medical.endswith('.XPT'):
            medical_conditions_data = pd.read_sas(medical)
        else:
            medical_conditions_data = pd.read_csv(medical)

        medical_conditions_data = medical_conditions_data[['SEQN',
                                                           health_condition]]
        if laboratory.endswith('.XPT'):
            data = pd.read_sas(laboratory)
        else:
            data = pd.read_csv(laboratory)
        data = data.dropna()
        medical_conditions_data = \
            medical_conditions_data.dropna(subset=[health_condition])

        combined_information_data = pd.merge(data,
                                             medical_conditions_data, on='SEQN')

    # Removes any Null values.
    where_are_NaNs = np.isnan(combined_information_data)
    combined_information_data[where_are_NaNs] = 0
    combined_information_data = combined_information_data

    data_labels = combined_information_data
    feature_use = combined_information_data

    # Changes all the value to 0 except if have health condition, which is changed to 1.
    data = combined_information_data[[health_condition]]. \
        applymap(lambda x: (0 if x != int(code_value) else 1))
    data_labels[health_condition] = data[health_condition]

    features = feature_use.drop(health_condition, axis=1)
    labels = data_labels[health_condition]

    return (data, features, labels)


def concatenate(features, labels):
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
    features_labels = np.append(features_columns, 'label')

    data.columns = features_labels

    return (data, features_columns)


def normalization(concatenate_data_labels, features_columns):
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


def pca(data):
    """
    Finds the maximum variance and summarizes towards two columns.
    :param data: Data
    :return: PCA , two-dimensional data
    """

    pca_data = PCA(n_components=2)
    principal_components_data = pca_data.fit_transform(data)
    return (pca_data, principal_components_data)


def covert_to_dataframe(principal_components_data):
    """
    Converts towards Panda data frame.
    :param principal_components_data: data
    :return: Data as dataframe
    """

    principal_data_Df = pd.DataFrame(data=principal_components_data,
                                     columns=['principal component 1',
                                              'principal component 2'])

    return principal_data_Df


def scale(data):
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


def smote(x, y):
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


def random_forest_classifier_print(
        X_train,
        X_test,
        y_train,
        y_test,
        laboratory,
        demographic,
        demographic_bool,
        medical,
        health_condition,
        code_value,
        info
):
    """
    Applying random_forest_classifier.
    :param X_train: Training data
    :param X_test: Testing data
    :param y_train: Training labels
    :param y_test: Testing labels
    :param laboratory: Laboratory data
    :param demographic: Demographic data
    :param demographic_bool: Bool to include demographic data
    :param medical: Medical data
    :param health_condition: Chosen health condition
    :param code_value: Inputted health code value
    :param info: String explaining which button was pressed
    """

    training_type = RandomForestClassifier(random_state=42)
    training_type.fit(X_train, y_train)
    preds = cross_val_predict(training_type, X_test, y_test, cv=10)

    print("Random forest classifier: ")
    print(f'Accuracy = {accuracy_score(y_test, preds):.2f}'
          f'\nRecall = {recall_score(y_test, preds):.2f}\n')
    cm = confusion_matrix(y_test, preds)
    print(cm)

    save_training(laboratory, demographic, demographic_bool,
                  medical, health_condition,
                  code_value, training_type, info,
                  X_test, y_test)

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


def random_forest_classifier(
        laboratory,
        demographic,
        demographic_bool,
        medical,
        health_condition,
        code_value,
):
    """
    Ensure correct steps taken when performing RFC
    :param laboratory: Laboratory data
    :param demographic: Demographic data
    :param demographic_bool: Either include demographic data or not
    :param medical: Medical data
    :param health_condition: Chosen health condition
    :param code_value: Value for health condition
    :return:
    """

    # Determines what model was applied.
    info = 'RFC'
    (data, x, labels) = combine_data(
        laboratory,
        demographic,
        demographic_bool,
        medical,
        health_condition,
        code_value,
    )

    # Scale
    x = scale(x)

    # SMOTE
    (X_train, X_test, y_train, y_test) = smote(x, labels)
    random_forest_classifier_print(
        X_train,
        X_test,
        y_train,
        y_test,
        laboratory,
        demographic,
        demographic_bool,
        medical,
        health_condition,
        code_value,
        info,
    )


def save_training(
        laboratory,
        demographic,
        demographic_bool,
        medical,
        health_condition,
        code_value,
        training_type,
        info,
        X_test,
        y_test,
):
    """
    Saves testing data and data files as one.
    :param laboratory: Laboratory data
    :param demographic: Demographic data
    :param demographic_bool: Bool to include demographic data
    :param medical: Medical data
    :param health_condition: Chosen health condition
    :param code_value: Inputted health code value
    :param training_type: Training
    :param info: String explaining which button was pressed
    :param X_test: Test data
    :param y_test: Test label data
    """

    partition_laboratory = laboratory  # .partition('.')[0]
    partition_demographic = demographic  # .partition('.')[0]
    partition_medical = medical  # .partition('.')[0]

    # 0 means dont want, 1 means want.
    if demographic_bool != '0':
        demographic_bool = 1
    else:
        demographic_bool = 0

    demographic_bool = '{}'.format(demographic_bool)
    health_condition = '{}'.format(health_condition)

    test_name = '{}+{}+{}+{}+{}+{}+{}.pkl'.format(
        partition_laboratory,
        partition_demographic,
        demographic_bool,
        partition_medical,
        health_condition,
        code_value,
        info,
    )

    document_name = '{}+{}+{}+{}+{}+{}+{}.sav'.format(
        partition_laboratory,
        partition_demographic,
        demographic_bool,
        partition_medical,
        health_condition,
        code_value,
        info,
    )

    # If the file path is too long, the except is exceeded reducing the length.
    try:
        document_name = SAV_URL + '\\' + document_name
        test_name = PKL_URL + '\\' + test_name

        pickle.dump(training_type, open(document_name, 'wb'))
        with open(test_name, 'wb') as f:
            pickle.dump([X_test, y_test, training_type], f)

    except OSError:
        partition_laboratory_first = partition_laboratory[0:11]
        partition_laboratory_last = partition_laboratory[-9:]
        partition_laboratory = partition_laboratory_first \
                               + partition_laboratory_last

        partition_demographic_first = partition_demographic[0:12]
        partition_demographic_last = partition_demographic[-9:]
        partition_demographic = partition_demographic_first \
                                + partition_demographic_last

        partition_medical_first = partition_medical[0:11]
        partition_medical_last = partition_medical[-9:]
        partition_medical = partition_medical_first \
                            + partition_medical_last

        test_name = '{}+{}+{}+{}+{}+{}+{}.pkl'.format(
            partition_laboratory,
            partition_demographic,
            demographic_bool,
            partition_medical,
            health_condition,
            code_value,
            info,
        )

        document_name = '{}+{}+{}+{}+{}+{}+{}.sav'.format(
            partition_laboratory,
            partition_demographic,
            demographic_bool,
            partition_medical,
            health_condition,
            code_value,
            info,
        )

        document_name = SAV_URL + '\\' + document_name
        test_name = PKL_URL + '\\' + test_name

        pickle.dump(training_type, open(document_name, 'wb'))
        with open(test_name, 'wb') as f:
            pickle.dump([X_test, y_test, training_type], f)


def run_svm(
        X_train,
        X_test,
        y_train,
        y_test,
        laboratory,
        demographic,
        demographic_bool,
        medical,
        health_condition,
        code_value,
        info,
):
    """
    Performs support vector machine.
    :param X_train: Training data
    :param X_test: Testing data
    :param y_train: Training label
    :param y_test: Testing label
    :param laboratory: Laboratory data
    :param demographic: Demographic data
    :param demographic_bool: Either to include demographic data or not
    :param medical: Medical data
    :param health_condition: Chosen health condition
    :param code_value: Value for health condition
    :param info: Type of button which was pressed
    :return: Testing data, support vector machine data, Training data
    """

    normaliser = StandardScaler()
    normaliser.fit(X_train)
    X_train = normaliser.transform(X_train)
    X_test = normaliser.transform(X_test)

    training_type = SVC(kernel='rbf', gamma='auto', C=1.0,
                        decision_function_shape='ovr').fit(X_train,
                                                           y_train)
    save_training(
        laboratory,
        demographic,
        demographic_bool,
        medical,
        health_condition,
        code_value,
        training_type,
        info,
        X_test,
        y_test,
    )

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


def field_svm_entries(
        laboratory,
        demographic,
        demographic_bool,
        medical,
        health_condition,
        code_value
):
    """
    Ensure correct steps taken when performing svm.
    :param laboratory: Laboratory data
    :param demographic: Demographic data
    :param demographic_bool: Either include demographic data or not
    :param medical: Medical data
    :param health_condition: Chosen health condition
    :param code_value: Value for health condition
    """

    # Determines what model was applied.
    info = 'SVM'
    (data, x, labels) = combine_data(
        laboratory,
        demographic,
        demographic_bool,
        medical,
        health_condition,
        code_value,
    )

    # Scale
    x = scale(x)

    # SMOTE
    (X_train, X_test, y_train, y_test) = smote(x, labels)

    # SVM
    (X_test, training_type, X_train) = run_svm(
        X_train,
        X_test,
        y_train,
        y_test,
        laboratory,
        demographic,
        demographic_bool,
        medical,
        health_condition,
        code_value,
        info,
    )
    scores = cross_val_score(training_type, X_test, y_test, cv=10,
                             scoring='accuracy')

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

    shape = X_train.shape
    number_of_col = shape[1] - 2

    # Print
    two_dim_graph_train(X_train, y_train, training_type, number_of_col)


def browse_files():
    """
    Open files explorer for user to chose a file.
    :return: Chosen file
    """
    #('XPT Files', '*.XPT'),
    data = [ ('All files', '*.*')]
    file = askopenfilename(filetypes=data, defaultextension=data,
                           title='Please select a file:')

    if len(file) != 0:
        showinfo(title='Selected File', message=file)
    else:
        showinfo(title='Selected File', message='No file was selected')

    return file


def search_web(url, file_name):
    """
    Searches the NHANES website with the url and training_type.
    Ensuring the selected file exists.
    :param url: Link to file
    :param file_name: passed in file name
    :return: Values of all files existing
    """

    values = []
    # Tries to connect to the website.
    try:
        content = urllib.request.urlopen(url)
        read_content = content.read()
        soup = BeautifulSoup(read_content, 'html.parser')
        # Loops through the site table.
        for all_elements in soup.find_all('table',
                                          {'class': 'table table-bordered '
                                                    'table-header-light '
                                                    'table-striped table-hover '
                                                    'table-hover-light nein-scroll'
                                           }):

            subAll = all_elements.find('tbody')
            # Finds the specific element of the table.
            for i in subAll.find_all('td', {'class': 'text-center'}):
                searched_file = i.text
                size = len(searched_file)
                searched_file = searched_file[:size - 4]
                searched_file = searched_file.strip()
                correct_file = searched_file
                values.append(correct_file)
    except URLError or HTTPError:
        values.append(os.path.basename(file_name))
    return values


def push_to_check(
        laboratory,
        demographic,
        medical,
        laboratory_year,
        demographic_year,
        medical_year,
):
    """
    Used for checking the file and saving the file
    :param laboratory: Laboratory data
    :param demographic: Demographic data
    :param medical: Medical data
    :param laboratory_year: Laboratory chosen year
    :param demographic_year: Demographic chosen year
    :param medical_year: Medical chosen year
    """

    laboratory_text = laboratory_year[:4]
    laboratory_url = \
        'https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component' \
        '=Laboratory&CycleBeginYear=' \
        + laboratory_text
    demographic_text = demographic_year[:4]
    demographic_url = \
        'https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component' \
        '=Demographics&CycleBeginYear=' \
        + demographic_text
    medical_text = medical_year[:4]
    medical_url = \
        'https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component' \
        '=Questionnaire&CycleBeginYear=' \
        + medical_text

    if len(laboratory) != 0:
        correct_string = search_web(laboratory_url, laboratory)
        file_name = os.path.basename(laboratory)
        size = len(file_name)
        file_name = file_name[:size - 4]
        check = False
        for i in correct_string:
            print(i)
            if i == file_name:
                check = True
        print(file_name)
        if check:
            file_name = file_name + '_' + laboratory_year + '.XPT'
            url = LABORATORY_URL + '/' + file_name
            exists = os.path.exists(url)
            if exists:
                showinfo(title='Error', message='The file NAME: '
                                                + file_name + ' exists')
            else:
                shutil.copyfile(laboratory, file_name)
                shutil.move(file_name, LABORATORY_URL)
                showinfo(title='Added', message='The file ' + file_name
                                                + ' was added...')
        else:
            showinfo(title='Error', message='The file NAME: '
                                            + file_name + ' Does not exist')

    if len(demographic) != 0:
        correct_string = search_web(demographic_url, demographic)
        file_name = os.path.basename(demographic)
        size = len(file_name)
        file_name = file_name[:size - 4]
        if file_name == correct_string[0]:
            file_name = file_name + '_' + demographic_year + '.XPT'
            url = DEMOGRAPHIC_URL + '/' + file_name
            exists = os.path.exists(url)
            if exists:
                showinfo(title='Error', message='The file NAME: '
                                                + file_name + ' exists')
            else:
                shutil.copyfile(demographic, file_name)
                shutil.move(file_name, DEMOGRAPHIC_URL)
                showinfo(title='Added', message='The file ' + file_name
                                                + ' was added...')
        else:
            text = correct_string[0]
            showinfo(title='Error', message='The file NAME: '
                                            + file_name + ' YEAR: ' + demographic_text
                                            + ' Does not match website Name: ' + text
                                            + ' With the Year: ' + demographic_text)

    if len(medical) != 0:
        correct_string = search_web(medical_url, medical)
        file_name = os.path.basename(medical)
        size = len(file_name)
        file_name = file_name[:size - 4]
        check = False
        for i in correct_string:
            if i == file_name:
                check = True
        if check:
            file_name = file_name + '_' + medical_year + '.XPT'
            url = MEDICAL_URL + '/' + file_name
            exists = os.path.exists(url)
            if exists:
                showinfo(title='Error', message='The file NAME: '
                                                + file_name + ' exists')
            else:
                shutil.copyfile(medical, file_name)
                shutil.move(file_name, MEDICAL_URL)
                showinfo(title='Added', message='The file ' + file_name
                                                + ' was added...')

        else:
            showinfo(title='Error', message='The file NAME: '
                                            + file_name + ' Does not exist')


def check_files(url):
    """
    Reads all the files in specific directory
    :param url: Directory location
    :return: Array of all the files
    """

    values = []
    for (root, dirs, files) in os.walk(url):
        for file in files:
            fileName = os.path.basename(file)
            values.append(fileName)

    return values


def get_medical_condition_columns(medical_condition):
    """
    Gets the columns of the medical condition
    :param medical_condition:
    :return: Columns of the medical condition
    """

    if len(medical_condition) != 0:
        str_url = MEDICAL_URL + '\\' + medical_condition
        if str_url.endswith('.XPT'):
            read_files = pd.read_sas(str_url)
            return read_files.columns
        else:
            read_files = pd.read_csv(str_url)
            return read_files.columns


def field_pca_entries(
        laboratory,
        demographic,
        demographic_bool,
        medical,
        health_condition,
        code_value,
        root,
):
    """
    Ensure correct steps taken when performing svm.
    :param laboratory: Laboratory data
    :param demographic: Demographic data
    :param demographic_bool: Either include demographic data or not
    :param medical: Medical data
    :param health_condition: Chosen health condition
    :param code_value: Value for health condition
    :param root: Original window
    """

    # shows which model was applied.
    info = 'SVM_PCA'
    (data, features, labels) = combine_data(
        laboratory,
        demographic,
        demographic_bool,
        medical,
        health_condition,
        code_value,
    )
    (concatenate_data_labels, features_columns) = concatenate(features,
                                                              labels)

    # PCA
    (x_data, feature_named_column) = \
        normalization(concatenate_data_labels, features_columns)
    (pca_data, principal_components_data) = pca(x_data)
    x = covert_to_dataframe(principal_components_data)

    # Scale
    x = scale(x)

    # SMOTE
    (X_train, X_test, y_train, y_test) = smote(x, labels)

    # SVM
    (X_test, training_type, X_train) = run_svm(
        X_train,
        X_test,
        y_train,
        y_test,
        laboratory,
        demographic,
        demographic_bool,
        medical,
        health_condition,
        code_value,
        info,
    )
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
    two_dim_graph_train(X_train, y_train, training_type, number_of_col)


def insert_dash(string, index):
    """
    Inserts dash towards a string at index.
    :param string: String
    :param index: Number where to insert dash
    :return: New string with dash
    """

    return string[:index] + '-' + string[index:]


def next_function(
        chosen_laboratory_file,
        chosen_demographic_file,
        demographic_selected,
        chosen_medical_file,
        step_two,
        root,
):
    """
    Shows selection of medical option and performance verification.
    :param chosen_laboratory_file: Laboratory file name
    :param chosen_demographic_file: Demographic file name
    :param demographic_selected: Either select demographic file or not
    :param chosen_medical_file: Medical file name
    :param step_two: Position
    :param root: Original window
    """

    file_name = chosen_medical_file.get()
    file_columns = get_medical_condition_columns(file_name)

    if len(file_columns) != 0:
        chosen_health_condition = tk.StringVar(root)

        # chosen_health_condition.set(file_columns[0])

        comboLab = tkinter.OptionMenu(step_two,
                                      chosen_health_condition, *file_columns)
        comboLab.grid(row=2, column=3, sticky='EW')

        entry_text_medical = tk.StringVar(value='Code value (Target)')
        inFileTxt = tkinter.Entry(step_two,
                                  textvariable=entry_text_medical)
        inFileTxt.grid(row=2, column=4, sticky='WE', pady=3, ipadx=10)

    inFileBtnMed = tkinter.Button(step_two, text='APPLY SVM ...',
                                  command=lambda: \
                                      check_medical_condition(
                                          chosen_laboratory_file,
                                          chosen_demographic_file,
                                          demographic_selected,
                                          chosen_medical_file,
                                          chosen_health_condition,
                                          entry_text_medical,
                                          'SVM',
                                          step_two,
                                          root,
                                      ))
    inFileBtnMed.grid(row=3, column=3, sticky='WE', padx=5, pady=2)

    inFileBtnMed = tkinter.Button(step_two, text='APPLY PCA & SVM ...',
                                  command=lambda: \
                                      check_medical_condition(
                                          chosen_laboratory_file,
                                          chosen_demographic_file,
                                          demographic_selected,
                                          chosen_medical_file,
                                          chosen_health_condition,
                                          entry_text_medical,
                                          'PCA&SVM',
                                          step_two,
                                          root,
                                      ))
    inFileBtnMed.grid(row=3, column=4, sticky='WE', padx=5, pady=2)

    inFileBtnMed = tkinter.Button(step_two, text='APPLY RFC ...',
                                  command=lambda: \
                                      check_medical_condition(
                                          chosen_laboratory_file,
                                          chosen_demographic_file,
                                          demographic_selected,
                                          chosen_medical_file,
                                          chosen_health_condition,
                                          entry_text_medical,
                                          'RFC',
                                          step_two,
                                          root,
                                      ))

    inFileBtnMed.grid(row=3, column=5, sticky='WE', padx=5, pady=2)


def check_medical_condition(
        chosen_laboratory_file,
        chosen_demographic_file,
        demographic_selected,
        chosen_medical_file,
        chosen_health_condition,
        entry_text_medical,
        training_type,
        step_two,
        root,
):
    """
    Checks the medical condition and input text with
    the medical file exists on the website.
    :param chosen_laboratory_file: Laboratory file name
    :param chosen_demographic_file: Demographic file name
    :param demographic_selected: Either select demographic file or not
    :param chosen_medical_file: Medical file name
    :param chosen_health_condition: Health condition name
    :param entry_text_medical: String of chosen health condition
    :param training_type: Training way
    :param step_two: Position
    :param root: Original window
    :return If condition met return nothing
    """

    medical_result = ''.join(i for i in chosen_medical_file.get()
                             if not i.isdigit())
    size = len(medical_result)
    medical_result = medical_result[:size - 6]
    partitionMedical = medical_result.split('_-_')

    medical_date = ''.join(i for i in chosen_medical_file.get()
                           if i.isdigit())

    n = 8
    partition_medical_date = [medical_date[i:i + n]
                              for i in range(0,len(medical_date), n)]

    for x in range(len(partitionMedical)):
        medical_date = insert_dash(partition_medical_date[x], 4)
        medical_result = partitionMedical[x]

        medical_url = 'https://wwwn.cdc.gov/Nchs/Nhanes/' \
                      + medical_date + '/' + medical_result + '.htm'

        try:
            content = urllib.request.urlopen(medical_url)
            read_content = content.read()
            soup = BeautifulSoup(read_content, 'html.parser')
            lower_case_string = chosen_health_condition.get()
            upper_case_string = chosen_health_condition.get()
            if len(lower_case_string) == 0 or len(upper_case_string) \
                    == 0:
                showinfo(title='Empty',
                         message='Need to pick medical condition')
                return
            else:
                if not lower_case_string[-1].isdigit():
                    removeLast = lower_case_string[:-1]
                    lower_case_string = lower_case_string[-1].lower()
                    lower_case_string = removeLast + lower_case_string

            output = ''
            store = []

            for all_elements in soup.find_all('div', {'id': 'Codebook'
                                                      }):
                for i in all_elements.find_all('div',
                                               {'class': 'pagebreak'}):
                    for z in i.find_all('h3',
                                        {'id': lower_case_string}):
                        sub_all = z.text
                        size = len(lower_case_string)
                        output = sub_all[:size]
                        outputNum = i

                        out = outputNum.find_all('td', {'scope': 'row'})
                        for y in out:
                            store.append(y.text)

                    for z in i.find_all('h3',
                                        {'id': upper_case_string}):
                        sub_all = z.text
                        size = len(upper_case_string)
                        output = sub_all[:size]
                        outputNum = i

                        out = outputNum.find_all('td', {'scope': 'row'})
                        for e in out:
                            store.append(e.text)

            get_bool = check_next(
                chosen_laboratory_file,
                chosen_demographic_file,
                demographic_selected,
                chosen_medical_file,
                step_two,
                root,
                RETURN,
                1,
            )

            if output.lower() == lower_case_string.lower() \
                    and entry_text_medical.get() \
                    in store:
                if get_bool:
                    if training_type == 'SVM':
                        field_svm_entries(
                            chosen_laboratory_file,
                            chosen_demographic_file,
                            demographic_selected.get(),
                            chosen_medical_file.get(),
                            chosen_health_condition.get(),
                            entry_text_medical.get(),
                            1
                        )
                        return
                    elif training_type == 'PCA&SVM':
                        field_pca_entries(
                            chosen_laboratory_file,
                            chosen_demographic_file,
                            demographic_selected.get(),
                            chosen_medical_file.get(),
                            chosen_health_condition.get(),
                            entry_text_medical.get(),
                            root,
                        )
                        return
                    elif training_type == 'RFC':
                        random_forest_classifier(
                            chosen_laboratory_file,
                            chosen_demographic_file,
                            demographic_selected.get(),
                            chosen_medical_file.get(),
                            chosen_health_condition.get(),
                            entry_text_medical.get(),
                        )
                        return
                    elif training_type == 'Features':
                        gradient_boosting_regression(
                            chosen_laboratory_file,
                            chosen_demographic_file,
                            demographic_selected.get(),
                            chosen_medical_file.get(),
                            chosen_health_condition.get(),
                            entry_text_medical.get(),
                        )
                        return
            else:
                showinfo(title='Selected File', message=medical_result
                                                        + ' Does not contain: '
                                                        + entry_text_medical.get())
        except URLError or HTTPError:

            get_bool = check_next(
                chosen_laboratory_file,
                chosen_demographic_file,
                demographic_selected,
                chosen_medical_file,
                step_two,
                root,
                RETURN,
                1
            )
            if get_bool:
                if training_type == 'SVM':
                    field_svm_entries(
                        chosen_laboratory_file,
                        chosen_demographic_file,
                        demographic_selected.get(),
                        chosen_medical_file.get(),
                        chosen_health_condition.get(),
                        entry_text_medical.get(),
                        1
                    )
                elif training_type == 'PCA&SVM':
                    field_pca_entries(
                        chosen_laboratory_file.get(),
                        chosen_demographic_file,
                        demographic_selected.get(),
                        chosen_medical_file.get(),
                        chosen_health_condition.get(),
                        entry_text_medical.get(),
                        root,
                    )
                elif training_type == 'RFC':
                    random_forest_classifier(
                        chosen_laboratory_file,
                        chosen_demographic_file,
                        demographic_selected.get(),
                        chosen_medical_file.get(),
                        chosen_health_condition.get(),
                        entry_text_medical.get(),
                        root,
                    )
                elif training_type == 'Features':
                    gradient_boosting_regression(
                        chosen_laboratory_file,
                        chosen_demographic_file,
                        demographic_selected.get(),
                        chosen_medical_file.get(),
                        chosen_health_condition.get(),
                        entry_text_medical.get(),
                        root,
                    )


def gradient_boosting_regression(
        laboratory,
        demographic,
        demographic_bool,
        medical,
        health_condition,
        code_value,
):
    """
    Gets the data ready for applying it towards feature select.
    Retrieving testing and training data.
    :param laboratory: Laboratory file
    :param demographic: Demographic file
    :param demographic_bool: Demographic checked
    :param medical: Medical file
    :param health_condition: Chosen medical column
    :param code_value: Inserte code value
    """
    (data, x, labels) = combine_data(
        laboratory,
        demographic,
        demographic_bool,
        medical,
        health_condition,
        code_value,
    )

    # Scale
    x = scale(x)

    X_train, X_test, y_train, y_test = train_test_split(x, labels, test_size=0.2,
                                                        random_state=42)
    perform_feature_selection(X_train, X_test, y_train, y_test)


def perform_feature_selection(X_train, X_test, y_train, y_test):
    """
    Method to perform feature selection using gradient booster.
    :param X_train: Training data
    :param X_test: Testing data
    :param y_train: Training label
    :param y_test: Testing label
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
    important_feature = important_feature.sort_values('importance', ascending=False)\
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


def clear(root):
    """
    Clears the window and reinitialize.
    :param root: Original window
    """

    window_list = root.grid_slaves()
    for all_elements in window_list:
        all_elements.destroy()
    add_data_to_gui(root)
    second_multiple_options(root)
    third_position_in_window(root)


def second_position_in_window(root, num):
    """
    Populates the window.
    :param root: Original window
    """

    stepTwo = tkinter.LabelFrame(root,
                                 text=' 2. Train Machine in detecting...'
                                 )
    stepTwo.grid(
        row=2,
        column=0,
        sticky='W',
        padx=5,
        pady=5,
        ipadx=5,
        ipady=5,
    )

    inFileLbl = tkinter.Label(stepTwo, text='Select the Laboratory data'
                              )
    inFileLbl.grid(row=0, sticky='E', column=0, padx=5, pady=2)

    # Instead of cbc needs to read the files.

    numL = 0
    get_files_l = check_files(LABORATORY_URL)
    if len(get_files_l) != 0:
        chosen_laboratory_file = tk.StringVar(root)
        # chosen_laboratory_file.set(get_files_l[0])
        comboLab = tkinter.OptionMenu(stepTwo, chosen_laboratory_file,
                                      *get_files_l)
        comboLab.grid(row=numL, column=1, sticky='EW', ipadx=10)
    else:

        chosen_laboratory_file = tk.StringVar(root)
        comboLab = tkinter.OptionMenu(stepTwo, chosen_laboratory_file,
                                      value='')
        comboLab.grid(row=numL, column=1, sticky='W', padx=5, pady=2)

    inFileLbl = tkinter.Label(stepTwo,
                              text='Select the demographic data')
    inFileLbl.grid(row=1, sticky='E', column=0, padx=5, pady=2)

    numD = 1
    get_files_d = check_files(DEMOGRAPHIC_URL)
    if len(get_files_d) != 0:
        chosen_demographic_file = tk.StringVar(root)
        comboLab = tkinter.OptionMenu(stepTwo, chosen_demographic_file,
                                      *get_files_d)
        comboLab.grid(row=numD, column=1, sticky='EW', ipadx=10)

        check_variable = tk.StringVar(root)
        getFldChkD = tkinter.Checkbutton(stepTwo,
                                         text='Include demographic data...',
                                         variable=check_variable)
        getFldChkD.grid(row=numD, column=2, columnspan=3, pady=2,
                        sticky='W')
    else:

        chosen_demographic_file = tk.StringVar(root)
        comboLab = tkinter.OptionMenu(stepTwo, chosen_demographic_file,
                                      value='')
        comboLab.grid(row=numD, column=1, sticky='W', padx=5, pady=2)

    inFileLbl = tkinter.Label(stepTwo, text='Select the medical data')
    inFileLbl.grid(row=2, sticky='E', column=0, padx=10, pady=2)

    numM = 2
    get_files_m = check_files(MEDICAL_URL)
    if len(get_files_m) != 0:
        chosen_medical_file = tk.StringVar(root)
        comboLab = tkinter.OptionMenu(stepTwo, chosen_medical_file,
                                      *get_files_m)
        comboLab.grid(row=numM, column=1, sticky='EW', ipadx=10)
    else:

        chosen_medical_file = tk.StringVar(root)
        comboLab = tkinter.OptionMenu(stepTwo, chosen_medical_file,
                                      value='')
        comboLab.grid(row=numM, column=1, sticky='W', padx=5, pady=2)

    inFileBtnMed = tkinter.Button(stepTwo, text='Next...',
                                  command=lambda: check_next(
                                      chosen_laboratory_file.get(),
                                      chosen_demographic_file.get(),
                                      check_variable,
                                      chosen_medical_file,
                                      stepTwo,
                                      root,
                                      NO_RETURN,
                                      num,
                                  ))
    inFileBtnMed.grid(row=2, column=2, sticky='W', padx=5, pady=2)

    inFileBtnMed = tkinter.Button(stepTwo, text='REFRESH ...',
                                  command=lambda: clear(root))
    inFileBtnMed.grid(row=3, column=2, sticky='WE', padx=5, pady=2)


def check_next(
        chosen_laboratory_file,
        chosen_demographic_file,
        demographic_selected,
        chosen_medical_file,
        step_two,
        root,
        string_condition,
        num,
):
    """
    Validation of selected files.
    :param chosen_laboratory_file: Laboratory file name
    :param chosen_demographic_file: Demographic file name
    :param demographic_selected: Either select demographic file or not
    :param chosen_medical_file: Medical file name
    :param step_two: Position
    :param root: Original window
    :param string_condition: string used for condition
    :return: Boolean depending on condition
    """

    laboratory_date = ''.join(i for i in chosen_laboratory_file
                              if i.isdigit())
    demographic_date = ''.join(i for i in chosen_demographic_file
                               if i.isdigit())
    medical_date = ''.join(i for i in chosen_medical_file.get()
                           if i.isdigit())

    # 0 means dont want, 1 means want
    try:
        if demographic_selected.get() != '0':
            demographic_bool = 1
        else:
            demographic_bool = 0
    except AttributeError:
        demographic_bool = 1

    if demographic_bool == 1:
        if string_condition == NO_RETURN:
            if laboratory_date == demographic_date and laboratory_date \
                    == medical_date:
                if num == 1:
                    next_function(
                        chosen_laboratory_file,
                        chosen_demographic_file,
                        demographic_selected,
                        chosen_medical_file,
                        step_two,
                        root,
                    )
                else:
                    feature_selection_window(
                        chosen_laboratory_file,
                        chosen_demographic_file,
                        demographic_selected,
                        chosen_medical_file,
                        step_two,
                        root,
                    )
            else:
                showinfo(title='Error',
                         message='The selected dates do not match - '
                                 'ensure all dates are the same... '
                                 'Laboratory Date, Demographic Date '
                                 'and Medical Date needs changing.'
                         )

                clear(root)
        elif string_condition == RETURN:
            if laboratory_date == demographic_date and laboratory_date \
                    == medical_date:
                return True
            else:
                showinfo(title='Error',
                         message='The selected dates do not match - '
                                 'ensure all dates are the same... '
                                 'Laboratory Date, Demographic Date and '
                                 'Medical Date needs changing.'
                         )

                clear(root)
                return False
    else:
        if string_condition == NO_RETURN:
            if laboratory_date == medical_date:
                if num == 1:
                    next_function(
                        chosen_laboratory_file,
                        chosen_demographic_file,
                        demographic_selected,
                        chosen_medical_file,
                        step_two,
                        root,
                    )
                else:
                    feature_selection_window(
                        chosen_laboratory_file,
                        chosen_demographic_file,
                        demographic_selected,
                        chosen_medical_file,
                        step_two,
                        root,
                    )
            else:
                showinfo(title='Error',
                         message='The selected dates do not match - '
                                 'ensure all dates are the same... '
                                 'Laboratory Date and Medical '
                                 'Date needs changing.'
                         )

                clear(root)
        elif string_condition == RETURN:
            if laboratory_date == demographic_date and laboratory_date \
                    == medical_date:
                return True
            else:
                showinfo(title='Error',
                         message='The selected dates do not match - '
                                 'ensure all dates are the same... '
                                 'Laboratory Date, Demographic Date and Medical '
                                 'Date needs changing.'
                         )

                clear(root)
                return False


def read_file():
    """
    Reads all the saved testing data in the directory and breaks the file name up.
    Adding towards array.
    :return: Laboratory array, demographic array, demographic checked array,
    medical array, health condition array, chosen health array, array of medical text.
    """

    laboratory_values = []
    demographic_values = []
    demographic_bool_values = []
    medical_values = []
    health_condition_values = []
    code_values = []
    info_values = []

    for (root, dirs, files) in os.walk(SAV_URL):
        for file in files:
            file_name = os.path.basename(file)
            (
                labroatory,
                demographic,
                demographicBool,
                medical,
                healthCondition,
                codeValue,
                info,
            ) = file_name.split('+')
            labroatory = '{}'.format(labroatory)
            demographic = '{}'.format(demographic)
            medical = '{}'.format(medical)
            info = info.partition('.')[0]

            if labroatory not in laboratory_values:
                laboratory_values.append(labroatory)
            if demographic not in demographic_values:
                demographic_values.append(demographic)
            if demographicBool not in demographic_bool_values:
                demographic_bool_values.append(demographicBool)
            if medical not in medical_values:
                medical_values.append(medical)
            if healthCondition not in health_condition_values:
                health_condition_values.append(healthCondition)
            if codeValue not in code_values:
                code_values.append(codeValue)
            if info not in info_values:
                info_values.append(info)

    return (
        laboratory_values,
        demographic_values,
        demographic_bool_values,
        medical_values,
        health_condition_values,
        code_values,
        info_values,
    )


def third_position_in_window(root):
    """
    Populates the original window.
    :param root: Original window
    """

    (
        laboratory_values,
        demographic_values,
        demographic_bool_values,
        medical_values,
        health_condition_values,
        code_values,
        info_values,
    ) = read_file()
    step_three = tkinter.LabelFrame(root,
                                    text=' 3. Using prediction using testing data.'
                                    )
    step_three.grid(
        row=3,
        columnspan=7,
        sticky='W',
        padx=5,
        pady=5,
        ipadx=5,
        ipady=5,
    )

    inFileLbl = tkinter.Label(step_three,
                              text='Select the Laboratory data')
    inFileLbl.grid(row=0, sticky='E', column=0, padx=5, pady=2)

    numL = 0
    if len(laboratory_values) != 0:
        variableL = tk.StringVar(root)

        comboLab = tkinter.OptionMenu(step_three, variableL,
                                      *laboratory_values)
        comboLab.grid(row=numL, column=1, sticky='EW', ipadx=10)
    else:

        variableL = tk.StringVar(root)
        comboLab = tkinter.OptionMenu(step_three, variableL, value='')
        comboLab.grid(row=numL, column=1, sticky='W', padx=5, pady=2)

    inFileLbl = tkinter.Label(step_three,
                              text='Select the demographic data')
    inFileLbl.grid(row=1, sticky='E', column=0, padx=5, pady=2)

    numL = 1
    if len(demographic_values) != 0:
        variableD = tk.StringVar(root)
        comboLab = tkinter.OptionMenu(step_three, variableD,
                                      *demographic_values)
        comboLab.grid(row=numL, column=1, sticky='EW', ipadx=10)
    else:

        variableL = tk.StringVar(root)
        comboLab = tkinter.OptionMenu(step_three, variableL, value='')
        comboLab.grid(row=numL, column=1, sticky='W', padx=5, pady=2)

    inFileLbl = tkinter.Label(step_three, text='Include demographics')
    inFileLbl.grid(row=2, sticky='E', column=0, padx=5, pady=2)

    numL = 2
    if len(demographic_bool_values) != 0:
        variableDB = tk.StringVar(root)

        comboLab = tkinter.OptionMenu(step_three, variableDB,
                                      *demographic_bool_values)
        comboLab.grid(row=numL, column=1, sticky='EW', ipadx=10)
    else:

        variableL = tk.StringVar(root)
        comboLab = tkinter.OptionMenu(step_three, variableL, value='')
        comboLab.grid(row=numL, column=1, sticky='W', padx=5, pady=2)

    inFileLbl = tkinter.Label(step_three, text='Select the medical data'
                              )
    inFileLbl.grid(row=3, sticky='E', column=0, padx=5, pady=2)

    numL = 3
    if len(medical_values) != 0:
        variableM = tk.StringVar(root)

        comboLab = tkinter.OptionMenu(step_three, variableM,
                                      *medical_values)
        comboLab.grid(row=numL, column=1, sticky='EW', ipadx=10)
    else:

        variableL = tk.StringVar(root)
        comboLab = tkinter.OptionMenu(step_three, variableL, value='')
        comboLab.grid(row=numL, column=1, sticky='W', padx=5, pady=2)

    inFileLbl = tkinter.Label(step_three,
                              text='Select the medical condition')
    inFileLbl.grid(row=4, sticky='E', column=0, padx=5, pady=2)

    numL = 4
    if len(health_condition_values) != 0:
        variableMC = tk.StringVar(root)

        comboLab = tkinter.OptionMenu(step_three, variableMC,
                                      *health_condition_values)
        comboLab.grid(row=numL, column=1, sticky='EW', ipadx=10)
    else:

        variableL = tk.StringVar(root)
        comboLab = tkinter.OptionMenu(step_three, variableL, value='')
        comboLab.grid(row=numL, column=1, sticky='W', padx=5, pady=2)

    inFileLbl = tkinter.Label(step_three, text='Select the code value')
    inFileLbl.grid(row=5, sticky='E', column=0, padx=5, pady=2)

    numL = 5
    if len(code_values) != 0:
        variableCV = tk.StringVar(root)

        comboLab = tkinter.OptionMenu(step_three, variableCV,
                                      *code_values)
        comboLab.grid(row=numL, column=1, sticky='EW', ipadx=10)
    else:

        variableL = tk.StringVar(root)
        comboLab = tkinter.OptionMenu(step_three, variableL, value='')
        comboLab.grid(row=numL, column=1, sticky='W', padx=5, pady=2)

    inFileLbl = tkinter.Label(step_three,
                              text='Select the way the machine was trained...'
                              )
    inFileLbl.grid(row=6, sticky='E', column=0, padx=5, pady=2)

    numL = 6
    if len(info_values) != 0:
        variableI = tk.StringVar(root)

        comboLab = tkinter.OptionMenu(step_three, variableI,
                                      *info_values)
        comboLab.grid(row=numL, column=1, sticky='EW', ipadx=10)
    else:

        variableL = tk.StringVar(root)
        comboLab = tkinter.OptionMenu(step_three, variableL, value='')
        comboLab.grid(row=numL, column=1, sticky='W', padx=5, pady=2)

    inFileBtnMed = tkinter.Button(step_three, text='REFRESH ...',
                                  command=lambda: clear(root))
    inFileBtnMed.grid(row=7, column=4, sticky='WE', padx=5, pady=2)

    inFileBtnMed = tkinter.Button(step_three, text='PREDICT ...',
                                  command=lambda: test_data_pressed(
                                      variableL.get(),
                                      variableD.get(),
                                      variableDB.get(),
                                      variableM.get(),
                                      variableMC.get(),
                                      variableCV.get(),
                                      variableI.get(),
                                  ))
    inFileBtnMed.grid(row=7, column=5, sticky='WE', padx=5, pady=2)


def feature_selection_window(chosen_laboratory_file,
                             chosen_demographic_file,
                             demographic_selected,
                             chosen_medical_file,
                             step_two,
                             root, ):
    """
    Populates the window of getting features
    :param chosen_laboratory_file: Laboratory file
    :param chosen_demographic_file: Demographic file
    :param demographic_selected: Demographic selected
    :param chosen_medical_file: Medical file
    :param step_two: Root placement
    :param root: root
    """
    file_name = chosen_medical_file.get()
    file_columns = get_medical_condition_columns(file_name)

    if len(file_columns) != 0:
        chosen_health_condition = tk.StringVar(root)

        comboLab = tkinter.OptionMenu(step_two,
                                      chosen_health_condition, *file_columns)
        comboLab.grid(row=2, column=3, sticky='EW')

        entry_text_medical = tk.StringVar(value='Code value (Target)')
        inFileTxt = tkinter.Entry(step_two,
                                  textvariable=entry_text_medical)
        inFileTxt.grid(row=2, column=4, sticky='WE', pady=3, ipadx=10)

        inFileBtnMed = tkinter.Button(step_two, text='Get features ...',
                                      command=lambda: \
                                          check_medical_condition(
                                              chosen_laboratory_file,
                                              chosen_demographic_file,
                                              demographic_selected,
                                              chosen_medical_file,
                                              chosen_health_condition,
                                              entry_text_medical,
                                              'Features',
                                              step_two,
                                              root,
                                          ))
        inFileBtnMed.grid(row=3, column=4, sticky='WE', padx=5, pady=2)

        inFileBtnMed = tkinter.Button(step_two, text='Remove ...',
                                      command=lambda: \
                                          remove_input(entry_text, chosen_laboratory_file,
                                                       chosen_demographic_file,
                                                       chosen_medical_file.get()))
        inFileBtnMed.grid(row=3, column=3, sticky='WE', padx=5, pady=2)

        entry_text = tk.StringVar(value='')
        inFileTxt = tkinter.Entry(step_two, textvariable=entry_text)
        inFileTxt.grid(row=3, column=1, sticky='WE', pady=3, ipadx=10)


def remove_input(entry_text,
                 chosen_laboratory_file,
                 chosen_demographic_file,
                 chosen_medical_file):
    """
    After feature selection, by passing in input.
    The input column is removed from the data.
    :param entry_text: Passed in input
    :param chosen_laboratory_file: Laboratory file
    :param chosen_demographic_file: Demographic file
    :param chosen_medical_file: Medical file
    """
    entry_text = entry_text.get()
    laboratory = LABORATORY_URL + '\\' + str(chosen_laboratory_file)
    demographic = DEMOGRAPHIC_URL + '\\' + str(chosen_demographic_file)
    medical = MEDICAL_URL + '\\' + str(chosen_medical_file)

    if demographic.endswith('.XPT'):
        demographic_data = pd.read_sas(demographic)
    else:
        demographic_data = pd.read_csv(demographic)

    if medical.endswith('.XPT'):
        medical_conditions_data = pd.read_sas(medical)
    else:
        medical_conditions_data = pd.read_csv(medical)

    if laboratory.endswith('.XPT'):
        data = pd.read_sas(laboratory)
    else:
        data = pd.read_csv(laboratory)

    if entry_text in data.columns:
        chosen_laboratory_file = str(chosen_laboratory_file)
        data = data.drop(labels=[entry_text], axis=1, inplace=False)

        size = len(chosen_laboratory_file)
        mod_string = chosen_laboratory_file[:size - 4]

        file_name = mod_string + '.csv'
        lab_url = LABORATORY_URL + '\\' + file_name

        df = pd.DataFrame(data, columns=data.columns)
        df.to_csv(lab_url, index=False, header=True)

    elif entry_text in demographic_data.columns:
        chosen_demographic_file = str(chosen_demographic_file)
        demographic_data = demographic_data.drop(labels=[entry_text],
                                                 axis=1, inplace=False)

        size = len(chosen_demographic_file)
        mod_string = chosen_demographic_file[:size - 4]

        file_name = mod_string + '.csv'
        dem_url = DEMOGRAPHIC_URL + '\\' + file_name

        df = pd.DataFrame(demographic_data, columns=demographic_data.columns)
        df.to_csv(dem_url, index=False, header=True)

    elif entry_text in medical_conditions_data.columns:
        chosen_medical_file = str(chosen_medical_file)
        medical_conditions_data = medical_conditions_data.drop(labels=[entry_text],
                                                               axis=1, inplace=False)

        size = len(chosen_medical_file)
        mod_string = chosen_medical_file[:size - 4]

        file_name = mod_string + '.csv'
        medical_url = MEDICAL_URL + '\\' + file_name

        df = pd.DataFrame(medical_conditions_data, columns=medical_conditions_data.columns)
        df.to_csv(medical_url, index=False, header=True)
    else:
        showinfo(title='Does not exist',
                 message='The selected name: ' + entry_text
                         + ' Does not exist.')


def test_data_pressed(
        chosen_laboratory_file,
        chosen_demographic_file,
        demographic_selected,
        chosen_medical_file,
        chosen_health_condition,
        entry_text_medical,
        string_type,
):
    """
    Opens the directory of testing data and retrieves testing data. Shown in a graph.
    :param chosen_laboratory_file: Laboratory file name
    :param chosen_demographic_file: Demographic file name
    :param demographic_selected: Either select demographic file or not
    :param chosen_medical_file: Medical file name
    :param chosen_health_condition: Health condition name
    :param entry_text_medical: String of chosen health condition
    :param string_type: Training way
    """

    selectedFile = chosen_laboratory_file + '+' \
                   + chosen_demographic_file + '+' + demographic_selected + '+' \
                   + chosen_medical_file + '+' + chosen_health_condition + '+' \
                   + entry_text_medical + '+' + string_type + '.pkl'
    selectedFile = PKL_URL + '\\' + selectedFile
    if len(selectedFile) != 0:
        try:
            with open(selectedFile, 'rb') as f:
                (X_test, y_test, training_type) = pickle.load(f)

                if string_type == "SVM":
                    y_pred = training_type.predict(X_test)

                    print("SVM: ")
                    print(f'Accuracy = {accuracy_score(y_test, y_pred):.2f}\n'
                          f'Recall = {recall_score(y_test, y_pred):.2f}\n')
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

                    shape = X_test.shape
                    numberOfCol = shape[1] - 2

                    two_dim_graph_train(X_test, y_test, training_type,
                                        numberOfCol)

                elif string_type == "RFC":
                    preds = training_type.predict(X_test)

                    print("Random forest classifier: ")
                    print(f'Accuracy = {accuracy_score(y_test, preds):.2f}\n'
                          f'Recall = {recall_score(y_test, preds):.2f}\n')
                    cm = confusion_matrix(y_test, preds)
                    print(cm)

                    plt.figure(figsize=(5, 7))
                    ax = sns.distplot(y_test, hist=False, color="r",
                                      label="Actual Value")
                    sns.distplot(preds, hist=False, color="b",
                                 label="Fitted Values", ax=ax)
                    plt.title('Actual vs Fitted Values')
                    plt.show()
                    plt.close()

                    Tree = training_type.estimators_[5]
                    plt.figure(figsize=(25, 15))
                    tree.plot_tree(Tree, filled=True,
                                   rounded=True,
                                   fontsize=14);
                    plt.tight_layout()
                    plt.show()
                    plt.close()
                else:
                    preds = training_type.predict(X_test)
                    print("PCA and SVM: ")
                    print(f'Accuracy = {accuracy_score(y_test, preds):.2f}\n'
                          f'Recall = {recall_score(y_test, preds):.2f}\n')
                    cm = confusion_matrix(y_test, preds)
                    print(cm)

                    shape = X_test.shape
                    numberOfCol = shape[1] - 2

                    two_dim_graph_train(X_test, y_test, training_type,
                                        numberOfCol)

        except EnvironmentError:
            showinfo(title='Does not exist',
                     message='The selected File: ' + selectedFile
                             + ' Does not exist.')


def add_data_to_gui(root):
    """
    Populates the original window with option menus of years and ability to open files.
    :param root: Original window
    """

    years = []
    try:
        content = \
            urllib.request.urlopen('https://wwwn.cdc.gov/nchs/nhanes/default.aspx')
        read_content = content.read()
        soup = BeautifulSoup(read_content, 'html.parser')
        for all_elements in soup.find_all('div',
                                          {'class': 'col-md-3 d-flex'}):
            subAll = all_elements.find('div',
                                       {'class': 'card-title h4 mb text-left'})
            years.append(subAll.br.next_sibling)
    except URLError or HTTPError:
        years = [
            '2019-2020',
            '2017-2018',
            '2015-2016',
            '2013-2014',
            '2011-2012',
            '2009-2010',
            '2007-2008',
            '2005-2006',
            '2003-2004',
            '2001-2002',
            '1999-2000',
        ]

    step_one = tkinter.LabelFrame(root, text=' 1. Add new data')
    step_one.grid(
        row=0,
        columnspan=7,
        sticky='WE',
        padx=5,
        pady=5,
        ipadx=5,
        ipady=5,
    )

    if len(years) != 0:
        laboratory_year = tk.StringVar(root)
        comboLab = tkinter.OptionMenu(step_one, laboratory_year, *years)
        comboLab.grid(row=0, column=7, sticky='EW', ipadx=10)

        demographic_year = tk.StringVar(root)
        comboLab = tkinter.OptionMenu(step_one, demographic_year,
                                      *years)
        comboLab.grid(row=2, column=7, sticky='EW', ipadx=10)

        medical_year = tk.StringVar(root)
        comboLab = tkinter.OptionMenu(step_one, medical_year, *years)
        comboLab.grid(row=4, column=7, sticky='EW', ipadx=10)

    inFileLbl = tkinter.Label(step_one,
                              text='Select new laboratory data:')
    inFileLbl.grid(row=0, sticky='E', column=0, padx=5, pady=2)

    entry_text_lab = tk.StringVar()
    inFileTxt = tkinter.Entry(step_one, textvariable=entry_text_lab)
    inFileTxt.grid(row=0, column=1, sticky='WE', pady=3, ipadx=150)

    inFileBtn = tkinter.Button(step_one, text='Browse ...',
                               command=lambda: \
                                   entry_text_lab.set(browse_files()))
    inFileBtn.grid(row=0, column=8, sticky='W', padx=5, pady=2)

    inFileDemo = tkinter.Label(step_one,
                               text='Select new demographic data:')
    inFileDemo.grid(row=2, sticky='E', column=0, padx=5, pady=2)

    entry_text_demo = tk.StringVar()
    inFileTxt = tkinter.Entry(step_one, textvariable=entry_text_demo)
    inFileTxt.grid(row=2, column=1, sticky='WE', pady=3, ipadx=150)

    inFileBtn = tkinter.Button(step_one, text='Browse ...',
                               command=lambda: \
                                   entry_text_demo.set(browse_files()))
    inFileBtn.grid(row=2, column=8, sticky='W', padx=5, pady=2)

    inFileDemo = tkinter.Label(step_one, text='Select new medical data:'
                               )
    inFileDemo.grid(row=4, sticky='E', column=0, padx=5, pady=2)

    entry_text_med = tk.StringVar()
    inFileTxt = tkinter.Entry(step_one, textvariable=entry_text_med)
    inFileTxt.grid(row=4, column=1, sticky='WE', pady=3, ipadx=150)

    inFileBtn = tkinter.Button(step_one, text='Browse ...',
                               command=lambda: \
                                   entry_text_med.set(browse_files()))
    inFileBtn.grid(row=4, column=8, sticky='W', padx=5, pady=2)

    outFileLbl = tkinter.Label(step_one, text='Save new data:')
    outFileLbl.grid(row=5, column=0, sticky='E', padx=5, pady=2)

    outFileBtn = tkinter.Button(step_one, text='Save ...',
                                command=lambda: push_to_check(
                                    entry_text_lab.get(),
                                    entry_text_demo.get(),
                                    entry_text_med.get(),
                                    laboratory_year.get(),
                                    demographic_year.get(),
                                    medical_year.get(),
                                ))
    outFileBtn.grid(row=5, column=1, sticky='WE', padx=5, pady=2)


def second_multiple_options(root):
    """
    Populating the window with option of combining data or using single data
    :param root: Original window
    """

    step_two = tkinter.LabelFrame(root,
                                  text=' 2. Choose correct option...')
    step_two.grid(
        row=2,
        column=0,
        sticky='W',
        padx=5,
        pady=5,
        ipadx=5,
        ipady=5,
    )

    inFileBtn = tkinter.Button(step_two, text='Train Machine ...',
                               command=lambda: \
                                   second_position_in_window(root, 1))
    inFileBtn.grid(row=0, column=2, sticky='W', padx=5, pady=2)

    inFileBtn = tkinter.Button(step_two, text='Combine Data...',
                               command=lambda: combined_data(root))
    inFileBtn.grid(row=0, column=3, sticky='W', padx=5, pady=2)

    inFileBtn = tkinter.Button(step_two, text='Features...',
                               command=lambda: features(root))
    inFileBtn.grid(row=0, column=4, sticky='W', padx=5, pady=2)


def features(root):
    stepTwo = tkinter.LabelFrame(root, text=' 2. Features...')
    stepTwo.grid(
        row=2,
        column=0,
        sticky='W',
        padx=5,
        pady=5,
        ipadx=5,
        ipady=5,
    )

    inFileLbl = tkinter.Label(stepTwo,
                              text='Get features first and then remove')
    inFileLbl.grid(row=0, sticky='E', column=0, padx=5, pady=2)

    inFileBtn = tkinter.Button(stepTwo, text='Refresh',
                               command=lambda: clear(root))
    inFileBtn.grid(row=1, column=0, sticky='W', padx=5, pady=2)

    inFileBtn = tkinter.Button(stepTwo, text='Get features...',
                               command=lambda: \
                                   second_position_in_window(root, 2))
    inFileBtn.grid(row=1, column=1, sticky='W', padx=5, pady=2)


def combined_data(root):
    """
    Populating the window with the amount of data the user wants to combine.
    :param root: Original window
    """

    stepTwo = tkinter.LabelFrame(root, text=' 2. Combine Data...')
    stepTwo.grid(
        row=2,
        column=0,
        sticky='W',
        padx=5,
        pady=5,
        ipadx=5,
        ipady=5,
    )

    inFileLbl = tkinter.Label(stepTwo,
                              text='How many data you want to combine:')
    inFileLbl.grid(row=0, sticky='E', column=0, padx=5, pady=2)

    entry_text = tk.StringVar(value='')
    inFileTxt = tkinter.Entry(stepTwo, textvariable=entry_text)
    inFileTxt.grid(row=0, column=1, sticky='WE', pady=3, ipadx=10)

    inFileBtn = tkinter.Button(stepTwo, text='Refresh',
                               command=lambda: clear(root))
    inFileBtn.grid(row=1, column=0, sticky='W', padx=5, pady=2)

    inFileBtn = tkinter.Button(stepTwo, text='Next ...',
                               command=lambda: \
                                   number_of_data(entry_text, root,
                                                  stepTwo))
    inFileBtn.grid(row=1, column=1, sticky='W', padx=5, pady=2)


def number_of_data(entry_text, root, step_two):
    """
    Selecting files to combine and verifies matching ones.
    :param entry_text: String of chosen number of files to combine
    :param root: Original window
    :param step_two: Position
    :return: Nothing as it used to leave the function
    """

    entry_text = entry_text.get()
    if not entry_text.isdigit():
        showinfo(title='Error', message='Required a number')
        return

    if int(entry_text) >= 9:
        showinfo(title='Error', message='Choose a smaller number')
        return

    inFileLbl = tkinter.Label(step_two,
                              text='Select the Laboratory data.. '
                                   'The column must match'
                              )
    inFileLbl.grid(row=2, sticky='E', column=0, padx=5, pady=2)

    laboratory_year_store = []
    demographic_year_store = []
    medical_year_store = []

    for x in range(int(entry_text)):
        numL = 3
        getFilesL = check_files(LABORATORY_URL)
        if len(getFilesL) != 0:
            laboratory_year_store.append('laboratory_year' + str(x))
            laboratory_year_store[x] = tk.StringVar(root)
            # laboratory_year_store[x].set(getFilesL[0])
            comboLab = tkinter.OptionMenu(step_two,
                                          laboratory_year_store[x], *getFilesL)
            comboLab.grid(row=numL, column=x, sticky='EW', ipadx=10)
        else:

            laboratory_year_store.append('laboratory_year' + str(x))
            laboratory_year_store[x] = tk.StringVar(root)
            comboLab = tkinter.OptionMenu(step_two,
                                          laboratory_year_store[x], value='')
            comboLab.grid(row=numL, column=x, sticky='W', padx=5,
                          pady=2)

    inFileLbl = tkinter.Label(step_two,
                              text='Select the demographic data... '
                                   'The column must match'
                              )
    inFileLbl.grid(row=4, sticky='E', column=0, padx=5, pady=2)

    for x in range(int(entry_text)):
        numD = 5
        getFilesD = check_files(DEMOGRAPHIC_URL)
        if len(getFilesD) != 0:
            demographic_year_store.append('demographic_year' + str(x))
            demographic_year_store[x] = tk.StringVar(root)
            # demographic_year_store[x].set(getFilesD[0])
            comboLab = tkinter.OptionMenu(step_two,
                                          demographic_year_store[x], *getFilesD)
            comboLab.grid(row=numD, column=x, sticky='EW', ipadx=10)
        else:
            demographic_year_store.append('demographic_year' + str(x))
            demographic_year_store[x] = tk.StringVar(root)
            comboLab = tkinter.OptionMenu(step_two,
                                          demographic_year_store[x], value='')
            comboLab.grid(row=numD, column=x, sticky='W', padx=5,
                          pady=2)

    inFileLbl = tkinter.Label(step_two,
                              text='Select the medical data... '
                                   'The column must match'
                              )
    inFileLbl.grid(row=6, sticky='E', column=0, padx=10, pady=2)

    for x in range(int(entry_text)):
        numM = 7
        getFilesM = check_files(MEDICAL_URL)
        if len(getFilesM) != 0:
            medical_year_store.append('medical_year' + str(x))
            medical_year_store[x] = tk.StringVar(root)
            comboLab = tkinter.OptionMenu(step_two,
                                          medical_year_store[x], *getFilesM)
            comboLab.grid(row=numM, column=x, sticky='EW', ipadx=10)
        else:

            medical_year_store.append('medical_year' + str(x))
            medical_year_store[x] = tk.StringVar(root)
            comboLab = tkinter.OptionMenu(step_two,
                                          medical_year_store[x], value='')
            comboLab.grid(row=numM, column=x, sticky='W', padx=5,
                          pady=2)

    inFileBtn = tkinter.Button(step_two, text='Combine ...',
                               command=lambda: more_data(
                                   laboratory_year_store,
                                   demographic_year_store,
                                   medical_year_store,
                                   step_two,
                                   root,
                                   RETURN,
                                   entry_text,
                               ))
    inFileBtn.grid(row=8, column=1, sticky='W', padx=5, pady=2)


def more_data(
        laboratory_year_store,
        demographic_year_store,
        medical_year_store,
        step_two,
        root,
        string_condition,
        entry_text,
):
    """
    Combining the files and storing the files.
    :param laboratory_year_store: Laboratory file
    :param demographic_year_store: Demographic file
    :param medical_year_store: Medical file
    :param step_two: Position
    :param root: Original window
    :param string_condition: Type of condition
    :param entry_text: Number of files to combine
    :return:
    """

    checkVariable = '1'
    returnedBool = False

    for x in range(int(entry_text)):
        returnedBool = check_next(
            str(laboratory_year_store[x].get()),
            str(demographic_year_store[x].get()),
            checkVariable,
            medical_year_store[x],
            step_two,
            root,
            string_condition,
            1
        )

    loop_num = int(entry_text)
    df_new = None
    laboratory_name = []
    if returnedBool:
        for x in range(loop_num):
            result_laboratory_first = ''.join(i for i in
                                              laboratory_year_store[x].get()
                                              if not i.isdigit())

            size = len(result_laboratory_first)
            result_laboratory_first = result_laboratory_first[:size - 6]

            laboratory_date_first = ''.join(i for i in
                                            laboratory_year_store[x].get()
                                            if i.isdigit())
            laboratory_date_first = insert_dash(laboratory_date_first,
                                                4)
            laboratory_date_first = result_laboratory_first + '_' \
                                    + laboratory_date_first

            if laboratory_date_first not in laboratory_name:
                laboratory_x = LABORATORY_URL + '\\' \
                               + str(laboratory_year_store[x].get())

                if laboratory_x.endswith('.XPT'):
                    laboratory_x = pd.read_sas(laboratory_x)
                else:
                    laboratory_x = pd.read_csv(laboratory_x)

                laboratory_date_first = laboratory_date_first + '_'
                laboratory_name.append(laboratory_date_first)

                df_new = laboratory_x.append(laboratory_x,
                                             ignore_index=True)

    file_name = ''
    for x in laboratory_name:
        file_name += x

    file_name = file_name + '.csv'
    lab_url = LABORATORY_URL + '\\' + file_name

    df = pd.DataFrame(df_new, columns=df_new.columns)
    df.to_csv(lab_url, index=False, header=True)

    df_new = None
    demographic_name = []
    if returnedBool:
        for x in range(loop_num):

            result_demographics_first = ''.join(i for i in
                                                demographic_year_store[x].get()
                                                if not i.isdigit())
            size = len(result_demographics_first)
            result_demographics_first = result_demographics_first[:size
                                                                   - 6]

            demographics_date_first = ''.join(i for i in
                                              demographic_year_store[x].get()
                                              if i.isdigit())
            demographics_date_first = \
                insert_dash(demographics_date_first, 4)
            demographics_date_first = result_demographics_first + '_' \
                                      + demographics_date_first

            if demographics_date_first not in demographic_name:
                demographics_x = DEMOGRAPHIC_URL + '\\' \
                                 + str(demographic_year_store[x].get())

                if demographics_x.endswith('.XPT'):
                    demographics_x = pd.read_sas(demographics_x)
                else:
                    demographics_x = pd.read_csv(demographics_x)

                demographics_date_first = demographics_date_first \
                                          + '_'
                demographic_name.append(demographics_date_first)

                df_new = demographics_x.append(demographics_x,
                                               ignore_index=True)

    file_name = ''
    for x in demographic_name:
        file_name += x

    file_name = file_name + '.csv'
    lab_url = DEMOGRAPHIC_URL + '\\' + file_name

    df = pd.DataFrame(df_new, columns=df_new.columns)
    df.to_csv(lab_url, index=False, header=True)

    df_new = None
    medical_name = []
    if returnedBool:
        for x in range(loop_num):
            result_medical_first = ''.join(i for i in
                                           medical_year_store[x].get()
                                           if not i.isdigit())
            size = len(result_medical_first)
            result_medical_first = result_medical_first[:size - 6]

            medical_date_first = ''.join(i for i in
                                         medical_year_store[x].get()
                                         if i.isdigit())
            medical_date_first = insert_dash(medical_date_first, 4)
            medical_date_first = result_medical_first + '_' \
                                 + medical_date_first

            if medical_date_first not in medical_name:
                medical_x = MEDICAL_URL + '\\' \
                            + str(medical_year_store[x].get())
                if medical_x.endswith('.XPT'):
                    medical_x = pd.read_sas(medical_x)
                else:
                    medical_x = pd.read_csv(medical_x)

                medical_date_first = medical_date_first + '_'
                medical_name.append(medical_date_first)

                df_new = medical_x.append(medical_x,
                                          ignore_index=True)

    file_name = ''
    for x in medical_name:
        file_name += x

    file_name = file_name + '.csv'
    lab_url = MEDICAL_URL + '\\' + file_name

    df = pd.DataFrame(df_new, columns=df_new.columns)
    df.to_csv(lab_url, index=False, header=True)

def standard(root):
    add_data_to_gui(root)
    second_multiple_options(root)
    third_position_in_window(root)




def split(root):
    select = tkinter.LabelFrame(root,
                                  text='Choose option...')
    select.grid(
        row=2,
        column=0,
        sticky='W',
        padx=5,
        pady=5,
        ipadx=5,
        ipady=5,
    )

    inFileBtn = tkinter.Button(select, text='NHANES Data ...',
                               command=lambda: \
                                   standard(root))
    inFileBtn.grid(row=0, column=2, sticky='W', padx=5, pady=2)

    inFileBtn = tkinter.Button(select, text='Different Data...',
                               command=lambda: break_up(root))
    inFileBtn.grid(row=0, column=3, sticky='W', padx=5, pady=2)

def break_up(root):
    different(root)
    get_labels(root)

def different(root):
    step_one = tkinter.LabelFrame(root, text=' 1. Add new data')
    step_one.grid(
        row=0,
        columnspan=7,
        sticky='WE',
        padx=5,
        pady=5,
        ipadx=5,
        ipady=5,
    )
    inFileLbl = tkinter.Label(step_one,
                              text='Select File')
    inFileLbl.grid(row=0, sticky='E', column=0, padx=5, pady=2)

    entry_text = tk.StringVar()
    inFileTxt = tkinter.Entry(step_one, textvariable=entry_text)
    inFileTxt.grid(row=0, column=1, sticky='WE', pady=3, ipadx=150)

    inFileBtn = tkinter.Button(step_one, text='Browse ...',
                               command=lambda: \
                                   entry_text.set(browse_files()))
    inFileBtn.grid(row=0, column=8, sticky='W', padx=5, pady=2)


    outFileBtn = tkinter.Button(step_one, text='save ...',
                                command=lambda: load(
                                    entry_text.get()
                                ))

    outFileBtn.grid(row=5, column=1, sticky='WE', padx=5, pady=2)


def get_labels(root):
    step_two = tkinter.LabelFrame(root,
                                 text=' 2. Select Label...'
                                 )
    step_two.grid(
        row=2,
        column=0,
        sticky='W',
        padx=5,
        pady=5,
        ipadx=5,
        ipady=5,
    )

    inFileLbl = tkinter.Label(step_two, text='Select the data'
                              )
    inFileLbl.grid(row=0, sticky='E', column=0, padx=5, pady=2)

    # Instead of cbc needs to read the files.

    get_files_l = check_files(FILE_URL)
    if len(get_files_l) != 0:
        chosen_file = tk.StringVar(root)
        # chosen_laboratory_file.set(get_files_l[0])
        comboLab = tkinter.OptionMenu(step_two, chosen_file,
                                      *get_files_l)
        comboLab.grid(row=0, column=1, sticky='EW', ipadx=10)
    else:

        chosen_file = tk.StringVar(root)
        comboLab = tkinter.OptionMenu(step_two, chosen_file,
                                      value='')
        comboLab.grid(row=0, column=1, sticky='W', padx=5, pady=2)

    inFileBtnMed = tkinter.Button(step_two, text='Next...',
                                  command=lambda: check_next_all(
                                      chosen_file, root, step_two
                                  ))
    inFileBtnMed.grid(row=0, column=2, sticky='W', padx=5, pady=2)


def check_next_all(file,root,step_two):
    file_name = file.get()
    file_columns = get_file_columns(file_name)

    if len(file_columns) != 0:
        chosen_columns = tk.StringVar(root)
        comboLab = tkinter.OptionMenu(step_two,
                                      chosen_columns, *file_columns)
        comboLab.grid(row=0, column=3, sticky='EW')

        enter_text = tk.StringVar(value='Code value (Target)')
        inFileTxt = tkinter.Entry(step_two,
                                  textvariable=enter_text)
        inFileTxt.grid(row=0, column=4, sticky='WE', pady=3, ipadx=10)

        inFileBtnMed = tkinter.Button(step_two, text='Extract...',
                                      command=lambda: extract_label(
                                          file_name, chosen_columns, enter_text,root
                                      ))
        inFileBtnMed.grid(row=0, column=5, sticky='W', padx=5, pady=2)

def extract_label(file, chosen_columns, enter_text, root):
    chosen_columns = chosen_columns.get()
    enter_text = enter_text.get()
    boolean_neg = 0
    boolean_post = 1
    if enter_text.strip().isdigit():
        enter_text = int(enter_text)
    else:
        boolean_neg = False
        boolean_post = True

    str_url = FILE_URL + '\\' + file
    read_files = pd.read_excel(str_url, index_col=0)

    labels = read_files
    features = read_files

    data = read_files[[chosen_columns]]. \
        applymap(lambda x: (boolean_neg if x != enter_text else boolean_post))
    labels[chosen_columns] = data[chosen_columns]

    features = features.drop(chosen_columns, axis=1)
    labels = labels[chosen_columns]
    part_three_extended(root, features, labels)


def part_three_extended(root, features, labels):
    step_three = tkinter.LabelFrame(root,
                                    text=' 3. Using prediction using testing data.'
                                    )
    step_three.grid(
        row=3,
        columnspan=7,
        sticky='W',
        padx=5,
        pady=5,
        ipadx=5,
        ipady=5,
    )

    inFileLbl = tkinter.Label(step_three,
                              text='Learn')
    inFileLbl.grid(row=0, sticky='E', column=0, padx=5, pady=2)

    inFileBtnMed = tkinter.Button(step_three, text='SVM',
                                  command=lambda: field_svm_entries_part_two(
                                      features, labels
                                  ))

    inFileBtnMed.grid(row=0, column=5, sticky='W', padx=5, pady=2)

    inFileBtnMed = tkinter.Button(step_three, text='RFC',
                                  command=lambda: random_forest_classifier_part_two(
                                      features, labels
                                  ))

    inFileBtnMed.grid(row=0, column=6, sticky='W', padx=5, pady=2)



def random_forest_classifier_part_two(
      features, labels
):
    """
    Ensure correct steps taken when performing RFC
    :param laboratory: Laboratory data
    :param demographic: Demographic data
    :param demographic_bool: Either include demographic data or not
    :param medical: Medical data
    :param health_condition: Chosen health condition
    :param code_value: Value for health condition
    :return:
    """

    # Scale
    x = scale(features)

    # SMOTE
    (X_train, X_test, y_train, y_test) = smote(x, labels)
    random_forest_classifier_print_part_two(
        X_train,
        X_test,
        y_train,
        y_test,
    )

def random_forest_classifier_print_part_two(
        X_train,
        X_test,
        y_train,
        y_test,
):
    """
    Applying random_forest_classifier.
    :param X_train: Training data
    :param X_test: Testing data
    :param y_train: Training labels
    :param y_test: Testing labels
    :param laboratory: Laboratory data
    :param demographic: Demographic data
    :param demographic_bool: Bool to include demographic data
    :param medical: Medical data
    :param health_condition: Chosen health condition
    :param code_value: Inputted health code value
    :param info: String explaining which button was pressed
    """

    training_type = RandomForestClassifier(random_state=42)
    training_type.fit(X_train, y_train)
    preds = training_type.predict(X_test)

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


def field_svm_entries_part_two(features, labels):
    """
    Ensure correct steps taken when performing svm.
    :param laboratory: Laboratory data
    :param demographic: Demographic data
    :param demographic_bool: Either include demographic data or not
    :param medical: Medical data
    :param health_condition: Chosen health condition
    :param code_value: Value for health condition
    """

    # Scale
    x = scale(features)

    # SMOTE
    (X_train, X_test, y_train, y_test) = smote(x, labels)

    #X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=1, train_size=0.80)

    # SVM
    (X_test, training_type, X_train) = run_svm_part_two(
        X_train,
        X_test,
        y_train,
    )

    scores = cross_val_score(training_type, X_test, y_test, cv=10,
                             scoring='accuracy')

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

    shape = X_train.shape
    number_of_col = shape[1] - 2

    # Print
    two_dim_graph_train(X_train, y_train, training_type, number_of_col)

def run_svm_part_two(
        X_train,
        X_test,
        y_train,
):
    """
    Performs support vector machine.
    :param X_train: Training data
    :param X_test: Testing data
    :param y_train: Training label
    :param y_test: Testing label
    :param laboratory: Laboratory data
    :param demographic: Demographic data
    :param demographic_bool: Either to include demographic data or not
    :param medical: Medical data
    :param health_condition: Chosen health condition
    :param code_value: Value for health condition
    :param info: Type of button which was pressed
    :return: Testing data, support vector machine data, Training data
    """

    normaliser = StandardScaler()
    normaliser.fit(X_train)
    X_train = normaliser.transform(X_train)
    X_test = normaliser.transform(X_test)

    training_type = SVC(kernel='rbf', gamma='auto', C=1.0,
                        decision_function_shape='ovr').fit(X_train,
                                                           y_train)

    return (X_test, training_type, X_train)


def get_file_columns(file):
    """
    Gets the columns of the medical condition
    :param medical_condition:
    :return: Columns of the medical condition
    """
    if len(file) != 0:
        str_url = FILE_URL + '\\' + file
        if str_url.endswith('.XPT'):
            read_files = pd.read_sas(str_url)
            return read_files.columns
        elif str_url.endswith('.xlsx'):
            read_files = pd.read_excel(str_url, index_col=0)
            return read_files.columns
        else:
            read_files = pd.read_csv(str_url)
            return read_files.columns


def load(file):
    if len(file) != 0:
        file_name = os.path.basename(file)
        url = FILE_URL + '/' + file_name
        exists = os.path.exists(url)
        if exists:
            showinfo(title='Error', message='The file NAME: '
                                            + file_name + ' exists')
        else:
            shutil.copyfile(file, file_name)
            shutil.move(file_name, FILE_URL)
            showinfo(title='Added', message='The file ' + file_name
                                            + ' was added...')










def run_code():
    """
    Running all functions to populate the window.
    """
    root = tk.Tk()
    root.title('Menu')
    split(root)
    root.mainloop()


if __name__ == '__main__':
    run_code()
