from scipy.stats import uniform
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from model.Constants import VISUALIZATION_SAVE_DIRECTORY
from model.DataFrameHandler import DataFrameHandler
from model.Logger import logger
from imblearn.over_sampling import SMOTE
from model.PrePocessor import PreProcessor
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def create_model():
    logger.info('In SVMModel | create_model started')
    C = np.logspace(0, 10, 100)
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    gamma = ['scale', 'auto']
    hyperparameters = dict(C=C, kernel=kernel, gamma=gamma)
    svc_classifier = SVC(random_state=42, probability=True, cache_size=2000, tol=0.1)
    rand = RandomizedSearchCV(svc_classifier, hyperparameters, random_state=42, n_iter=100, cv=10, n_jobs=-1,
                              scoring=['recall', 'accuracy'], refit='accuracy')
    logger.info('In SVMModel | create_model finished')
    return svc_classifier, rand


def run_model(X, y, model_name, perform_smote=False):
    logger.info('In SVMModel | run_model started')
    y = y.apply(lambda x: 1 if x == 'yes' else 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if perform_smote:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    std_model, randomize_model = create_model()
    print('Running Standard SVM Classification on ' + model_name)
    std_model.fit(X_train, y_train)
    y_pred = std_model.predict(X_test)
    y_pred_train = std_model.predict(X_train)
    print('Confusion Martix for Standard SVM Classification model on ' + model_name)
    print(confusion_matrix(y_test, y_pred))
    print("Training Accuracy for Standard SVM Classification Model on " + model_name)
    print(metrics.accuracy_score(y_train, y_pred_train))
    print("Training Log loss for Standard SVM Classification Model on " + model_name)
    print(metrics.log_loss(y_train, y_pred_train))
    print("Accuracy for Standard SVM Classification Model on " + model_name)
    print(metrics.accuracy_score(y_test, y_pred))
    print("Log loss for Standard SVM Classification Model on " + model_name)
    print(metrics.log_loss(y_test, y_pred))
    print("Classification Report for Standard SVM Classification Model on " + model_name)
    print(classification_report(y_test, y_pred))
    probs = std_model.predict_proba(X_test)
    probs = probs[:, 1]
    auc = roc_auc_score(y_test, probs)
    print('AUC: %.2f' % auc)
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid()
    plt.ion()
    plt.show()
    plt.savefig(os.path.join(VISUALIZATION_SAVE_DIRECTORY, 'Std_SVM_' + model_name + '_roc_auc'))
    plt.pause(1)
    plt.close()
    # print('Running RandomizedSearch SVM Classification on ' + model_name)
    # randomize_best_model = randomize_model.fit(X_train, y_train)
    # print('Best C:', randomize_best_model.best_estimator_.get_params()['C'])
    # print('Best kernel:', randomize_best_model.best_estimator_.get_params()['kernel'])
    # print('Best gamma:', randomize_best_model.best_estimator_.get_params()['gamma'])
    # y_pred = randomize_best_model.predict(X_test)
    # y_pred_train = randomize_best_model.predict(X_train)
    # print('Confusion Martix for Best Randomized SVM Classification model on ' + model_name)
    # print(confusion_matrix(y_test, y_pred))
    # print("Training Accuracy for Best Randomized SVM Classification Model on " + model_name)
    # print(metrics.accuracy_score(y_train, y_pred_train))
    # print("Training Log Loss for Best Randomized SVM Classification Model on " + model_name)
    # print(metrics.log_loss(y_train, y_pred_train))
    # print("Accuracy for Best Randomized SVM Classification Model on " + model_name)
    # print(metrics.accuracy_score(y_test, y_pred))
    # print("Log Loss for Best Randomized SVM Classification Model on " + model_name)
    # print(metrics.log_loss(y_test, y_pred))
    # print("Classification Report for Randomized SVM Classification Model on " + model_name)
    # print(classification_report(y_test, y_pred))
    # probs = randomize_best_model.predict_proba(X_test)
    # probs = probs[:, 1]
    # auc = roc_auc_score(y_test, probs)
    # print('AUC: %.2f' % auc)
    # fpr, tpr, thresholds = roc_curve(y_test, probs)
    # plt.plot(fpr, tpr, color='orange', label='ROC')
    # plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend()
    # plt.grid()
    # plt.ion()
    # plt.show()
    # plt.savefig(os.path.join(VISUALIZATION_SAVE_DIRECTORY, 'Randomized_SVM_' + model_name + '_roc_auc'))
    # plt.pause(1)
    # plt.close()
    logger.info('In SVMModel | run_model finished')


def main():
    pre_process = PreProcessor()
    df = pre_process.load_data()
    df_handler = DataFrameHandler(df)
    ######################################################################################################################
    #  Model 1: Dummy Encoding for Categorical Variables-> MinMax Scaling for Numerical Variables                        #
    ######################################################################################################################
    dummies_df = df_handler.get_dummies_data()
    logger.debug(dummies_df.head())
    scaled_df = df_handler.get_scaled_data(df=dummies_df)
    attribute_target_split_result = df_handler.split_attribute_and_target(df=scaled_df)
    X = attribute_target_split_result['attributes']
    y = attribute_target_split_result['target']
    print(X.head())
    run_model(X, y, 'DummyEncoded_MinMaxScaling')
    ######################################################################################################################
    #  Model 2: Dummy Encoding for Categorical Variables-> MinMax Scaling for Numerical Variables-> SMOTE on training Set#
    ######################################################################################################################
    X = attribute_target_split_result['attributes']
    y = attribute_target_split_result['target']
    print(X.head())
    run_model(X, y, 'DummyEncoded_MinMaxScaling_SMOTE', perform_smote=True)
    ######################################################################################################################
    #  Model 3: Dummy Encoding for Categorical Variables-> SMOTE on training Set                                         #
    ######################################################################################################################
    attribute_target_split_result = df_handler.split_attribute_and_target(df=dummies_df)
    X = attribute_target_split_result['attributes']
    y = attribute_target_split_result['target']
    print(X.head())
    run_model(X, y, 'DummyEncoded_SMOTE', perform_smote=True)
    ######################################################################################################################
    #  Model 4: Dummy Encoding for Categorical Variables                                                                 #
    ######################################################################################################################
    attribute_target_split_result = df_handler.split_attribute_and_target(df=dummies_df)
    X = attribute_target_split_result['attributes']
    y = attribute_target_split_result['target']
    print(X.head())
    run_model(X, y, 'DummyEncoded', perform_smote=False)
    ######################################################################################################################
    #  Model 5: Dummy Encoding for Categorical Variables-> Binning for Numerical Variables                               #
    ######################################################################################################################
    dummies_df = df_handler.get_dummies_data()
    logger.debug(dummies_df.head())
    binned_df = df_handler.get_binned_data(df=dummies_df)
    attribute_target_split_result = df_handler.split_attribute_and_target(df=binned_df)
    X = attribute_target_split_result['attributes']
    y = attribute_target_split_result['target']
    print(X.head())
    run_model(X, y, 'DummyEncoded_Binning', perform_smote=False)
    ######################################################################################################################
    #  Model 6: Dummy Encoding for Categorical Variables-> Binning for Numerical Variables-> SMOTE on training Set       #
    ######################################################################################################################
    X = attribute_target_split_result['attributes']
    y = attribute_target_split_result['target']
    print(X.head())
    run_model(X, y, 'DummyEncoded_Binning_SMOTE', perform_smote=True)
    ######################################################################################################################
    #  Model 7: Label Encoding for Categorical Variables-> MinMax Scaling for Numerical Variables                        #
    ######################################################################################################################
    label_df = df_handler.get_label_encoded_data()
    scaled_df = df_handler.get_scaled_data(df=label_df)
    attribute_target_split_result = df_handler.split_attribute_and_target(df=scaled_df)
    X = attribute_target_split_result['attributes']
    y = attribute_target_split_result['target']
    print(X.head())
    run_model(X, y, 'LabeEncoded_MinMaxScaling', perform_smote=False)
    ######################################################################################################################
    #  Model 8: Label Encoding for Categorical Variables-> MinMax Scaling for Numerical Variables-> SMOTE on training Set#
    ######################################################################################################################
    X = attribute_target_split_result['attributes']
    y = attribute_target_split_result['target']
    print(X.head())
    run_model(X, y, 'LabelEncoded_MinMaxScaling_SMOTE', perform_smote=True)
    ######################################################################################################################
    #  Model 9: Label Encoding for Categorical Variables-> SMOTE on training Set                                         #
    ######################################################################################################################
    attribute_target_split_result = df_handler.split_attribute_and_target(df=label_df)
    X = attribute_target_split_result['attributes']
    y = attribute_target_split_result['target']
    print(X.head())
    run_model(X, y, 'LabelEncoded_SMOTE', perform_smote=True)
    ######################################################################################################################
    #  Model 10: Label Encoding for Categorical Variables                                                                 #
    ######################################################################################################################
    attribute_target_split_result = df_handler.split_attribute_and_target(df=label_df)
    X = attribute_target_split_result['attributes']
    y = attribute_target_split_result['target']
    print(X.head())
    run_model(X, y, 'LabelEncoded', perform_smote=False)
    ######################################################################################################################
    #  Model 11: Label Encoding for Categorical Variables-> Binning for Numerical Variables                               #
    ######################################################################################################################
    label_df = df_handler.get_label_encoded_data()
    binned_df = df_handler.get_binned_data(df=label_df)
    attribute_target_split_result = df_handler.split_attribute_and_target(df=binned_df)
    X = attribute_target_split_result['attributes']
    y = attribute_target_split_result['target']
    print(X.head())
    run_model(X, y, 'LabelEncoded_Binning', perform_smote=False)
    ######################################################################################################################
    #  Model 12: Label Encoding for Categorical Variables-> Binning for Numerical Variables-> SMOTE on training Set       #
    ######################################################################################################################
    X = attribute_target_split_result['attributes']
    y = attribute_target_split_result['target']
    print(X.head())
    run_model(X, y, 'LabelEncoded_Binning_SMOTE', perform_smote=True)


if __name__ == '__main__':
    main()
