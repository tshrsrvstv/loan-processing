from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn import metrics
from xgboost import XGBClassifier

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
    logger.info('In XGBoostModel | create_model started')
    hyperparameters = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    }
    xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                        silent=True, nthread=1)
    rand = RandomizedSearchCV(xgb, param_distributions=hyperparameters, n_iter=10, n_jobs=-1, random_state=42, scoring=['recall', 'accuracy', 'neg_log_loss', 'f1', 'roc_auc'], refit='accuracy')
    logger.info('In XGBoostModel | create_model finished')
    return xgb, rand


def run_model(X, y, model_name, perform_smote=False):
    logger.info('In XGBoostModel | run_model started')
    y = y.apply(lambda x: 1 if x == 'yes' else 0)
    X.columns = [colname.replace(',','#').replace('(', '$').replace(']', '@') for colname in X.columns]
    print(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if perform_smote:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    std_model, randomize_model = create_model()
    print('Running Standard XGBoost Classification on ' + model_name)
    std_model.fit(X_train, y_train)
    y_pred = std_model.predict(X_test)
    y_pred_train = std_model.predict(X_train)
    print('Confusion Martix for Standard XGBoost Classification model on ' + model_name)
    print(confusion_matrix(y_test, y_pred))
    print("Training Accuracy for Standard XGBoost Classification Model on " + model_name)
    print(metrics.accuracy_score(y_train, y_pred_train))
    print("Training Log loss for Standard XGBoost Classification Model on " + model_name)
    print(metrics.log_loss(y_train, y_pred_train))
    print("Accuracy for Standard XGBoost Classification Model on " + model_name)
    print(metrics.accuracy_score(y_test, y_pred))
    print("Log loss for Standard XGBoost Classification Model on " + model_name)
    print(metrics.log_loss(y_test, y_pred))
    print("Classification Report for Standard XGBoost Classification Model on " + model_name)
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
    plt.savefig(os.path.join(VISUALIZATION_SAVE_DIRECTORY, 'Std_XGB_' + model_name + '_roc_auc'))
    plt.pause(1)
    plt.close()
    print('Running RandomizedSearch XGBoost Classification on ' + model_name)
    randomize_best_model = randomize_model.fit(X_train, y_train)
    print('Best min_child_weight:', randomize_best_model.best_estimator_.get_params()['min_child_weight'])
    print('Best gamma:', randomize_best_model.best_estimator_.get_params()['gamma'])
    print('Best subsample:', randomize_best_model.best_estimator_.get_params()['subsample'])
    print('Best colsample_bytree:', randomize_best_model.best_estimator_.get_params()['colsample_bytree'])
    print('Best max_depth:', randomize_best_model.best_estimator_.get_params()['max_depth'])
    y_pred = randomize_best_model.predict(X_test)
    y_pred_train = randomize_best_model.predict(X_train)
    print('Confusion Martix for Best Randomized XGBoost Classification model on ' + model_name)
    print(confusion_matrix(y_test, y_pred))
    print("Training Accuracy for Best Randomized XGBoost Classification Model on " + model_name)
    print(metrics.accuracy_score(y_train, y_pred_train))
    print("Training Log Loss for Best Randomized XGBoost Classification Model on " + model_name)
    print(metrics.log_loss(y_train, y_pred_train))
    print("Accuracy for Best Randomized XGBoost Classification Model on " + model_name)
    print(metrics.accuracy_score(y_test, y_pred))
    print("Log Loss for Best Randomized XGBoost Classification Model on " + model_name)
    print(metrics.log_loss(y_test, y_pred))
    print("Classification Report for Randomized XGBoost Classification Model on " + model_name)
    print(classification_report(y_test, y_pred))
    probs = randomize_best_model.predict_proba(X_test)
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
    plt.savefig(os.path.join(VISUALIZATION_SAVE_DIRECTORY, 'Randomized_XGB_' + model_name + '_roc_auc'))
    plt.pause(1)
    plt.close()
    logger.info('In XGBoostModel | run_model finished')


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
