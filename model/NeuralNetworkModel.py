import os
import imblearn
from docutils.nodes import label
from imblearn.over_sampling import SMOTE
from keras import metrics
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from keras.metrics import categorical_accuracy
from model.Constants import VISUALIZATION_SAVE_DIRECTORY
from model.DataFrameHandler import DataFrameHandler
from model.Logger import logger
from model.PrePocessor import PreProcessor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_model(colnum):
    logger.info('In NeuralNetworkModel | create_model Started')
    model = Sequential()
    model.add(Dense(150, activation='relu', input_shape=(colnum,)))
    model.add(Dense(100, activation='relu', input_shape=(colnum,)))
    model.add(Dense(50, activation='relu', input_shape=(colnum,)))
    model.add(Dense(2, activation='softmax'))
    print(model.summary())
    logger.info('In NeuralNetworkModel | create_model finished')
    return model


def run_model(X, y, model_name, df_handler, perform_smote=False):
    logger.info('In NeuralNetworkModel | run_model Started for ' + model_name + ' model.')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if perform_smote:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        y_train = pd.get_dummies(y_train, prefix=df_handler.target_col)
        y_test = pd.get_dummies(y_test, prefix=df_handler.target_col)
    colnum = X_train.shape[1]
    model = create_model(colnum)
    model.compile(optimizer='adamax',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=45)
    score = model.evaluate(X_test, y_test, verbose=0)
    print(model_name + ' Model Testing Score : ' + str(score))
    plt.plot(np.arange(0, len(history.history['loss'])), history.history['loss'])
    plt.title("Loss")
    plt.grid()
    plt.ion()
    plt.show()
    plt.savefig(os.path.join(VISUALIZATION_SAVE_DIRECTORY, model_name + '_losses'))
    plt.pause(1)
    plt.close()
    plt.plot(np.arange(0, len(history.history['accuracy'])), history.history['accuracy'])
    plt.title("Accuracy")
    plt.grid()
    plt.ion()
    plt.show()
    plt.savefig(os.path.join(VISUALIZATION_SAVE_DIRECTORY, model_name + '_accuracy'))
    plt.pause(1)
    plt.close()
    y_pred = model.predict(X_test)
    logger.debug(y_pred)
    rounded_predictions = model.predict_classes(X_test)
    logger.debug(rounded_predictions)
    print('Confusion Matrix: ')
    print(confusion_matrix(y_test['target_yes'], rounded_predictions))
    print('Classification Report: ')
    print(classification_report(y_test['target_yes'], rounded_predictions))
    probs = model.predict_proba(X_test)
    probs = probs[:, 1]
    auc = roc_auc_score(y_test['target_yes'], probs)
    print('AUC: %.2f' % auc)
    fpr, tpr, thresholds = roc_curve(y_test['target_yes'], probs)
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid()
    plt.ion()
    plt.show()
    plt.savefig(os.path.join(VISUALIZATION_SAVE_DIRECTORY, model_name + '_roc_auc'))
    plt.pause(1)
    plt.close()
    logger.info('In NeuralNetworkModel | run_model finished')


def main():
    logger.info('In NeuralNetworkModel | Main Started')
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
    y = pd.get_dummies(y, prefix=df_handler.target_col)
    print(X.head())
    run_model(X, y, 'DummyEncoded_MinMaxScaling', df_handler)
    ######################################################################################################################
    #  Model 2: Dummy Encoding for Categorical Variables-> MinMax Scaling for Numerical Variables-> SMOTE on training Set#
    ######################################################################################################################
    X = attribute_target_split_result['attributes']
    y = attribute_target_split_result['target']
    print(X.head())
    run_model(X, y, 'DummyEncoded_MinMaxScaling_SMOTE', df_handler, perform_smote=True)
    ######################################################################################################################
    #  Model 3: Dummy Encoding for Categorical Variables-> SMOTE on training Set                                         #
    ######################################################################################################################
    attribute_target_split_result = df_handler.split_attribute_and_target(df=dummies_df)
    X = attribute_target_split_result['attributes']
    y = attribute_target_split_result['target']
    print(X.head())
    run_model(X, y, 'DummyEncoded_SMOTE', df_handler, perform_smote=True)
    ######################################################################################################################
    #  Model 4: Dummy Encoding for Categorical Variables                                                                 #
    ######################################################################################################################
    attribute_target_split_result = df_handler.split_attribute_and_target(df=dummies_df)
    X = attribute_target_split_result['attributes']
    y = attribute_target_split_result['target']
    y = pd.get_dummies(y, prefix=df_handler.target_col)
    print(X.head())
    run_model(X, y, 'DummyEncoded', df_handler, perform_smote=False)
    ######################################################################################################################
    #  Model 5: Dummy Encoding for Categorical Variables-> Binning for Numerical Variables                               #
    ######################################################################################################################
    dummies_df = df_handler.get_dummies_data()
    logger.debug(dummies_df.head())
    binned_df = df_handler.get_binned_data(df=dummies_df)
    attribute_target_split_result = df_handler.split_attribute_and_target(df=binned_df)
    X = attribute_target_split_result['attributes']
    y = attribute_target_split_result['target']
    y = pd.get_dummies(y, prefix=df_handler.target_col)
    print(X.head())
    run_model(X, y, 'DummyEncoded_Binning', df_handler, perform_smote=False)
    ######################################################################################################################
    #  Model 6: Dummy Encoding for Categorical Variables-> Binning for Numerical Variables-> SMOTE on training Set       #
    ######################################################################################################################
    X = attribute_target_split_result['attributes']
    y = attribute_target_split_result['target']
    print(X.head())
    run_model(X, y, 'DummyEncoded_Binning_SMOTE', df_handler, perform_smote=True)
    ######################################################################################################################
    #  Model 7: Label Encoding for Categorical Variables-> MinMax Scaling for Numerical Variables                        #
    ######################################################################################################################
    label_df = df_handler.get_label_encoded_data()
    scaled_df = df_handler.get_scaled_data(df=label_df)
    attribute_target_split_result = df_handler.split_attribute_and_target(df=scaled_df)
    X = attribute_target_split_result['attributes']
    y = attribute_target_split_result['target']
    y = pd.get_dummies(y, prefix=df_handler.target_col)
    print(X.head())
    run_model(X, y, 'LabeEncoded_MinMaxScaling', df_handler, perform_smote=False)
    ######################################################################################################################
    #  Model 8: Label Encoding for Categorical Variables-> MinMax Scaling for Numerical Variables-> SMOTE on training Set#
    ######################################################################################################################
    X = attribute_target_split_result['attributes']
    y = attribute_target_split_result['target']
    print(X.head())
    run_model(X, y, 'LabelEncoded_MinMaxScaling_SMOTE', df_handler, perform_smote=True)
    ######################################################################################################################
    #  Model 9: Label Encoding for Categorical Variables-> SMOTE on training Set                                         #
    ######################################################################################################################
    attribute_target_split_result = df_handler.split_attribute_and_target(df=label_df)
    X = attribute_target_split_result['attributes']
    y = attribute_target_split_result['target']
    print(X.head())
    run_model(X, y, 'LabelEncoded_SMOTE', df_handler, perform_smote=True)
    ######################################################################################################################
    #  Model 10: Label Encoding for Categorical Variables                                                                 #
    ######################################################################################################################
    attribute_target_split_result = df_handler.split_attribute_and_target(df=label_df)
    X = attribute_target_split_result['attributes']
    y = attribute_target_split_result['target']
    y = pd.get_dummies(y, prefix=df_handler.target_col)
    print(X.head())
    run_model(X, y, 'LabelEncoded', df_handler, perform_smote=False)
    ######################################################################################################################
    #  Model 11: Label Encoding for Categorical Variables-> Binning for Numerical Variables                               #
    ######################################################################################################################
    label_df = df_handler.get_label_encoded_data()
    binned_df = df_handler.get_binned_data(df=label_df)
    attribute_target_split_result = df_handler.split_attribute_and_target(df=binned_df)
    X = attribute_target_split_result['attributes']
    y = attribute_target_split_result['target']
    y = pd.get_dummies(y, prefix=df_handler.target_col)
    print(X.head())
    run_model(X, y, 'LabelEncoded_Binning', df_handler, perform_smote=False)
    ######################################################################################################################
    #  Model 12: Label Encoding for Categorical Variables-> Binning for Numerical Variables-> SMOTE on training Set       #
    ######################################################################################################################
    X = attribute_target_split_result['attributes']
    y = attribute_target_split_result['target']
    print(X.head())
    run_model(X, y, 'LabelEncoded_Binning_SMOTE', df_handler, perform_smote=True)
    logger.info('In NeuralNetworkModel | Main Finished')


if __name__ == '__main__':
    main()
