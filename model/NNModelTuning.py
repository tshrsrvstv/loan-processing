import os
import imblearn
from docutils.nodes import label
from imblearn.over_sampling import SMOTE
from keras import metrics
from keras.optimizers import Adadelta
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
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


def create_model(colnum, neurons=[150, 100, 50]):
    model = Sequential()
    model.add(Dense(neurons[0], activation='relu', input_shape=(colnum,)))
    model.add(Dropout(0.2))
    model.add(Dense(neurons[1], activation='relu', input_shape=(colnum,)))
    model.add(Dropout(0.2))
    model.add(Dense(neurons[2], activation='relu', input_shape=(colnum,)))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    print(model.summary())
    model.compile(optimizer='adamax',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def run_model(X, y, model_name, df_handler, perform_smote=False):
    logger.info('In NNModelTuning | run_model Started for ' + model_name + ' model.')
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if perform_smote:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        y_train = pd.get_dummies(y_train, prefix=df_handler.target_col)
        y_test = pd.get_dummies(y_test, prefix=df_handler.target_col)
    colnum = X_train.shape[1]
    model = KerasClassifier(build_fn=create_model, colnum=colnum, epochs=45)
    neurons = ((50, 50, 50), (100, 50, 50), (150, 50, 50), (200, 50, 50), (250, 50, 50),
               (50, 100, 50), (100, 100, 50), (150, 100, 50), (200, 100, 50), (250, 100, 50),
               (50, 150, 50), (100, 150, 50), (150, 150, 50), (200, 150, 50), (250, 150, 50))
    param_grid = dict(neurons=neurons)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10)
    grid_result = grid.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    logger.info('In NeuralNetworkModel | run_model finished')


def create_neuron_tuned_model(colnum):
    model = Sequential()
    model.add(Dense(250, activation='relu', input_shape=(colnum,)))
    model.add(Dropout(0.2))
    model.add(Dense(150, activation='relu', input_shape=(colnum,)))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu', input_shape=(colnum,)))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    print(model.summary())
    model.compile(optimizer='adamax',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def run_neuron_tuned_model(X, y, model_name, df_handler, perform_smote=False):
    logger.info('In NNModelTuning | run_neuron_tuned_model Started for ' + model_name + ' model.')
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if perform_smote:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        y_train = pd.get_dummies(y_train, prefix=df_handler.target_col)
        y_test = pd.get_dummies(y_test, prefix=df_handler.target_col)
    colnum = X_train.shape[1]
    model = KerasClassifier(build_fn=create_neuron_tuned_model, colnum=colnum)
    batch_size = [20, 50, 80, 110]
    epochs = [10, 50, 90, 100]
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10)
    grid_result = grid.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    logger.info('In NeuralNetworkModel | run_neuron_tuned_model finished')


def create_batch_tuned_model(colnum, optimizer='adam'):
    model = Sequential()
    model.add(Dense(250, activation='relu', input_shape=(colnum,)))
    model.add(Dropout(0.2))
    model.add(Dense(150, activation='relu', input_shape=(colnum,)))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu', input_shape=(colnum,)))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    print(model.summary())
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def run_batch_tuned_model(X, y, model_name, df_handler, perform_smote=False):
    logger.info('In NNModelTuning | run_batch_tuned_model Started for ' + model_name + ' model.')
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if perform_smote:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        y_train = pd.get_dummies(y_train, prefix=df_handler.target_col)
        y_test = pd.get_dummies(y_test, prefix=df_handler.target_col)
    colnum = X_train.shape[1]
    model = KerasClassifier(build_fn=create_batch_tuned_model, colnum=colnum, batch_size=50, epochs=100)
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    param_grid = dict(optimizer=optimizer)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10)
    grid_result = grid.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    logger.info('In NeuralNetworkModel | run_batch_tuned_model finished')


def create_optimization_tuned_model(colnum, learn_rate=0.01):
    model = Sequential()
    model.add(Dense(250, activation='relu', input_shape=(colnum,)))
    model.add(Dropout(0.2))
    model.add(Dense(150, activation='relu', input_shape=(colnum,)))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu', input_shape=(colnum,)))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    print(model.summary())
    optimizer = Adadelta(lr=learn_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def run_optimization_tuned_model(X, y, model_name, df_handler, perform_smote=False):
    logger.info('In NNModelTuning | run_optimization_tuned_model Started for ' + model_name + ' model.')
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if perform_smote:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        y_train = pd.get_dummies(y_train, prefix=df_handler.target_col)
        y_test = pd.get_dummies(y_test, prefix=df_handler.target_col)
    colnum = X_train.shape[1]
    model = KerasClassifier(build_fn=create_optimization_tuned_model, colnum=colnum, batch_size=50, epochs=100)
    learn_rate = (0.0001, 0.001, 0.01, 0.1, 1)
    param_grid = dict(learn_rate=learn_rate)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10)
    grid_result = grid.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    logger.info('In NeuralNetworkModel | run_optimization_tuned_model finished')


def create_lr_tuned_model(colnum, init_mode='uniform'):
    model = Sequential()
    model.add(Dense(250, activation='relu', input_shape=(colnum,), kernel_initializer=init_mode))
    model.add(Dropout(0.2))
    model.add(Dense(150, activation='relu', input_shape=(colnum,), kernel_initializer=init_mode))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu', input_shape=(colnum,), kernel_initializer=init_mode))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax', kernel_initializer=init_mode))
    print(model.summary())
    optimizer = Adadelta(lr=1)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def run_lr_tuned_model(X, y, model_name, df_handler, perform_smote=False):
    logger.info('In NNModelTuning | run_lr_tuned_model Started for ' + model_name + ' model.')
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if perform_smote:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        y_train = pd.get_dummies(y_train, prefix=df_handler.target_col)
        y_test = pd.get_dummies(y_test, prefix=df_handler.target_col)
    colnum = X_train.shape[1]
    model = KerasClassifier(build_fn=create_lr_tuned_model, colnum=colnum, batch_size=50, epochs=100)
    init_mode = (
        'uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform')
    param_grid = dict(init_mode=init_mode)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10)
    grid_result = grid.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    logger.info('In NeuralNetworkModel | run_lr_tuned_model finished')


def create_weight_tuned_model(colnum, activation='relu'):
    model = Sequential()
    model.add(Dense(250, activation=activation, input_shape=(colnum,), kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(150, activation=activation, input_shape=(colnum,), kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation=activation, input_shape=(colnum,), kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax', kernel_initializer='glorot_uniform'))
    print(model.summary())
    optimizer = Adadelta(lr=1)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def run_weight_tuned_model(X, y, model_name, df_handler, perform_smote=False):
    logger.info('In NNModelTuning | run_weight_tuned_model Started for ' + model_name + ' model.')
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if perform_smote:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        y_train = pd.get_dummies(y_train, prefix=df_handler.target_col)
        y_test = pd.get_dummies(y_test, prefix=df_handler.target_col)
    colnum = X_train.shape[1]
    model = KerasClassifier(build_fn=create_weight_tuned_model, colnum=colnum, batch_size=50, epochs=100)
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    param_grid = dict(activation=activation)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10)
    grid_result = grid.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    logger.info('In NeuralNetworkModel | run_lr_tuned_model finished')


def create_activation_tuned_model(colnum, dropout_rate):
    model = Sequential()
    model.add(Dense(250, activation='relu', input_shape=(colnum,), kernel_initializer='glorot_uniform'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(150, activation='relu', input_shape=(colnum,), kernel_initializer='glorot_uniform'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(50, activation='relu', input_shape=(colnum,), kernel_initializer='glorot_uniform'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(2, activation='softmax', kernel_initializer='glorot_uniform'))
    print(model.summary())
    optimizer = Adadelta(lr=1)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def run_activation_tuned_model(X, y, model_name, df_handler, perform_smote=False):
    logger.info('In NNModelTuning | run_activation_tuned_model Started for ' + model_name + ' model.')
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if perform_smote:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        y_train = pd.get_dummies(y_train, prefix=df_handler.target_col)
        y_test = pd.get_dummies(y_test, prefix=df_handler.target_col)
    colnum = X_train.shape[1]
    model = KerasClassifier(build_fn=create_activation_tuned_model, colnum=colnum, batch_size=50, epochs=100)
    dropout_rate = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    param_grid = dict(dropout_rate=dropout_rate)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10)
    grid_result = grid.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    logger.info('In NeuralNetworkModel | run_activation_tuned_model finished')


def main():
    logger.info('In NeuralNetworkModel | Main Started')
    pre_process = PreProcessor()
    df = pre_process.load_data()
    df_handler = DataFrameHandler(df)
    label_df = df_handler.get_label_encoded_data()
    scaled_df = df_handler.get_scaled_data(df=label_df)
    attribute_target_split_result = df_handler.split_attribute_and_target(df=scaled_df)
    X = attribute_target_split_result['attributes']
    y = attribute_target_split_result['target']
    print(X.head())
    run_activation_tuned_model(X, y, 'LabeEncoded_MinMaxScaling', df_handler, perform_smote=True)


if __name__ == '__main__':
    main()
