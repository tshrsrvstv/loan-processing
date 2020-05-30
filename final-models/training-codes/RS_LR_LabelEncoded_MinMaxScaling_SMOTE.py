#  RandomizedSearch Logistic Regression Model with Label Encoding Categorical Values,
#  MinMax Scaling Numerical values and SMOTE applied on training data gives
#  below best params as the result.
#  Ref. File results/LR_Model_Output_Summary.txt Line Number 441
#  Best Penalty: l2
#  Best C: 70548.02310718645
#  Best solver: lbfgs
#  We will train a Logistic Regression model on train.csv with above hyperparameters
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import sys
import traceback
import os
import pickle

DEFAULT_DIRECTORY = '../datasets-provided/'
DATA_CSV_FILENAME = 'train.csv'
MODEL_DIRECTORY = '../trained-models/'


class ErrorHandler(object):
    def handleErr(self, err):
        tb = sys.exc_info()[-1]
        stk = traceback.extract_tb(tb, 1)
        functionName = stk[0][2]
        return functionName + ":" + err


class DataFrameHandler():
    def __init__(self, df, parent=None):
        try:
            self.errObj = ErrorHandler()
            self.data_frame_original = df
            self.data_cols = self.data_frame_original.columns
            self.data_shape = self.data_frame_original.shape
            self.categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                                     'poutcome']
            self.numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
            self.target_col = 'target'
            self.scaler = MinMaxScaler()
            self.labelEncoder = LabelEncoder()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))

    def get_dummies_data(self, df=None):
        if df is None:
            df = self.data_frame_original
        try:
            dummies_dataframe = df.copy()
            for col in self.categorical_cols:
                cat_list = pd.get_dummies(dummies_dataframe[col], prefix=col)
                dummies_dataframe = dummies_dataframe.join(cat_list)
            all_dummies_cols = dummies_dataframe.columns.values.tolist()
            cols_to_keep = [col for col in all_dummies_cols if col not in self.categorical_cols]
            dummies_dataframe = dummies_dataframe[cols_to_keep]
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))
        return dummies_dataframe

    def get_label_encoded_data(self, df=None):
        if df is None:
            df = self.data_frame_original
        try:
            label_encoded_dataframe = df.copy()
            for col in self.categorical_cols:
                label_encoded_dataframe[col] = self.labelEncoder.fit_transform(label_encoded_dataframe[col])
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))
        return label_encoded_dataframe

    def get_scaled_data(self, df=None):
        if df is None:
            df = self.data_frame_original
        try:
            scaled_dataframe = df.copy()
            scaled_dataframe[self.numerical_cols] = self.scaler.fit_transform(
                scaled_dataframe[self.numerical_cols])
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))
        return scaled_dataframe

    def split_attribute_and_target(self, df=None):
        if df is None:
            df = self.data_frame_original
        try:
            target = df[self.target_col]
            attribute_set = df.drop(self.target_col, axis=1)
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))
        return {'attributes': attribute_set, 'target': target}

    def get_binned_data(self, df=None, bins_per_col=4):
        if df is None:
            df = self.data_frame_original
        try:
            binned_dataframe = df.copy()
            for col in self.numerical_cols:
                bins = np.linspace(binned_dataframe[col].min(), binned_dataframe[col].max(), bins_per_col + 1)
                binned_dataframe[col] = pd.cut(binned_dataframe[col], bins, precision=1, include_lowest=True,
                                               right=True)
                cat_list = pd.get_dummies(binned_dataframe[col], prefix=col)
                binned_dataframe = binned_dataframe.join(cat_list)
                binned_dataframe = binned_dataframe.drop(col, axis=1)
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))
        return binned_dataframe


def load_data(default_directory=DEFAULT_DIRECTORY):
    try:
        data_file = default_directory + DATA_CSV_FILENAME
        df = pd.read_csv(data_file)
    except Exception as exp:
        err = self.errObj.handleErr(str(exp))
        print(str(err))
    return df


def create_model():
    try:
        model = LogisticRegression(solver='lbfgs', random_state=42, C=70548.02311, penalty='l2', n_jobs=-1, max_iter=1000)
    except Exception as exp:
        err = self.errObj.handleErr(str(exp))
        print(str(err))
    return model


def train_model(X, y, perform_smote=False):
    if perform_smote:
        sm = SMOTE(random_state=42)
        X, y = sm.fit_resample(X, y)
    model = create_model()
    model.fit(X, y)
    return model


def serialize_model(model):
    model_file_name = MODEL_DIRECTORY + 'Logistic_solver_lbfgs_penalty_l2_C_70548'
    with open(model_file_name, 'wb') as f:
        pickle.dump(model, f)


def main():
    df = load_data()
    df_handler = DataFrameHandler(df)
    label_df = df_handler.get_label_encoded_data()
    scaled_df = df_handler.get_scaled_data(df=label_df)
    attribute_target_split_result = df_handler.split_attribute_and_target(df=scaled_df)
    X = attribute_target_split_result['attributes']
    y = attribute_target_split_result['target']
    print(X.head())
    model = train_model(X, y, perform_smote=True)
    print(model)
    serialize_model(model)


if __name__ == '__main__':
    main()
