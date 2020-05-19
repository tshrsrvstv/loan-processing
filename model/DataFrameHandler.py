from model.ErrorHandler import ErrorHandler
from model.Logger import logger
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd
import numpy as np


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
            logger.error(str(err))

    def get_dummies_data(self, df=None):
        logger.info("In DataFrameHandler | get_dummies_data started")
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
            logger.debug('Columns after Dummy Encoding : ' + str(dummies_dataframe.columns.values))
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In DataFrameHandler | get_dummies_data finished")
        return dummies_dataframe

    def get_label_encoded_data(self, df=None):
        logger.info("In DataFrameHandler | get_label_encoded_data started")
        if df is None:
            df = self.data_frame_original
        try:
            label_encoded_dataframe = df.copy()
            for col in self.categorical_cols:
                label_encoded_dataframe[col] = self.labelEncoder.fit_transform(label_encoded_dataframe[col])
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In DataFrameHandler | get_label_encoded_data finished")
        return label_encoded_dataframe

    def get_scaled_data(self, df=None):
        logger.info("In DataFrameHandler | get_scaled_data started")
        if df is None:
            df = self.data_frame_original
        try:
            scaled_dataframe = df.copy()
            scaled_dataframe[self.numerical_cols] = self.scaler.fit_transform(
                scaled_dataframe[self.numerical_cols])
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In DataFrameHandler | get_scaled_data finished")
        return scaled_dataframe

    def split_attribute_and_target(self, df=None):
        logger.info("In DataFrameHandler | split_attribute_and_target started")
        if df is None:
            df = self.data_frame_original
        try:
            target = df[self.target_col]
            attribute_set = df.drop(self.target_col, axis=1)
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In DataFrameHandler | split_attribute_and_target finished")
        return {'attributes': attribute_set, 'target': target}

    def get_binned_data(self, df=None, bins_per_col=4):
        logger.info("In DataFrameHandler | get_binned_data started")
        if df is None:
            df = self.data_frame_original
        try:
            binned_dataframe = df.copy()
            for col in self.numerical_cols:
                bins = np.linspace(binned_dataframe[col].min(), binned_dataframe[col].max(), bins_per_col + 1)
                binned_dataframe[col] = pd.cut(binned_dataframe[col], bins, precision=1, include_lowest=True, right=True)
                cat_list = pd.get_dummies(binned_dataframe[col], prefix=col)
                binned_dataframe = binned_dataframe.join(cat_list)
                binned_dataframe = binned_dataframe.drop(col, axis=1)
            logger.debug('Columns after Dummy Encoding : ' + str(binned_dataframe.columns.values))
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In DataFrameHandler | get_binned_data finished")
        return binned_dataframe
