import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from model.Constants import DEFAULT_DIRECTORY, DATA_CSV_FILENAME, COLUMNS_CATEGORIZATION_APPLICABLE
from model.ErrorHandler import ErrorHandler
from model.Logger import logger


class PreProcessor():
    def __init__(self, parent=None):
        try:
            self.errObj = ErrorHandler()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))

    def load_data(self, default_directory=DEFAULT_DIRECTORY):
        logger.info("In PreProcessor | load_data started")
        try:
            data_file = os.path.join(default_directory, DATA_CSV_FILENAME)
            logger.debug("In load_data | Reading Data File : " + data_file)
            df = pd.read_csv(data_file)
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In PreProcessor | load_data finished")
        return df

    def detect_categorical_columns(self, df):
        logger.info("In PreProcessor | detect_categorical_columns started")
        try:
            logger.debug("In detect_categorical_columns | " + str(df.dtypes))
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In PreProcessor | detect_categorical_columns finished")
        return df.columns[df.dtypes == np.object]

    def convert_to_categorical_values(self, df, cat_cols, use_label_encoder=False):
        logger.info("In PreProcessor | convert_to_categorical_values started")
        try:
            if use_label_encoder:
                for col in cat_cols:
                    if col in COLUMNS_CATEGORIZATION_APPLICABLE:
                        logger.debug('Categorizing Column : ' + str(col))
                        encoder = LabelEncoder()
                        logger.debug('Column unique value : ' + str(df[col].unique()))
                        encoder.fit(df[col].unique())
                        df[col] = encoder.fit_transform(df[col])
                        EncoderStore.save(col, encoder)
            if not use_label_encoder:
                one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                for col in cat_cols:
                    enc_df = pd.DataFrame(one_hot_encoder.fit_transform(df[[col]]))
                    enc_df.columns = one_hot_encoder.get_feature_names([col])
                    df = df.join(enc_df)
                    df = df.drop(col, axis=1)
                logger.info('Columns in dataframe after one hot encoding: ' + str(df.columns))
                logger.info('Shape of dataframe after one hot encoding: ' + str(df.shape))
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In PreProcessor | convert_to_categorical_values finished")
        return df


    def dimension_stat_analysis(self, df, categorical_cols):
        logger.info("In PreProcessor | dimension_stat_analysis started")
        try:
            logger.info('Dataset shape : ' + str(df.shape))
            non_categorical_cols = [col for col in df.columns if col not in categorical_cols]
            for col in non_categorical_cols:
                logger.info('Column Name : ' + str(col))
                logger.info('Column Mean : ' + str(df[col].mean()))
                logger.info('Column Median : ' + str(df[col].median()))
                logger.info('Column Standard Deviation : ' + str(df[col].std()))
                logger.info('Column Minima : ' + str(df[col].min()))
                logger.info('Column Maxima : ' + str(df[col].max()))
                logger.info('Column Quantile : ' + str(df[col].quantile([0.25, 0.5, 0.75])))
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In PreProcessor | dimension_stat_analysis finished")

    def get_numeric_cols(self, data):
        try:
            cols = data.select_dtypes(include=["number"]).columns
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        return cols

    def apply_min_max_scaling(self, df):
        logger.info("In PreProcessor | apply_min_max_scaling started")
        try:
            scaler = MinMaxScaler()
            for col in self.get_numeric_cols(df):
                df[col] = scaler.fit_transform(df[[col]])
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In PreProcessor | apply_min_max_scaling finished")
        return df
