import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from plotnine import *
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
import sys, os, gc, traceback
import missingno as msno
from impyute.imputation.cs import mice, fast_knn
from scipy import stats
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

from model.FeatureEngineering import FeatureEngineering
from model.Constants import DEFAULT_DIRECTORY, DATA_CSV_FILENAME, COLUMNS_CATEGORIZATION_APPLICABLE, \
    VISUALIZATION_SAVE_DIRECTORY, COLUMN_WISE_IMPUTE_TECHNIQUE_MAP
from model.DataVisualization import DataVisualisation
from model.EncoderStore import EncoderStore
from model.ErrorHandler import ErrorHandler
from model.Logger import logger
from model.OutlierDetection import Outlier


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


class VisualizeCorrelation():
    def __init__(self, parent=None):
        try:
            self.errObj = ErrorHandler()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))


class MissingValue:
    def __init__(self, parent=None):
        try:
            self.errObj = ErrorHandler()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))

    def get_missing_values_info(self, df):
        logger.info("In MissingValue | get_missing_values_info started")
        info = {}
        try:
            for col in df.columns:
                missing_val_count = df[col].isnull().sum()
                total_row_count = df[col].shape[0]
                logger.debug("Missing values in Column " + col + " : " + str(missing_val_count))
                logger.debug("Total Entries in Column " + col + " : " + str(total_row_count))
                info[col] = {
                    'count': missing_val_count,
                    'percentage': (missing_val_count / total_row_count) * 100
                }
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In MissingValue | get_missing_values_info finished")
        return info

    def visualize_missing_values(self, df, postSubscript=False):
        logger.info("In MissingValue | visualize_missing_values started")
        try:
            if not os.path.exists(VISUALIZATION_SAVE_DIRECTORY):
                os.makedirs(VISUALIZATION_SAVE_DIRECTORY)
            msno.matrix(df)
            plt.ion()
            plt.show()
            filename = 'missing_number_matrix_visualization.png' if not postSubscript else 'missing_number_matrix_visualization_post.png'
            plt.savefig(os.path.join(VISUALIZATION_SAVE_DIRECTORY, filename))
            plt.pause(1)
            plt.close()
            msno.bar(df)
            plt.ion()
            plt.show()
            filename = 'missing_number_bar_chart.png' if not postSubscript else 'missing_number_bar_chart_post.png'
            plt.savefig(os.path.join(VISUALIZATION_SAVE_DIRECTORY, filename))
            plt.pause(1)
            plt.close()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In MissingValue | visualize_missing_values finished")

    def visualize_missing_value_heatmap(self, df):
        logger.info("In MissingValue | visualize_missing_value_heatmap started")
        try:
            msno.heatmap(df)
            plt.ion()
            plt.show()
            plt.savefig(os.path.join(VISUALIZATION_SAVE_DIRECTORY, 'correlation_heatmap.png'))
            plt.pause(1)
            plt.close()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In MissingValue | visualize_missing_value_heatmap finished")

    def impute_missing_values(self, df, missing_val_info, method='strategic'):
        logger.info("In MissingValue | impute_missing_value started")
        try:
            possible_methods = ['strategic', 'knn', 'mice']
            if method in possible_methods:
                if method == 'strategic':
                    for col in df.columns:
                        if missing_val_info[col]['percentage'] > 0:
                            logger.debug('Strategically imputing column : ' + str(col))
                            column_imputation_method = COLUMN_WISE_IMPUTE_TECHNIQUE_MAP.get(col)
                            if column_imputation_method == 'mode':
                                self.__impute_by_mode(df, col)
                            elif column_imputation_method == 'mean':
                                self.__impute_by_mean(df, col)
                            elif column_imputation_method == 'median':
                                self.__impute_by_median(df, col)
                            elif column_imputation_method == 'value':
                                self.__impute_by_value(df, col, 0)
                elif method == 'knn':
                    self.__impute_by_knn(df)
                elif method == 'mice':
                    self.__impute_by_mice(df)
            else:
                logger.error("Incorrect Imputation Method !!! Possible values : strategic, knn, mice")
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In MissingValue | impute_missing_value finished")

    def __impute_by_mode(self, df, col):
        logger.info("In MissingValue | __impute_by_mode started")
        try:
            column_mode = df[col].mode()
            logger.debug("Mode obtained for column " + str(col) + " : " + str(column_mode))
            df[col] = df[col].fillna(column_mode)
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In MissingValue | __impute_by_mode finished")

    def __impute_by_mean(self, df, col):
        logger.info("In MissingValue | __impute_by_mean started")
        try:
            column_mean = df[col].mean()
            logger.debug("Mean obtained for column " + str(col) + " : " + str(column_mean))
            df[col] = df[col].fillna(column_mean)
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In MissingValue | __impute_by_mean finished")

    def __impute_by_median(self, df, col):
        logger.info("In MissingValue | __impute_by_median started")
        try:
            column_median = df[col].median()
            logger.debug("Mean obtained for column " + str(col) + " : " + str(column_median))
            df[col] = df[col].fillna(column_median)
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In MissingValue | __impute_by_median finished")

    def __impute_by_value(self, df, col, value):
        logger.info("In MissingValue | __impute_by_value started")
        try:
            logger.debug("Value to replace NAN for column " + str(col) + " : " + str(value))
            df[col] = df[col].fillna(value)
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In MissingValue | __impute_by_value finished")

    def __impute_by_knn(self, df):
        logger.info("In MissingValue | __impute_by_knn started")
        try:
            logger.debug("Applying KNN for imputation with k=1")
            df = fast_knn(k=1, data=df)
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In MissingValue | __impute_by_knn finished")
        return df

    def __impute_by_mice(self, df):
        logger.info("In MissingValue | __impute_by_mice started")
        try:
            df = mice(data=df)
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In MissingValue | __impute_by_mice finished")
        return df


def main():
    logger.info('Main Started')
    pre_process = PreProcessor()
    df = pre_process.load_data()
    df = df.drop(['contact', 'day', 'month'], axis=1)
    logger.info(df.columns)
    df = pre_process.apply_min_max_scaling(df)
    logger.info(df.head())
    categorical_cols = pre_process.detect_categorical_columns(df)
    logger.info('Categorical Columns in dataset : ' + str(categorical_cols))
    pre_process.dimension_stat_analysis(df, categorical_cols)
    df = pre_process.convert_to_categorical_values(df, categorical_cols, use_label_encoder=True)
    logger.info('DataFrame After Categorical Column Encoding :')
    logger.info(df.head())
    outlier = Outlier()
    outlier.visualize_outlier(df)
    outlier.outlier_DBSCAN(df, pre_process.get_numeric_cols(df))
    visualization = DataVisualisation()
    visualization.visualize_target(df)
    visualization.visualize_age_vs_target(df)
    visualization.visualize_job_vs_target(df)
    visualization.visualize_marital_status_vs_target(df)
    visualization.visualize_education_vs_target(df)
    visualization.visualize_default_vs_target(df)
    visualization.visualize_balance_vs_target(df)
    visualization.visualize_duration_vs_target(df)
    visualization.visualize_campaign_vs_target(df)
    visualization.visualize_feature_correlation_heat_map(df)
    del visualization
    missing_val = MissingValue()
    missing_val_info = missing_val.get_missing_values_info(df)
    dataset_has_missing_values = False
    for key in missing_val_info:
        if missing_val_info[key]['percentage'] > 0:
            dataset_has_missing_values = True
            break
    if dataset_has_missing_values:
        logger.info('Column wise Missing Values in dataset : ' + str(missing_val_info))
        missing_val.visualize_missing_values(df)
        missing_val.visualize_missing_value_heatmap(df)
        missing_val.impute_missing_values(df, missing_val_info, method='strategic')
        missing_val.visualize_missing_values(df, postSubscript=True)
    else:
        logger.info('The given dataset has no missing values.')
    del missing_val
    target = df['target']
    df = df.drop('target', axis=1)
    logger.info(categorical_cols.size)
    categorical_cols = categorical_cols.delete(categorical_cols.size-1)
    logger.info(categorical_cols)
    feature_engg = FeatureEngineering()
    df = feature_engg.create_category_percent(df, categorical_cols)
    df = pre_process.convert_to_categorical_values(df, categorical_cols, use_label_encoder=False)
    df = feature_engg.create_bin(df, ['age', 'balance', 'duration', 'pdays', 'previous', 'campaign'], number_of_bins=5)
    logger.info('Shape After feature engineering : ' + str(df.shape))
    df.to_csv(r'C:\AIML\Capstone Project Loan Processing\CodeBase\loan-processing\dataset\modified_df.csv', index=False)
    logger.info('Main Finished')


if __name__ == '__main__':
    main()
