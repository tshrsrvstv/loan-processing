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
from impyute.imputation.cs import mice
from scipy import stats
from Logger import logger
from sklearn.preprocessing import LabelEncoder
from EncoderStore import EncoderStore

DEFAULT_DIRECTORY = os.path.join(os.sep.join(map(str, os.getcwd().split(os.sep)[:-1])), 'dataset')

DATA_CSV_FILENAME = 'LoanApplyData-bank.csv'
#DATA_CSV_FILENAME = 'LoanApplyData-bank-EditedForTest.csv'

VISUALIZATION_SAVE_DIRECTORY = os.path.join(os.sep.join(map(str, os.getcwd().split(os.sep)[:-1])), 'visualizations')
COLUMNS_CATEGORIZATION_APPLICABLE = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'campaign',
                                     'previous', 'poutcome', 'target']


class ErrorHandler(object):
    def handleErr(self, err):
        tb = sys.exc_info()[-1]
        stk = traceback.extract_tb(tb, 1)
        functionName = stk[0][2]
        return functionName + ":" + err


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

    def convert_to_categorical_values(self, df, cat_cols):
        logger.info("In PreProcessor | convert_to_categorical_values started")
        try:
            for col in cat_cols:
                if col in COLUMNS_CATEGORIZATION_APPLICABLE:
                    logger.debug('Categorizing Column : ' + str(col))
                    encoder = LabelEncoder()
                    logger.debug('Column unique value : ' + str(df[col].unique()))
                    encoder.fit(df[col].unique())
                    df[col] = encoder.fit_transform(df[col])
                    EncoderStore.save(col, encoder)
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

    def visualize_missing_values(self, df):
        logger.info("In MissingValue | visualize_missing_values started")
        try:
            if not os.path.exists(VISUALIZATION_SAVE_DIRECTORY):
                os.makedirs(VISUALIZATION_SAVE_DIRECTORY)
            msno.matrix(df)
            plt.ion()
            plt.show()
            plt.savefig(os.path.join(VISUALIZATION_SAVE_DIRECTORY, 'missing_number_matrix_visualization.png'))
            plt.pause(1)
            plt.close()
            msno.bar(df)
            plt.ion()
            plt.show()
            plt.savefig(os.path.join(VISUALIZATION_SAVE_DIRECTORY, 'missing_number_bar_chart.png'))
            plt.pause(1)
            plt.close()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In MissingValue | visualize_missing_values finished")

    def visualize_heatmap(self, df):
        logger.info("In MissingValue | visualize_heatmap started")
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
        logger.info("In MissingValue | visualize_heatmap finished")


class Outlier:
    def __init__(self, parent=None):
        try:
            self.errObj = ErrorHandler()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))


def main():
    logger.info('Main Started')
    pre_process = PreProcessor()
    df = pre_process.load_data()
    categorical_cols = pre_process.detect_categorical_columns(df)
    logger.info('Categorical Columns in dataset : ' + str(categorical_cols))
    pre_process.dimension_stat_analysis(df, categorical_cols)
    pre_process.convert_to_categorical_values(df, categorical_cols)
    logger.info('DataFrame After Categorical Column Label Encoding :')
    logger.info(df.head())
    del pre_process
    missing_val = MissingValue()
    missing_val_info = missing_val.get_missing_values_info(df)
    missing_val.visualize_missing_values(df)
    missing_val.visualize_heatmap(df)
    logger.info('Column wise Missing Values in dataset : ' + str(missing_val_info))
    del missing_val
    logger.info('Main Finished')


if __name__ == '__main__':
    main()
