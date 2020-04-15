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

DEFAULT_DIRECTORY = os.path.join(os.sep.join(map(str, os.getcwd().split(os.sep)[:-1])), 'dataset')
DATA_CSV_FILENAME = 'LoanApplyData-bank.csv'


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
            logger.error(str(exp))
        logger.info("In PreProcessor | load_data finished")
        return df

    def detect_categorical_columns(self, df):
        logger.info("In PreProcessor | detect_categorical_columns started")
        try:
            logger.debug("In detect_categorical_columns | " + str(df.dtypes))
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(exp))
        logger.info("In PreProcessor | detect_categorical_columns finished")
        return df.columns[df.dtypes == np.object]

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
            logger.error(str(exp))
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


class Outlier:
    def __init__(self, parent=None):
        try:
            self.errObj = ErrorHandler()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))


def main():
    logger.info('Main Started')
    preProcessor = PreProcessor()
    df = preProcessor.load_data()
    categorical_cols = preProcessor.detect_categorical_columns(df)
    logger.info('Categorical Columns in dataset : ' + str(categorical_cols))
    preProcessor.dimension_stat_analysis(df, categorical_cols)
    del preProcessor
    logger.info('Main Finished')


if __name__ == '__main__':
    main()
