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

    @classmethod
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
    df = PreProcessor().load_data()
    logger.info('Main Finished')


if __name__ == '__main__':
    main()
