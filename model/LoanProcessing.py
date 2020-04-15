# pip install plotnine

import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
#from plotnine import *

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
import sys, os, gc, traceback
#import missingno as msno
#from impyute.imputation.cs import mice
from scipy import stats

DEFAULT_DIRECTORY = os.getcwd()
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
            print(str(err))

    @classmethod
    def load_data(default_directory=DEFAULT_DIRECTORY):
        try:
            data_file = os.path.join(default_directory, DATA_CSV_FILENAME)
            df = pd.read_csv(data_file)
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(exp))
        return df


class VisualizeCorrelation():
    def __init__(self, parent=None):
        try:
            self.errObj = ErrorHandler()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))


class MissingValue:
    def __init__(self, parent=None):
        try:
            self.errObj = ErrorHandler()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))


class Outlier:
    def __init__(self, parent=None):
        try:
            self.errObj = ErrorHandler()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            print(str(err))


def main():
    df = PreProcessor().load_data()
    df.head()
    print(df)

if __name__ == '__main__':
    main()
