import numpy as np
import pandas as pd

from model.ErrorHandler import ErrorHandler
from model.Logger import logger


class FeatureEngineering():
    def __init__(self, parent=None):
        try:
            self.errObj = ErrorHandler()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))

    def create_bin(self, data, numerical_cols, number_of_bins=4):
        try:
            logger.info("In FeatureEngineering | create_bin started")
            d1 = data.copy()
            for col in numerical_cols:
                bins = np.linspace(d1[col].min(), d1[col].max(), number_of_bins)
                d1[col + '_bin'] = pd.cut(d1[col], bins, precision=1, include_lowest=True, right=True)
            logger.info("In FeatureEngineering | create_bin finished")
            return d1
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))

    def create_category_percent(self, data, categorical_cols):
        try:
            logger.info("In FeatureEngineering | create_category_percent started")
            d1 = data.copy()
            length = len(d1)
            for col in categorical_cols:
                d1[col + 'Pct'] = (d1[col].groupby(d1[i]).transform('count')) * 100 / length
            logger.info("In FeatureEngineering | create_category_percent finished")
            return d1
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
