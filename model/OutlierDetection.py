import os

import pandas as pd
from seaborn import boxplot
import matplotlib.pyplot as plt

from model.Constants import VISUALIZATION_SAVE_DIRECTORY
from model.ErrorHandler import ErrorHandler
from model.Logger import logger
from sklearn.cluster import DBSCAN
from matplotlib import cm
from matplotlib import pyplot as plt


class Outlier:
    def __init__(self, parent=None):
        try:
            self.errObj = ErrorHandler()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))

    def visualize_outlier(self, df):
        logger.info("In OutlierDetection | visualize_outlier started")
        try:
            chart = boxplot(x='variable', y='value', data=pd.melt(df), width=0.5, palette="colorblind")
            chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
            plt.ion()
            plt.show()
            plt.savefig(os.path.join(VISUALIZATION_SAVE_DIRECTORY, 'outlier_visualization'))
            plt.pause(1)
            plt.close()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In OutlierDetection | visualize_outlier finished")

    def outlier_DBSCAN(self, df, numerical_cols):
        for col in numerical_cols:
            outlier_detector = DBSCAN(
            eps=.2,
            metric='euclidean',
            min_samples=5,
            n_jobs=-1)
            clusters = outlier_detector.fit_predict(df[[col]])
            cmap = cm.get_cmap('Set1')
            df.plot.scatter(x=col, y='target', c = clusters, cmap = cmap,colorbar = False)
            plt.ion()
            plt.show()
            plt.savefig(os.path.join(VISUALIZATION_SAVE_DIRECTORY, 'outlier_visualization_dbscan' + str(col)))
            plt.pause(1)
            plt.close()
