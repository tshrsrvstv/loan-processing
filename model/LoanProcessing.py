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
from model.MissingValue import MissingValue
from model.OutlierDetection import Outlier
from model.PrePocessor import PreProcessor


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
