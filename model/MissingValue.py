from model.ErrorHandler import ErrorHandler
from model.Logger import logger


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