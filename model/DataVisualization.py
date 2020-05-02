import os
import numpy as np
from model.Constants import VISUALIZATION_SAVE_DIRECTORY
from model.EncoderStore import EncoderStore
from model.ErrorHandler import ErrorHandler
from model.Logger import logger
import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualisation():
    def __init__(self, parent=None):
        try:
            self.errObj = ErrorHandler()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))

    def visualize_target(self, df):
        logger.info("In DataVisualisation | visualize_target started")
        try:
            target_encoder = EncoderStore.get('target')
            labels = target_encoder.classes_
            logger.debug("Target Labels : " + str(labels))
            inverse_target = target_encoder.inverse_transform(df['target'])
            target_as_no = (inverse_target == 'no').sum()
            target_as_yes = (inverse_target == 'yes').sum()
            sizes = [target_as_no, target_as_yes]
            logger.debug("Target counts : " + str(sizes))
            colors = ['lightcoral', 'yellowgreen']
            patches, texts, percent = plt.pie(sizes, colors=colors, autopct='%1.1f%%', labels=labels, startangle=90,
                                              wedgeprops={'edgecolor': 'w'})
            plt.legend(patches, labels, loc="best")
            plt.axis('equal')
            plt.tight_layout()
            plt.ion()
            plt.show()
            plt.savefig(os.path.join(VISUALIZATION_SAVE_DIRECTORY, 'pie_visualization_target'))
            plt.pause(1)
            plt.close()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In DataVisualisation | visualize_target finished")

    def visualize_age_vs_target(self, df):
        logger.info("In DataVisualisation | visualize_age_vs_target started")
        try:
            target_encoder = EncoderStore.get('target')
            bar_labels = target_encoder.classes_
            inverse_target = target_encoder.inverse_transform(df['target'])
            target_as_no = (inverse_target == 'no').sum()
            target_as_yes = (inverse_target == 'yes').sum()
            age_gt_50_and_target_no = ((df['age'] > 50) & (inverse_target == 'no')).sum()
            age_lte_50_and_target_no = ((df['age'] <= 50) & (inverse_target == 'no')).sum()
            age_gt_50_and_target_yes = ((df['age'] > 50) & (inverse_target == 'yes')).sum()
            age_lte_50_and_target_yes = ((df['age'] <= 50) & (inverse_target == 'yes')).sum()
            x_labels = ['age>50', 'age<=50']
            x = np.arange(2)
            ax = plt.subplot(1, 1, 1)
            w = 0.3
            not_paid = [age_gt_50_and_target_no / target_as_no, age_lte_50_and_target_no / target_as_no]
            paid = [age_gt_50_and_target_yes / target_as_yes, age_lte_50_and_target_yes / target_as_yes]
            plt.xticks(x + w / 2, x_labels)
            not_paid_bar = ax.bar(x, not_paid, color="lightcoral", width=w)
            paid_bar = ax.bar(x + w, paid, color="yellowgreen", width=w)
            plt.legend([not_paid_bar, paid_bar], bar_labels)
            plt.ion()
            plt.show()
            plt.savefig(os.path.join(VISUALIZATION_SAVE_DIRECTORY, 'bar_visualization_age_vs_target'))
            plt.pause(1)
            plt.close()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In DataVisualisation | visualize_age_vs_target finished")

    def visualize_job_vs_target(self, df):
        logger.info("In DataVisualisation | visualize_job_vs_target started")
        try:
            target_encoder = EncoderStore.get('target')
            inverse_target = target_encoder.inverse_transform(df['target'])
            job_encoder = EncoderStore.get('job')
            job_labels = job_encoder.classes_
            inverse_job = job_encoder.inverse_transform(df['job'])
            sizes_not_paid = []
            sizes_paid = []
            for label in job_labels:
                job_label_and_target_no = ((inverse_job == label) & (inverse_target == 'no')).sum()
                job_label_and_target_yes = ((inverse_job == label) & (inverse_target == 'yes')).sum()
                sizes_not_paid.append(job_label_and_target_no)
                sizes_paid.append(job_label_and_target_yes)
            colors = ["aqua", "azure", "brown", "chartreuse", "coral", "crimson", "cyan", "fuchsia", "goldenrod",
                      "lavender", "purple", "teal"]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 20))
            ax1.pie(sizes_not_paid, autopct='%1.1f%%', labels=job_labels, startangle=90, colors=colors,
                    wedgeprops={'edgecolor': 'w'})
            ax1.axis('equal')
            ax1.set_title('Target: no')
            ax2.pie(sizes_paid, autopct='%1.1f%%', labels=job_labels, startangle=90,
                    colors=colors,
                    wedgeprops={'edgecolor': 'w'})
            ax2.set_title('Target: yes')
            ax2.axis('equal')
            plt.tight_layout()
            plt.ion()
            plt.show()
            plt.savefig(os.path.join(VISUALIZATION_SAVE_DIRECTORY, 'pie_visualization_job_vs_target'))
            plt.pause(1)
            plt.close()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In DataVisualisation | visualize_job_vs_target finished")

    def visualize_marital_status_vs_target(self, df):
        logger.info("In DataVisualisation | visualize_marital_status_vs_target started")
        try:
            target_encoder = EncoderStore.get('target')
            bar_labels = target_encoder.classes_
            inverse_target = target_encoder.inverse_transform(df['target'])
            target_as_no = (inverse_target == 'no').sum()
            target_as_yes = (inverse_target == 'yes').sum()
            marital_status_encoder = EncoderStore.get('marital')
            inverse_marital_status = marital_status_encoder.inverse_transform(df['marital'])
            x_labels = marital_status_encoder.classes_
            not_paid = []
            paid = []
            for stat in x_labels:
                marital_stat_and_target_no = ((inverse_marital_status == stat) & (inverse_target == 'no')).sum()
                marital_stat_and_target_yes = ((inverse_marital_status == stat) & (inverse_target == 'yes')).sum()
                not_paid.append(marital_stat_and_target_no / target_as_no)
                paid.append(marital_stat_and_target_yes / target_as_yes)
            x = np.arange(3)
            ax = plt.subplot(1, 1, 1)
            w = 0.3
            plt.xticks(x + w / 2, x_labels)
            not_paid_bar = ax.bar(x, not_paid, color="lightcoral", width=w)
            paid_bar = ax.bar(x + w, paid, color="yellowgreen", width=w)
            plt.legend([not_paid_bar, paid_bar], bar_labels)
            plt.ion()
            plt.show()
            plt.savefig(os.path.join(VISUALIZATION_SAVE_DIRECTORY, 'bar_visualization_marital_status_vs_target'))
            plt.pause(1)
            plt.close()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In DataVisualisation | visualize_marital_status_vs_target finished")

    def visualize_education_vs_target(self, df):
        logger.info("In DataVisualisation | visualize_education_vs_target started")
        try:
            target_encoder = EncoderStore.get('target')
            bar_labels = target_encoder.classes_
            inverse_target = target_encoder.inverse_transform(df['target'])
            target_as_no = (inverse_target == 'no').sum()
            target_as_yes = (inverse_target == 'yes').sum()
            education_encoder = EncoderStore.get('education')
            inverse_education = education_encoder.inverse_transform(df['education'])
            x_labels = education_encoder.classes_
            not_paid = []
            paid = []
            for stat in x_labels:
                education_stat_and_target_no = ((inverse_education == stat) & (inverse_target == 'no')).sum()
                education_stat_and_target_yes = ((inverse_education == stat) & (inverse_target == 'yes')).sum()
                not_paid.append(education_stat_and_target_no / target_as_no)
                paid.append(education_stat_and_target_yes / target_as_yes)
            x = np.arange(4)
            ax = plt.subplot(1, 1, 1)
            w = 0.3
            plt.xticks(x + w / 2, x_labels)
            not_paid_bar = ax.bar(x, not_paid, color="lightcoral", width=w)
            paid_bar = ax.bar(x + w, paid, color="yellowgreen", width=w)
            plt.legend([not_paid_bar, paid_bar], bar_labels)
            plt.ion()
            plt.show()
            plt.savefig(os.path.join(VISUALIZATION_SAVE_DIRECTORY, 'bar_visualization_education_vs_target'))
            plt.pause(1)
            plt.close()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In DataVisualisation | visualize_education_vs_target finished")

    def visualize_default_vs_target(self, df):
        logger.info("In DataVisualisation | visualize_default_vs_target started")
        try:
            target_encoder = EncoderStore.get('target')
            inverse_target = target_encoder.inverse_transform(df['target'])
            default_encoder = EncoderStore.get('default')
            target_labels = target_encoder.classes_
            inverse_default = default_encoder.inverse_transform(df['default'])
            sizes_default_no = [((inverse_default == 'no') & (inverse_target == 'no')).sum(), ((inverse_default == 'no') & (inverse_target == 'yes')).sum()]
            sizes_default_yes = [((inverse_default == 'yes') & (inverse_target == 'no')).sum(), ((inverse_default == 'yes') & (inverse_target == 'yes')).sum()]
            sizes = sizes_default_no + sizes_default_yes
            overall_labels = ['Default:no, Target:no', 'Default:no, Target:yes', 'Default:yes, Target:no', 'Default:yes, Target:yes']
            overall_colors = ['olive', 'pink', 'orange', 'blue']
            colors = ['lightcoral', 'yellowgreen']
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
            ax1.pie(sizes_default_no, autopct='%1.1f%%', labels=target_labels, startangle=90, colors=colors,
                    wedgeprops={'edgecolor': 'w'})
            ax1.axis('equal')
            ax1.set_title('Default: no')
            ax2.pie(sizes_default_yes, autopct='%1.1f%%', labels=target_labels, startangle=90,
                    colors=colors,
                    wedgeprops={'edgecolor': 'w'})
            ax2.set_title('Default: yes')
            ax2.axis('equal')
            ax3.pie(sizes, autopct='%1.1f%%', labels=overall_labels, startangle=90,
                    colors=overall_colors,
                    wedgeprops={'edgecolor': 'w'})
            ax3.set_title('Overall')
            ax3.axis('equal')
            plt.tight_layout()
            plt.ion()
            plt.show()
            plt.savefig(os.path.join(VISUALIZATION_SAVE_DIRECTORY, 'pie_visualization_default_vs_target'))
            plt.pause(1)
            plt.close()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In DataVisualisation | visualize_default_vs_target finished")

    def visualize_balance_vs_target(self, df):
        logger.info("In DataVisualisation | visualize_balance_vs_target started")
        try:
            target_encoder = EncoderStore.get('target')
            bar_labels = target_encoder.classes_
            inverse_target = target_encoder.inverse_transform(df['target'])
            target_as_no = (inverse_target == 'no').sum()
            target_as_yes = (inverse_target == 'yes').sum()
            bal_gt_800_and_target_no = ((df['balance'] > 800) & (inverse_target == 'no')).sum()
            bal_lte_800_and_target_no = ((df['balance'] <= 800) & (inverse_target == 'no')).sum()
            bal_gt_800_and_target_yes = ((df['balance'] > 800) & (inverse_target == 'yes')).sum()
            bal_lte_800_and_target_yes = ((df['balance'] <= 800) & (inverse_target == 'yes')).sum()
            x_labels = ['balance>800', 'balance<=800']
            x = np.arange(2)
            ax = plt.subplot(1, 1, 1)
            w = 0.3
            not_paid = [bal_gt_800_and_target_no / target_as_no, bal_lte_800_and_target_no / target_as_no]
            paid = [bal_gt_800_and_target_yes / target_as_yes, bal_lte_800_and_target_yes / target_as_yes]
            plt.xticks(x + w / 2, x_labels)
            not_paid_bar = ax.bar(x, not_paid, color="lightcoral", width=w)
            paid_bar = ax.bar(x + w, paid, color="yellowgreen", width=w)
            plt.legend([not_paid_bar, paid_bar], bar_labels)
            plt.ion()
            plt.show()
            plt.savefig(os.path.join(VISUALIZATION_SAVE_DIRECTORY, 'bar_visualization_balance_vs_target'))
            plt.pause(1)
            plt.close()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In DataVisualisation | visualize_balance_vs_target finished")

    def visualize_campaign_vs_target(self, df):
        logger.info("In DataVisualisation | visualize_duration_vs_target started")
        try:
            target_encoder = EncoderStore.get('target')
            bar_labels = target_encoder.classes_
            inverse_target = target_encoder.inverse_transform(df['target'])
            target_as_no = (inverse_target == 'no').sum()
            target_as_yes = (inverse_target == 'yes').sum()
            campaign_gt_5_and_target_no = ((df['campaign'] > 5) & (inverse_target == 'no')).sum()
            campaign_lte_5_and_target_no = ((df['campaign'] <= 5) & (inverse_target == 'no')).sum()
            campaign_gt_5_and_target_yes = ((df['campaign'] > 5) & (inverse_target == 'yes')).sum()
            campaign_lte_5_and_target_yes = ((df['campaign'] <= 5) & (inverse_target == 'yes')).sum()
            x_labels = ['campaign>5', 'campaign<=5']
            x = np.arange(2)
            ax = plt.subplot(1, 1, 1)
            w = 0.3
            not_paid = [campaign_gt_5_and_target_no / target_as_no, campaign_lte_5_and_target_no / target_as_no]
            paid = [campaign_gt_5_and_target_yes / target_as_yes, campaign_lte_5_and_target_yes / target_as_yes]
            plt.xticks(x + w / 2, x_labels)
            not_paid_bar = ax.bar(x, not_paid, color="lightcoral", width=w)
            paid_bar = ax.bar(x + w, paid, color="yellowgreen", width=w)
            plt.legend([not_paid_bar, paid_bar], bar_labels)
            plt.ion()
            plt.show()
            plt.savefig(os.path.join(VISUALIZATION_SAVE_DIRECTORY, 'bar_visualization_campaign_vs_target'))
            plt.pause(1)
            plt.close()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In DataVisualisation | visualize_duration_vs_target finished")

    def visualize_duration_vs_target(self, df):
        logger.info("In DataVisualisation | visualize_duration_vs_target started")
        try:
            target_encoder = EncoderStore.get('target')
            bar_labels = target_encoder.classes_
            inverse_target = target_encoder.inverse_transform(df['target'])
            target_as_no = (inverse_target == 'no').sum()
            target_as_yes = (inverse_target == 'yes').sum()
            duration_gt_180_and_target_no = ((df['duration'] > 180) & (inverse_target == 'no')).sum()
            duration_lte_180_and_target_no = ((df['duration'] <= 180) & (inverse_target == 'no')).sum()
            duration_gt_180_and_target_yes = ((df['duration'] > 180) & (inverse_target == 'yes')).sum()
            duration_lte_180_and_target_yes = ((df['duration'] <= 180) & (inverse_target == 'yes')).sum()
            x_labels = ['duration>180', 'duration<=180']
            x = np.arange(2)
            ax = plt.subplot(1, 1, 1)
            w = 0.3
            not_paid = [duration_gt_180_and_target_no / target_as_no, duration_lte_180_and_target_no / target_as_no]
            paid = [duration_gt_180_and_target_yes / target_as_yes, duration_lte_180_and_target_yes / target_as_yes]
            plt.xticks(x + w / 2, x_labels)
            not_paid_bar = ax.bar(x, not_paid, color="lightcoral", width=w)
            paid_bar = ax.bar(x + w, paid, color="yellowgreen", width=w)
            plt.legend([not_paid_bar, paid_bar], bar_labels)
            plt.ion()
            plt.show()
            plt.savefig(os.path.join(VISUALIZATION_SAVE_DIRECTORY, 'bar_visualization_duration_vs_target'))
            plt.pause(1)
            plt.close()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In DataVisualisation | visualize_duration_vs_target finished")

    def visualize_feature_correlation_heat_map(self, df):
        logger.info("In DataVisualisation | visualize_feature_correlation_heat_map started")
        try:
            fig,ax = plt.subplots(figsize=(20,20))
            chart = sns.heatmap(df.corr(), ax=ax, annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
            chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
            chart.set_yticklabels(chart.get_yticklabels(), rotation=0)
            plt.ion()
            plt.show()
            plt.savefig(os.path.join(VISUALIZATION_SAVE_DIRECTORY, 'sns_correlation_heatmap'))
            plt.pause(1)
            plt.close()
        except Exception as exp:
            err = self.errObj.handleErr(str(exp))
            logger.error(str(err))
        logger.info("In DataVisualisation | visualize_feature_correlation_heat_map finished")
