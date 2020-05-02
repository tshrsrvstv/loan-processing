import os

DEFAULT_DIRECTORY = os.path.join(os.sep.join(map(str, os.getcwd().split(os.sep)[:-1])), 'dataset')

DATA_CSV_FILENAME = 'LoanApplyData-bank.csv'
# DATA_CSV_FILENAME = 'LoanApplyData-bank-EditedForTest.csv'

VISUALIZATION_SAVE_DIRECTORY = os.path.join(os.sep.join(map(str, os.getcwd().split(os.sep)[:-1])), 'visualizations')
COLUMNS_CATEGORIZATION_APPLICABLE = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                                     'campaign',
                                     'previous', 'poutcome', 'target']
COLUMN_WISE_IMPUTE_TECHNIQUE_MAP = {
    'job': 'mode',
    'marital': 'mode',
    'education': 'mode',
    'default': 'mode',
    'balance': 'value',
    'housing': 'mode',
    'loan': 'mode',
    'contact': 'mode',
    'day': 'mode',
    'month': 'mode',
    'duration': 'mean',
    'campaign': 'mode',
    'pdays': 'mean',
    'previous': 'mean',
    'poutcome': 'mode',
    'target': 'mode'
}
