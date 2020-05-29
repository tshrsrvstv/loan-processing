import h2o
from h2o.automl import H2OAutoML


def run_model():
    h2o.init()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = y_train.asfactor()
    y_test = y_test.asfactor()
    aml = H2OAutoML(max_models=20, seed=1)
    aml.train(x=x, y=y, training_frame=h2o.H2OFrame(train1))
    lb = aml.leaderboard
    print(lb.head(rows=lb.nrows))
    preds = aml.predict(h2o.H2OFrame(test1))
    lb = h2o.automl.get_leaderboard(aml, extra_columns='ALL')
    return lb