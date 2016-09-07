import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import time
import datetime

# Run XGBoost
def run_xgb(X_full_train, y_full_train, X_test, random_state=1337):
    eta = 0.2
    max_depth = 5
    subsample = 0.8
    colsample_bytree = 0.8
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state,
    }
    num_boost_round = 500
    early_stopping_rounds = 50
    test_size = 0.3

    X_train, X_valid, y_train, y_valid = train_test_split(X_full_train, y_full_train, test_size=test_size, random_state=random_state)
    print 'Length train:', X_train.shape[0]
    print 'Length valid:', X_valid.shape[0]
    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_valid, y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid), ntree_limit=gbm.best_iteration)
    score = roc_auc_score(y_valid[y_valid.columns[0]].tolist(), check)
    print('Check error value: {:.6f}'.format(score))

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(X_test), ntree_limit=gbm.best_iteration)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction.tolist(), score

# Read in the train/test data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv")
y_test_activity_ids = pd.read_csv("y_test_activity_ids.csv", dtype={'activity_id': np.str})
features = X_train.columns.values

# Print input data shapes for validation
print 'Length of train: ', X_train.shape[0]
print 'Length of y train: ', len(y_train)
print 'Length of test: ', X_test.shape[0]
print('Features [{}]: {}'.format(len(features), sorted(features)))

# Run XGBoost
test_prediction, score = run_xgb(X_train, y_train, X_test)

# Prepare the submission file
test_prediction_df = pd.DataFrame(test_prediction)
print test_prediction_df.shape
print y_test_activity_ids.shape
submission_df = pd.concat([y_test_activity_ids, test_prediction_df], axis=1)
reordered_columns = ['activity_id'] + test_prediction_df.columns.values.tolist()
submission_df = submission_df[reordered_columns]
submission_df.columns = ['activity_id','outcome']
this_filename = 'XGB_submission_df' + str(score) + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + ".csv"
submission_df.to_csv(this_filename, index=False)