import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from scipy.io import arff
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import time

# Read data
start = time.time()
df = pd.read_csv("american_bankruptcy_dataset.csv")
df = df.groupby("company_name").last().reset_index()

# Convert to integer-encoded labels
df.loc[:, 'status_label'] = df.loc[:, 'status_label'].eq('failed').mul(1)
training_df = df[(df['year'] == 2015) | (df['year'] == 2016) | (df['year'] == 2017)]

training_df = training_df.drop(['year', 'company_name'], axis=1)

y = training_df.iloc[:, 0]

X = training_df.iloc[:, 1:]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)

# XGBoost
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
xgb_param_grid={"n_estimators":[100],"max_depth":[10],"learning_rate":[0.30],"min_child_weight":[7],
                'alpha':[0.01],'scale_pos_weight':[10],'colsample_bytree':[1]}
xgb = XGBClassifier(objective="binary:logistic", eval_metric="auc", random_state=42)
xgb.fit(Xtrain,ytrain, eval_set=[(Xtrain, ytrain), (Xtest, ytest)])

eval_result = xgb.evals_result()

with open('xgboost_acc.csv', 'w') as f:
    for acc in xgb.evals_result()['validation_0']['auc']:
        f.write("%s,\n"%(acc))

# CatBoost
# from catboost import CatBoostClassifier
# cat_sm = CatBoostClassifier(eval_metric='AUC', learning_rate=0.1, iterations=100, random_seed=42, max_depth=10, auto_class_weights='Balanced', verbose=True)

# cat_sm.fit(Xtrain, ytrain, eval_set=[(Xtrain, ytrain), (Xtest, ytest)])
# eval_result = cat_sm.get_evals_result()

# with open('catboost_acc.csv', 'w') as f:
#     for acc in cat_sm.get_evals_result()['validation_0']['AUC']:
#         f.write("%s,\n"%(acc))

# LightGBM
# import lightgbm as lgb
# lgb_model = lgb.LGBMClassifier()

# lgb_model.fit(Xtrain, ytrain, eval_set=[(Xtrain, ytrain), (Xtest, ytest)], eval_metric='AUC')
# eval_result = lgb_model.evals_result_

# with open('lgbm_acc.csv', 'w') as f:
#     for acc in lgb_model.evals_result_['training']['auc']:
#         f.write("%s,\n"%(acc))

y_pred = xgb.predict(Xtest)
#y_pred = cat_sm.predict(Xtest)
#y_pred = lgb_model.predict(Xtest)

# Get metrics
conf_matrix = confusion_matrix(ytest, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp)
accuracy = (tn + tp) / (tp + tn +  fn + fp)
precision = tp / (tp + fp)
sensitivity = tp / (tp + fn)
f_measure = (2*precision*sensitivity) / (precision + sensitivity)
fpr, tpr, threshold = roc_curve(ytest, y_pred)

# Print the results
print(f"Accuracy: {accuracy:.5f}")
print(f"Precision: {precision:.5f}")
print(f"Sensitivity: {sensitivity:.5f}")
print(f"Specificity: {specificity:.5f}")
print(f"F-measure: {f_measure:.5f}")
print(f"Confusion matrix:\n{conf_matrix}")
print("AUC: ", auc(fpr, tpr))

end = time.time()
print("time: ", end - start)
print("")