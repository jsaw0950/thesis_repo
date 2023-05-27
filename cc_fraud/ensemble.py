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

start = time.time()
df = pd.read_csv('creditcard_csv.csv')
df_cols = df.columns
df.drop_duplicates(inplace=True)

# scaler = MinMaxScaler()
# sampled_df = scaler.fit_transform(df)
# sampled_df = pd.DataFrame(sampled_df, columns=df_cols)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state = 42)

# from imblearn.over_sampling import SMOTE
# sm = SMOTE()
# X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

# from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
# xgb = XGBClassifier(objective="binary:logistic", eval_metric="auc", random_state=42, n_estimators = 100, max_depth = 10, learning_rate = 0.1)
# xgb.fit(Xtrain,ytrain, eval_set=[(Xtrain, ytrain), (Xtest, ytest)])
# eval_result = xgb.evals_result()

# with open('xgboost_acc.csv', 'w') as f:
#     for acc in xgb.evals_result()['validation_0']['auc']:
#         f.write("%s,\n"%(acc))


# from catboost import CatBoostClassifier

# cat_params = {'eval_metric': ['AUC'],
#               'iterations': [100],
#               'learning_rate' : [0.1],
#               'random_seed' : [42],
#               'auto_class_weights' : ['Balanced']
#             }
# cat_sm = CatBoostClassifier(verbose = 0)

# cat_sm.fit(Xtrain, ytrain, eval_set=[(Xtrain, ytrain), (Xtest, ytest)])
# eval_result = cat_sm.get_evals_result()

# with open('catboost_acc.csv', 'w') as f:
#     for acc in cat_sm.get_evals_result()['validation_0']['AUC']:
#         f.write("%s,\n"%(acc))

# rand_cat = GridSearchCV(cat_sm, cat_params)
# rand_cat.fit(Xtrain, ytrain)

import lightgbm as lgb
lgb_model = lgb.LGBMClassifier(random_state= 42, n_estimators=100, learning_rate=0.1, max_depth=10)

lgb_model.fit(Xtrain, ytrain, eval_set=[(Xtrain, ytrain), (Xtest, ytest)], eval_metric = 'AUC')
eval_result = lgb_model.evals_result_

with open('lgbm_acc.csv', 'w') as f:
    for acc in lgb_model.evals_result_['valid_1']['auc']:
        f.write("%s,\n"%(acc))

#y_pred = xgb.predict(Xtest)
#y_pred = cat_sm.predict(Xtest)
y_pred = lgb_model.predict(Xtest)

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