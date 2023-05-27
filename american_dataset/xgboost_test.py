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
df = pd.read_csv("american_bankruptcy_dataset.csv")
df = df.groupby("company_name").last().reset_index()

df.loc[:, 'status_label'] = df.loc[:, 'status_label'].eq('failed').mul(1)
training_df = df[(df['year'] == 2015) | (df['year'] == 2016) | (df['year'] == 2017)]

training_df = training_df.drop(['year', 'company_name'], axis=1)

# sampled_df = pd.DataFrame(df.iloc[:, 0])
# sampled_df['X2'] = df['X17']/df['X10']
# sampled_df['X6'] = df['X15']/df['X10']
# sampled_df['X7'] = df['X12']/df['X10']
# sampled_df['X9'] = df['X9']/df['X10']
# sampled_df['X10'] = df['X8']/df['X10']
# sampled_df['X17'] = df['X10']/df['X17']
# sampled_df['X18'] = df['X13']/df['X10']
# sampled_df['X19'] = df['X13']/df['X9']
# sampled_df['X29'] = np.log(df['X10'])
# sampled_df['X34'] = df['X18']/df['X17']
# sampled_df['X36'] = df['X9']/df['X10']

# bankrupt_df = df.loc[df["status_label"] == 1]
# non_bankrupt_df = df.loc[df["status_label"] == 0]
# bankrupt_df = bankrupt_df.sample(n = 1000, random_state=42)
# non_bankrupt_df = non_bankrupt_df.sample(n = 10000, random_state=42)
# sampled_df = pd.concat([bankrupt_df, non_bankrupt_df], axis=0, ignore_index=True)

y = training_df.iloc[:, 0]

X = training_df.iloc[:, 1:]
#X = (X-X.mean())/X.std()

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)

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

# from catboost import CatBoostClassifier
# cat_sm = CatBoostClassifier(eval_metric='AUC', learning_rate=0.1, iterations=100, random_seed=42, max_depth=10, auto_class_weights='Balanced', verbose=True)

# cat_sm.fit(Xtrain, ytrain, eval_set=[(Xtrain, ytrain), (Xtest, ytest)])
# eval_result = cat_sm.get_evals_result()

# with open('catboost_acc.csv', 'w') as f:
#     for acc in cat_sm.get_evals_result()['validation_0']['AUC']:
#         f.write("%s,\n"%(acc))

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