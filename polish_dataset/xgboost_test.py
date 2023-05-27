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
d1= arff.loadarff('../5year.arff')
df = pd.DataFrame(d1[0])
df.drop_duplicates(inplace=True)
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr=imr.fit(df)
imputed_data=imr.transform(df.values)
imputed_data_df=pd.DataFrame(imputed_data)
df=pd.DataFrame(imputed_data_df.values, columns=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', 'Class'])

# scaler = MinMaxScaler()
# sampled_df = scaler.fit_transform(df)
# sampled_df = pd.DataFrame(sampled_df, columns=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', 'Class'])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

from imblearn.over_sampling import SMOTE
sm = SMOTE()
X, y = sm.fit_resample(X, y)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state = 42)

# from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
# xgb_param_grid={"n_estimators":[100],"max_depth":[10],"learning_rate":[0.30],"min_child_weight":[7],
#                 'alpha':[0.01],'scale_pos_weight':[10],'colsample_bytree':[1]}
# xgb = XGBClassifier(objective="binary:logistic", eval_metric="auc", random_state=42)
# xgb.fit(Xtrain,ytrain, eval_set=[(Xtrain, ytrain), (Xtest, ytest)])

# eval_result = xgb.evals_result()

# with open('xgboost_acc.csv', 'w') as f:
#     for acc in xgb.evals_result()['validation_0']['auc']:
#         f.write("%s,\n"%(acc))


from catboost import CatBoostClassifier
cat_sm = CatBoostClassifier(verbose = 0, eval_metric='AUC', random_seed = 2, iterations = 100)

cat_params = {'eval_metric': ['F1'],
                'iterations': [100],
                'learning_rate' : [0.1],
                'random_seed' : [42],
                'max_depth': [10],
                'auto_class_weights' : ['Balanced'],
                'verbose': [True]}

cat_sm.fit(Xtrain, ytrain, eval_set=[(Xtrain, ytrain), (Xtest, ytest)])
eval_result = cat_sm.get_evals_result()

with open('catboost_acc.csv', 'w') as f:
    for acc in cat_sm.get_evals_result()['validation_0']['AUC']:
        f.write("%s,\n"%(acc))

rand_cat = GridSearchCV(cat_sm, cat_params)
rand_cat.fit(Xtrain, ytrain)

# import lightgbm as lgb
# lgb_model = lgb.LGBMClassifier()

# # import os
# # import inspect
# # print(os.path.abspath(inspect.getfile(lgb.LGBMClassifier)))

# lgb_model.fit(Xtrain, ytrain, eval_set=[(Xtrain, ytrain), (Xtest, ytest)], eval_metric='AUC')
# eval_result = lgb_model.evals_result_

# with open('lgbm_acc.csv', 'w') as f:
#     for acc in lgb_model.evals_result_['training']['auc']:
#         f.write("%s,\n"%(acc))

#y_pred = xgb.predict(Xtest)
y_pred = rand_cat.predict(Xtest)
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