import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn import  metrics 

# result1=pd.read_csv('train_distance_times_days_lat_long_missing_mean.csv')
result1=pd.read_csv('train_distance_time_day_lat_long_missing_mean.csv')
# result5=pd.read_csv('test_distance.csv')
# result5=pd.read_csv('test_distance_times_days_lat_long.csv')
result5=pd.read_csv('test_distance_time_day_lat_long.csv')
test=result5.drop(['tripid','pickup_time','drop_time'],axis=1)
x=result1.drop(['tripid','pickup_time','drop_time','label'],axis=1)
y=result1['label']
codes={'correct':1, 'incorrect':0}
y=y.map(codes)
x_train, x_test,y_train, y_test=train_test_split(x,y,test_size=0.20,random_state=42)


xgb_clf = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=9,
 min_child_weight=1,
 gamma=0.2,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27,
 reg_alpha=1e-05)

xgb_param = xgb_clf.get_xgb_params()
xgtrain = xgb.DMatrix(x_train, label=y_train)
cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb_clf.get_params()['n_estimators'], nfold=5,
                          metrics='auc', early_stopping_rounds=50)
xgb_clf.set_params(n_estimators=cvresult.shape[0])
xgb_clf.fit(x_train, y_train)
y_pred_xgb=xgb_clf.predict(x_test)
y_pred_xgb_test_data=xgb_clf.predict(test)
score = accuracy_score(y_test, y_pred_xgb)
f1_score_xgboost=f1_score(y_test,y_pred_xgb)

print(cvresult.shape[0])


print(
    "\nModel Report")
print(
    "Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred_xgb))
print(
    "auc Score (Train): %f" % metrics.roc_auc_score(y_test, y_pred_xgb))

print('xgboost_accuracy: ',score)
# print('xgboost_score_accuracy: ',score1)
print('xgboost_f1 score: ',f1_score_xgboost)
#
dict1={'tripid': result5['tripid'],'prediction':y_pred_xgb_test_data}
df1=pd.DataFrame(dict1)
df1.to_csv('submitted/xgboost_distance_day_time_lat_long_mean_auc.csv',index=False)