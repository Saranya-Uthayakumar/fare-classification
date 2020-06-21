import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn import  metrics
from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import NearMiss 

result1=pd.read_csv('train_distance_times_days_lat_long_missing_mean.csv')
# result5=pd.read_csv('test_distance.csv')
result5=pd.read_csv('test_distance_times_days_lat_long.csv')
test=result5.drop(['tripid','pickup_time','drop_time'],axis=1)
x=result1.drop(['tripid','pickup_time','drop_time','label'],axis=1)
y=result1['label']
codes={'correct':1, 'incorrect':0}
y=y.map(codes)
x_train, x_test,y_train, y_test=train_test_split(x,y,test_size=0.20,random_state=42)
# scaler = StandardScaler()
# x_train_scaled=scaler.fit_transform(x_train)
# x_test_scaled=scaler.fit_transform(x_test)


sm = SMOTE(random_state = 2) 
x_train_smote, y_train_smote = sm.fit_sample(x_train, y_train) 


nr = NearMiss() 
x_train_near_miss, y_train_near_miss = nr.fit_sample(x_train, y_train) 


log_regression = LogisticRegression(solver='lbfgs')
# log_regression_smote = LogisticRegression(solver='lbfgs')
# log_regression_near_miss = LogisticRegression(solver='lbfgs')
log_regression.fit(x_train,y_train)
# log_regression_smote.fit(x_train_smote,y_train_smote)
# log_regression_near_miss.fit(x_train_near_miss,y_train_near_miss)
y_pred_log_regression=log_regression.predict(x_test)
# y_pred_log_regression_smote=log_regression_smote.predict(x_test)
# y_pred_log_regression_near_miss=log_regression_near_miss.predict(x_test)
y_predict_log_regression_test_data=log_regression.predict(test)
# y_predict_log_regression_smote_test_data=log_regression_smote.predict(test)
# y_predict_log_regression_near_miss_test_data=log_regression_near_miss.predict(test)
accuracy_log_regression=accuracy_score(y_test,y_pred_log_regression)
# accuracy_log_regression_smote=accuracy_score(y_test,y_pred_log_regression_smote)
# accuracy_log_regression_near_miss=accuracy_score(y_test,y_pred_log_regression_near_miss)
f1_score_log_regression=f1_score(y_test,y_pred_log_regression)
# f1_score_log_regression_smote=f1_score(y_test,y_pred_log_regression_smote)
# f1_score_log_regression_near_miss=f1_score(y_test,y_pred_log_regression_near_miss)
print('accuracy_log_regression:', accuracy_log_regression)
# print('accuracy_log_regression:', accuracy_log_regression_smote)
# print('accuracy_log_regression:', accuracy_log_regression_near_miss)
print('f1 score_log_regression:', f1_score_log_regression)
# print('f1 score_log_regression:', f1_score_log_regression_smote)
# print('f1 score_log_regression:', f1_score_log_regression_near_miss)


dict1={'tripid': result5['tripid'],'prediction':y_predict_log_regression_test_data}
df1=pd.DataFrame(dict1)
df1.to_csv('submitted/logistric_regression_distance_day_time_lat_long_mean_auc.csv',index=False)