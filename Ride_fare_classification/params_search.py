import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

result1=pd.read_csv('train_distance_days_times_lat_long_missing_mean.csv')
x=result1.drop(['tripid','pickup_time','drop_time','label'],axis=1)
y=result1['label']
codes={'correct':1, 'incorrect':0}
y=y.map(codes)
x_train, x_test,y_train, y_test=train_test_split(x,y,test_size=0.20,random_state=42)

# param_test0 = {
#  'scale_pos_weight':[13,14,12,1]
# }
# gsearch0 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=165, max_depth=5,
#  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
#  objective= 'binary:logistic', nthread=2, scale_pos_weight=1, seed=27),
#  param_grid = param_test0, scoring='f1',n_jobs=4,iid=False, cv=5)
# gsearch0.fit(x_train,y_train)
# print(gsearch0.best_params_, gsearch0.best_score_)

param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(x_train,y_train)
print(gsearch1.best_params_, gsearch1.best_score_)

param_test1a = {
 'max_depth':[8,9,10],
 'min_child_weight':[1,2,3,4]
}
gsearch1a = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=165, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
 param_grid = param_test1a, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1a.fit(x_train,y_train)
print(gsearch1a.best_params_, gsearch1a.best_score_)

param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=9,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
 param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(x_train,y_train)
print(gsearch3.best_params_, gsearch3.best_score_)

param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=9,
 min_child_weight=2, gamma=0.0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
 param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch4.fit(x_train,y_train)
print(gsearch4.best_params_, gsearch4.best_score_)

param_test5 = {
 'colsample_bytree':[i/100.0 for i in range(75,90,5)],
 'subsample':[i/100.0 for i in range(75,90,5)]
}
gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=9,
 min_child_weight=2, gamma=0.0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
 param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch5.fit(x_train,y_train)
print(gsearch5.best_params_, gsearch5.best_score_)

param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=9,
 min_child_weight=2, gamma=0.0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
 param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch6.fit(x_train,y_train)
print(gsearch6.best_params_, gsearch6.best_score_)

param_test7 = {
 'reg_alpha':[0.01, 0.1,0.005,0.5,0.002]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=6,
 min_child_weight=2, gamma=0.1, subsample=0.75, colsample_bytree=0.75,
 objective= 'binary:logistic', nthread=2, scale_pos_weight=0.07,seed=27),
 param_grid = param_test7, scoring='f1',n_jobs=4,iid=False, cv=5)
gsearch6.fit(x_train,y_train)
print(gsearch6.best_params_, gsearch6.best_score_)