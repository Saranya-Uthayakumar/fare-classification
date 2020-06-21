import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


result1=pd.read_csv('train_distance_times_days_lat_long_missing_mean.csv')
# result5=pd.read_csv('test_distance.csv')
result5=pd.read_csv('test_distance_times_days_lat_long.csv')
test=result5.drop(['tripid','pickup_time','drop_time'],axis=1)
x=result1.drop(['tripid','pickup_time','drop_time','label'],axis=1)
y=result1['label']
codes={'correct':1, 'incorrect':0}
y=y.map(codes)
x_train, x_test,y_train, y_test=train_test_split(x,y,test_size=0.20,random_state=42)

decision_tree_gini = DecisionTreeClassifier(criterion="gini",
                                  random_state=100, max_depth=3, min_samples_leaf=5)
decision_tree_gini.fit(x_train, y_train)
y_decision_tree_gini=decision_tree_gini.predict(x_test)
accuracy_decision_tree_gini=accuracy_score(y_test,y_decision_tree_gini)
f1_score_decision_tree_gini=f1_score(y_test,y_decision_tree_gini)
print('accuracy_decision_tree_gini:', accuracy_decision_tree_gini)
print('f1_score_decision_tree_gini:', f1_score_decision_tree_gini)

decision_tree_entropy = DecisionTreeClassifier(
    criterion="entropy", random_state=100,
    max_depth=3, min_samples_leaf=5)
decision_tree_entropy.fit(x_train, y_train)
y_decision_tree_entropy=decision_tree_entropy.predict(x_test)
accuracy_decision_tree_entropy=accuracy_score(y_test,y_decision_tree_entropy)
f1_score_decision_tree_entropy=f1_score(y_test,y_decision_tree_entropy)
print('accuracy_decision_tree_entropy:', accuracy_decision_tree_entropy)
print('f1_score_decision_tree_entropy:', f1_score_decision_tree_entropy)

dict1={'tripid': result5['tripid'],'prediction':y_decision_tree_gini}
df1=pd.DataFrame(dict1)
df1.to_csv('submitted/decision_tree_distance_day_time_lat_long_mean_auc.csv',index=False)