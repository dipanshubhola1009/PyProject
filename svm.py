import pandas as pd
from sklearn import svm # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_validate
# from sklearn.model_selection import KFold
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import svm
from sklearn import metrics
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.tree import plot_tree

import os
for dirname, _, filenames in os.walk('/PycharmProjects/MlProject'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

input_dir = '/'

df_test = pd.read_csv('/home/dipanshu/PycharmProjects/MlProject/KDDTest+.txt')
columns = (['duration'
,'protocol_type'
,'service'
,'flag'
,'src_bytes'
,'dst_bytes'
,'land'
,'wrong_fragment'
,'urgent'
,'hot'
,'num_failed_logins'
,'logged_in'
,'num_compromised'
,'root_shell'
,'su_attempted'
,'num_root'
,'num_file_creations'
,'num_shells'
,'num_access_files'
,'num_outbound_cmds'
,'is_host_login'
,'is_guest_login'
,'count'
,'srv_count'
,'serror_rate'
,'srv_serror_rate'
,'rerror_rate'
,'srv_rerror_rate'
,'same_srv_rate'
,'diff_srv_rate'
,'srv_diff_host_rate'
,'dst_host_count'
,'dst_host_srv_count'
,'dst_host_same_srv_rate'
,'dst_host_diff_srv_rate'
,'dst_host_same_src_port_rate'
,'dst_host_srv_diff_host_rate'
,'dst_host_serror_rate'
,'dst_host_srv_serror_rate'
,'dst_host_rerror_rate'
,'dst_host_srv_rerror_rate'
,'attack'
,'level'])

df_test.columns = columns

df_test.head()
is_attack = df_test.attack.map(lambda a: 0 if a == 'normal' else 1)
df_test['class'] = is_attack
# print(df_test['class'])
df_test.groupby('class').size()

# print(df_test)
var_columns = [c for c in df_test.columns if c not in ['protocol_type', 'service', 'flag', 'attack' ,'class']]
X = df_test.loc[:,var_columns]
y = df_test.loc[:,'class']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_train)
# print(y_train)
# print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

clf = svm.SVC(gamma="auto")
print('error gone')
clf.fit(X_train,y_train)
print('fit')

y_pred = clf.predict(X_valid)
y_train_pred = clf.predict(X_train)



print("Accuracy:",metrics.accuracy_score(y_train_pred, y_train))
print("Precision:",metrics.precision_score(y_valid, y_pred))

auc_train = metrics.accuracy_score(y_valid, y_pred)
auc_valid = metrics.precision_score(y_valid, y_pred)

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_valid, y_pred))

print("AUC Train = {}\nAUC Valid = {}".format(round(auc_train,4), round(auc_valid,4)))


# def make_meshgrid(x, y, h=.02):
#     x_min, x_max = x.min() - 1, x.max() + 1
#     y_min, y_max = y.min() - 1, y.max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#     return xx, yy

# def plot_contours(ax, clf, xx, yy, **params):
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     out = ax.contourf(xx, yy, Z, **params)
#     return out



# fig, ax = plt.subplots()
# # title for the plots
# title = ('Decision surface of linear SVC ')
# # Set-up grid for plotting.
# X0, X1 = X[:, 0], X[:, 1]
# xx, yy = make_meshgrid(X0, X1)

# plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
# ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
# ax.set_ylabel('y label here')
# ax.set_xlabel('x label here')
# ax.set_xticks(())
# ax.set_yticks(())
# ax.set_title(title)
# ax.legend()
# plt.show()
print('end')