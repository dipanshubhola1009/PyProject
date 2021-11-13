import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

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

model_tree = DecisionTreeClassifier(max_leaf_nodes=8, class_weight='balanced')
model_tree.fit(X_train, y_train)

plt.figure(figsize=(7,4))

#Create the tree plot
plot_tree(model_tree,
           feature_names = var_columns, #Feature names
           class_names = ["0","1"], #Class names
           rounded = True,
           filled = True)

plt.show()

y_train_pred = model_tree.predict(X_train)
y_valid_pred = model_tree.predict(X_valid)

auc_train = metrics.roc_auc_score(y_train, y_train_pred)
auc_valid = metrics.roc_auc_score(y_valid, y_valid_pred)



print("AUC Train = {}\nAUC Valid = {}".format(round(auc_train,4), round(auc_valid,4)))

