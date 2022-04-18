import pandas as pd
from ipdb import set_trace as bp
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from matplotlib import pyplot as plt
import joblib

data_orig = pd.read_csv('toddler_asd.csv')
data_orig.replace(to_replace={'Who completed the test':"Health care professional"}, value="Health Care Professional", inplace=True)
all_label_mappings = {}

# print(list(data_orig.columns.values))

def encode_categorical_column(column_name):
	data_orig[column_name] = data_orig[column_name].astype('category')
	all_label_mappings[column_name] = dict( enumerate(data_orig[column_name].cat.categories ) )
	data_orig[column_name] = data_orig[column_name].cat.codes

encode_categorical_column("Sex")  ## Female: 0   Male: 1
encode_categorical_column("Ethnicity")  ## 
encode_categorical_column("Jaundice")  # Yes: 0   No: 1
encode_categorical_column("Family_mem_with_ASD")  ##  No: 0  Yes: 1
encode_categorical_column("Who completed the test")  ## Family Member: 0  Health Care Professional: 1  Self: 2   Others: 3
encode_categorical_column("Class/ASD Traits ")  ## No: 0  Yes: 1
# data_orig["Sex"] = data_orig["Sex"].astype('category') 
# data_orig["Sex"] = data_orig["Sex"].cat.codes
# data_orig["Sex"].codes
# print(data_orig["Sex"].codes)
data_orig = data_orig.drop(columns='Qchat-10-Score')

X_data = data_orig.drop(columns='Class/ASD Traits ')
y_label = data_orig['Class/ASD Traits ']

all_keys = list(X_data.columns)

X_data = X_data.to_numpy()
y_label = y_label.to_numpy()

print(all_label_mappings)


X_train, X_test, y_train, y_test = train_test_split(X_data, y_label, test_size=0.2, random_state=42)

## Gradient Boosting Classifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=3, random_state=0).fit(X_train, y_train)
print("Gradient Boosting Classifier Score: ", clf.score(X_test, y_test))
joblib.dump(clf, 'gradient_boost.pkl', compress=9)

## Random Forest Classifier
forest = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0).fit(X_train, y_train)
print("Random Forest Classifier Score: ", forest.score(X_test, y_test))


joblib.dump(forest, 'forest.pkl', compress=9)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
forest_importances = pd.Series(importances, index=all_keys)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()


clf = DecisionTreeClassifier(max_depth = 4, random_state = 0).fit(X_train, y_train)
print("Decision Tree Classifier Score: ", clf.score(X_test, y_test))
# plt.figure()
# plt.figure(figsize=(20,12))
_, ax = plt.subplots(figsize=(14,8)) # Resize figure
tree.plot_tree(clf,filled=True,feature_names=all_keys, class_names=['normal','ASD'],ax=ax)




plt.show()
# plt.savefig("Decision_Tree_Classifier.pdf",dpi=100)
# bp()

