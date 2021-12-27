#%%
from numpy.lib.function_base import average
import pandas as pd
import numpy as np
from six import print_
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot

dataset = pd.read_csv('./abalone.data')
dataset.columns = ['Gender', 'Length', 'Diameter', 'Height', 'Whole Weight', 
                   'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Rings']

_label = preprocessing.LabelEncoder()
dataset['Gender'] = _label.fit_transform(dataset['Gender'])
print("First 8 data\n", dataset.head(8))
print("Last 5 data\n", dataset.tail(5))
print("Checking data set\n" ,dataset.isnull().sum(axis = 0))
print("Describtion of the data\n" ,dataset.describe())
print("column names\n", dataset.columns)
print("The shape of the data\n", dataset.shape)
X = dataset[['Gender', 'Length', 'Diameter', 'Height', 'Whole Weight', 
                   'Shucked Weight', 'Viscera Weight', 'Shell Weight']]
X.columns = ['Gender', 'Length', 'Diameter', 'Height', 'Whole Weight', 
                   'Shucked Weight', 'Viscera Weight', 'Shell Weight']
Y = dataset['Rings']
dataset.groupby('Rings').hist(figsize=(9,9))
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.25, random_state=1)

DecisionTree = DecisionTreeClassifier(criterion = "entropy", max_depth =3)
DecisionTree.fit(X_train, Y_train)
y_pred = DecisionTree.predict(X_test)


print("Decision Tree Accuracy Score is: ", metrics.accuracy_score(Y_test,y_pred))
print("Decision Tree F1 Score is: ", metrics.f1_score(Y_test,y_pred, average='macro'))
print("Decision Tree Recall Score is: ", metrics.recall_score(Y_test,y_pred, average = 'macro'))
print("Decision Tree Precision Score is: ", metrics.precision_score(Y_test,y_pred, average = 'macro'))
print("Decision Tree Confusion Score is: ", metrics.confusion_matrix(Y_test,y_pred))

cnf_matrix = confusion_matrix(Y_test, y_pred)
p = sns.heatmap(cnf_matrix, annot = True, cmap = 'YlGnBu', fmt = 'g')
plt.title('Decision Tree Confusion matrix', y=1.1)
plt.ylabel('Acutal')
plt.xlabel('Predicted')
plt.show()

Bayes = GaussianNB()
Bayes.fit(X_train,Y_train)
Bayes_pred = Bayes.predict(X_test)

print("\nBayes Accuracy Score is: ", metrics.accuracy_score(Y_test,Bayes_pred))
print("Bayes Accuracy Score is: ", metrics.accuracy_score(Y_test,Bayes_pred))
print("Bayes F1 Score is: ", metrics.f1_score(Y_test,Bayes_pred, average='macro'))
print("Bayes Recall Score is: ", metrics.recall_score(Y_test,Bayes_pred, average = 'macro'))
print("Bayes Precision Score is: ", metrics.precision_score(Y_test,Bayes_pred, average = 'macro'))
print("Bayes Confusion Score is: ", metrics.confusion_matrix(Y_test,Bayes_pred))

cnf_matrix = confusion_matrix(Y_test, Bayes_pred)
p = sns.heatmap(cnf_matrix, annot = True, cmap = 'YlGnBu', fmt = 'g')
plt.title('Bayes Confusion matrix', y=1.1)
plt.ylabel('Acutal')
plt.xlabel('Predicted')
plt.show()

Knn = KNeighborsClassifier(n_neighbors = 3)
Knn.fit(X_train, Y_train)
Knn_pred = Knn.predict(X_test)

print("\nKnn Accuracy Score is: ", metrics.accuracy_score(Y_test,Knn_pred))
print("Knn Accuracy Score is: ", metrics.accuracy_score(Y_test,Knn_pred))
print("Knn F1 Score is: ", metrics.f1_score(Y_test,Knn_pred, average='macro'))
print("Knn Recall Score is: ", metrics.recall_score(Y_test,Knn_pred, average = 'macro'))
print("Knn Precision Score is: ", metrics.precision_score(Y_test,Knn_pred, average = 'macro'))
print("Knn Confusion Score is: ", metrics.confusion_matrix(Y_test,Knn_pred))

cnf_matrix = confusion_matrix(Y_test, Knn_pred)
p = sns.heatmap(cnf_matrix, annot = True, cmap = 'YlGnBu', fmt = 'g')
plt.title('Knn Confusion matrix', y=1.1)
plt.ylabel('Acutal')
plt.xlabel('Predicted')
plt.show()

Svc = svm.SVC(kernel='linear', C = 1, gamma = 'auto').fit(X_train, Y_train)
Svc_pred = Svc.predict(X_test)

print("\nSVC Accuracy Score is: ", metrics.accuracy_score(Y_test,Svc_pred))
print("SVC F1 Score is: ", metrics.f1_score(Y_test,Svc_pred, average='macro'))
print("SVC Recall Score is: ", metrics.recall_score(Y_test,Svc_pred, average = 'macro'))
print("SVC Precision Score is: ", metrics.precision_score(Y_test,Svc_pred, average = 'macro'))
print("SVC Confusion Score is: ", metrics.confusion_matrix(Y_test,Svc_pred))

cnf_matrix = confusion_matrix(Y_test, Svc_pred)
p = sns.heatmap(cnf_matrix, annot = True, cmap = 'YlGnBu', fmt = 'g')
plt.title('SVC Confusion matrix', y=1.1)
plt.ylabel('Acutal')
plt.xlabel('Predicted')
plt.show()


k = 5
kf = KFold(n_splits=k, random_state=None)
kflodDecision = DecisionTreeClassifier(criterion='entropy', max_depth=3)
accuracy_score = []
f1_score = []
recall_score = []
precision_score = []
confusion_score = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    kflodDecision.fit(X_train, Y_train)
    predicted_values = kflodDecision.predict(X_test)
    
    accuracy_score.append(metrics.accuracy_score(Y_test, predicted_values))
    f1_score.append(metrics.f1_score(Y_test, predicted_values, average = "macro"))
    recall_score.append(metrics.recall_score(Y_test, predicted_values, average = "macro"))
    precision_score.append(metrics.precision_score(Y_test, predicted_values, average = "macro"))
    
average_accuracy_score = sum(accuracy_score)/k
print("Accuracy of each flod is: {}".format(accuracy_score))
print("Average accuracy is: {}".format(average_accuracy_score))
average_f1_score = sum(f1_score)/k
print("F1 of each flod is {}:".format(f1_score))
print("Average F1 Score is: {}".format(average_f1_score))
average_recall_score = sum(recall_score)/k
print("Recall of each flod is: {}".format(recall_score))
print("Average recall score is: {}".format(average_recall_score))
average_precision_score = sum(precision_score)/k
print("Precision of each flod is: {}".format(precision_score))
print("Average precision score is: {}".format(average_precision_score))

# %%
