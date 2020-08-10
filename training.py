import pandas as pd
import numpy as np
import Config as config
import matplotlib.pyplot as plt
import warnings
from prettytable import PrettyTable
import pickle


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

# read csv files
dataSet = pd.read_csv(config.dataSetPath)


# print full dataset
print("Dataset:\n===================\n")
dataSet.index.names = ['index']
print(dataSet)


# printing attributes in train dataset
print('\n\nAttributes Names in Train Dataset:\n==================================\n')
print(dataSet.columns)


print('\n\nNumber of instances in Train Dataset:'
      '\n======================================'
      '\nTrain Data instances: ', dataSet.shape[0])

# drop passenger id becacuse of no use
dataSet = dataSet.drop("PassengerId", axis=1)

label = "Survived"
y = dataSet[label]
X = dataSet.drop(label, axis=1)
train_attributes, test_attributr, train_output,  test_output = train_test_split(
    X, y, train_size=0.8)

print(train_output)


# original train data
train_attributes['Survived'] = train_output

print('\n\nTraining Data:\n==================================\n')
print(train_attributes)


# Original test data
test_attributr['Survived'] = test_output
print('\n\nTesting Data:\n==================================\n')
print(test_attributr)


# preprocess the data covert the categorical data into the numeric data
dataSet = pd.get_dummies(dataSet, columns=["Sex", "Embarked"])

print('\n\ndata set after encoding:\n==================================\n')
print(dataSet)
# fix the new indexes
indexes = [
    "PClass",
    "Sibling",
    "Sex_male",
    "Sex_female",
    "Embarked_C",
    "Embarked_S",
    "Embarked_Q",
    "Survived",
]


dataSet = dataSet.reindex(columns=indexes)

label = "Survived"
y = dataSet[label]
X = dataSet.drop(label, axis=1)

# split the data into the training and testing phase 80% training and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)


# ploting number of survive in bar chart
print('Total number of \'Survive people\' and \'un Survive\' in Train Dataset:'
      '\n================================================================================\n')

y_train.value_counts().plot.bar(color=['red', 'green'])
plt.xlabel('Survied')
plt.legend(['Frequency'], loc='lower center')
warnings.filterwarnings("ignore")
plt.show()

# # apply machine learning algo from sciket to train the given data

model_list = []
Acuraccy_score_list = []

# use random forest model
print('\n random forest algorithum\n======================================\n')
random_forest_classifier = RandomForestClassifier()
model_list.append('RandomForest')

# fitting this model
random_forest_classifier.fit(X_train, y_train)

y_prediction = random_forest_classifier.predict(X_test)

# show output come on testing
test_dataSet_prediction = test_attributr.drop("Survived", axis=1)
test_dataSet_prediction['Survived'] = y_prediction

print('\nPredictive data\n===================================\n')
print(test_dataSet_prediction)

Accuracy_score = round(random_forest_classifier.score(X_test, y_test), 2)

print('\n\nAccuracy score : \n'
      '===============\n', Accuracy_score)

Acuraccy_score_list.append(Accuracy_score)


# initializing the multilayer perceptron
print('\nMulti layer perceptron algorithum\n======================================\n')
multi_layer_perceptron = MLPClassifier(
    solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)


model_list.append('Multi_layer_perceptron')

# fitting the model
multi_layer_perceptron.fit(X_train, y_train)

perceptron_prediction = multi_layer_perceptron.predict(X_test)

# show output come on testing
test_dataSet_prediction = test_attributr.drop("Survived", axis=1)
test_dataSet_prediction['Survived'] = perceptron_prediction

print('\nPredictive data\n===================================\n')
print(test_dataSet_prediction)
Accuracy_score = round(multi_layer_perceptron.score(X_test, y_test), 2)

print('\n\nAccuracy score : \n'
      '===============\n', Accuracy_score)

Acuraccy_score_list.append(Accuracy_score)


# initializing the Knn Algorithum
print('\nKnn algorithum\n======================================\n')

knn = NearestCentroid()

model_list.append('Knn')

knn.fit(X_train, y_train)

knn_prediction = knn.predict(X_test)


# show output come on testing
test_dataSet_prediction = test_attributr.drop("Survived", axis=1)
test_dataSet_prediction['Survived'] = knn_prediction

print('\nPredictive data\n===================================\n')
print(test_dataSet_prediction)
Accuracy_score = round(accuracy_score(y_test, knn_prediction), 2)

print('\n\nAccuracy score : \n'
      '===============\n', Accuracy_score)

Acuraccy_score_list.append(Accuracy_score)


# initializing the Nave base Algorithum
print('\nNave base algorithum\n======================================\n')
# BernoulliNB
bernoulli_nb = BernoulliNB()

# fitting this model
bernoulli_nb.fit(X_train, y_train)
model_list.append('BernoulliNB')

bernouli_prediction = bernoulli_nb.predict(X_test)


# show output come on testing
test_dataSet_prediction = test_attributr.drop("Survived", axis=1)
test_dataSet_prediction['Survived'] = bernouli_prediction


print('\nPredictive data\n===================================\n')
print(test_dataSet_prediction)
Accuracy_score = round(accuracy_score(y_test, bernouli_prediction), 2)

print('\n\nAccuracy score : \n'
      '===============\n', Accuracy_score)

Acuraccy_score_list.append(Accuracy_score)

# initializing the SVC Algorithum
print('\nLinera SVC algorithum\n======================================\n')

# LinearSVC
linear_svc = LinearSVC()

# fitting this model
linear_svc.fit(X_train, y_train)
model_list.append('LinearSVC')

liner_prediction = linear_svc.predict(X_test)


# show output come on testing
test_dataSet_prediction = test_attributr.drop("Survived", axis=1)
test_dataSet_prediction['Survived'] = liner_prediction

print('\nPredictive data\n===================================\n')
print(test_dataSet_prediction)
Accuracy_score = round(accuracy_score(y_test, liner_prediction), 2)

print('\n\nAccuracy score : \n'
      '===============\n', Accuracy_score)

Acuraccy_score_list.append(Accuracy_score)


# Selection of best model
print('\nDetailed Performance of all the models.\n'
      '=======================================\n\n')

prettyTable1 = PrettyTable(['Model', 'Accuracy Score'])

for model_names, scores in zip(model_list, Acuraccy_score_list):
    prettyTable1.add_row([model_names, scores])

print(prettyTable1)


# save the training in the storage for the future use

pickle.dump(random_forest_classifier, open(config.random_forest_path, 'wb'))
pickle.dump(multi_layer_perceptron, open(
    config.perceptron_path, 'wb'))
pickle.dump(knn, open(config.knn_path, 'wb'))
pickle.dump(bernoulli_nb, open(config.bernoli_path, 'wb'))
pickle.dump(linear_svc, open(config.linear_svc_path, 'wb'))
