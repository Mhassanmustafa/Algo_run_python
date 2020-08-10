import numpy as np
import pandas as pd
import pickle
import Config as config

prediction_list = []


def make_userprediction(Pclass, gender, siblings, embarked, algorithum):

    user_input_df = pd.DataFrame(
        {
            'PClass': Pclass,
            'Sex': gender,
            'Sibling': siblings,
            'Embarked': embarked
        })

    # covert categorical attributes to numberical
    dataSet = pd.get_dummies(user_input_df, columns=["Sex", "Embarked"])
    indexes = [
        "PClass",
        "Sibling",
        "Sex_male",
        "Sex_female",
        "Embarked_C",
        "Embarked_S",
        "Embarked_Q",
    ]

    dataSet = dataSet.reindex(columns=indexes)
    X = dataSet

    print((X.fillna(0).astype(int)))

    if algorithum == 'Random Forest':
        prediction_list.clear()
        best_classifier = pickle.load(open(config.random_forest_path, 'rb'))
        prediction = best_classifier.predict((X.fillna(0).astype(int)))
        print(prediction)
        prediction_list.append(prediction)
    elif algorithum == 'Multi layer perceptron':
        prediction_list.clear()
        best_classifier = pickle.load(open(config.perceptron_path, 'rb'))
        prediction = best_classifier.predict((X.fillna(0).astype(int)))
        print(prediction)
        prediction_list.append(prediction)
    elif algorithum == 'K neraest Neighbour':
        prediction_list.clear()
        best_classifier = pickle.load(open(config.knn_path, 'rb'))
        prediction = best_classifier.predict((X.fillna(0).astype(int)))
        prediction_list.append(prediction)
        print(prediction)
    elif algorithum == 'Nave base':
        prediction_list.clear()
        best_classifier = pickle.load(open(config.bernoli_path, 'rb'))
        prediction = best_classifier.predict((X.fillna(0).astype(int)))
        prediction_list.append(prediction)
        print(prediction)
    elif algorithum == 'Linear SVC':
        prediction_list.clear()
        best_classifier = pickle.load(open(config.linear_svc_path, 'rb'))
        prediction = best_classifier.predict(X.fillna(0).astype(int))
        print(prediction)
        prediction_list.append(prediction)
