import category_encoders as ce
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split



def feature_categorization(dataset: pd.DataFrame) -> tuple:
    
    listOfFeatures: list = dataset.columns.tolist()
    listOfFeatures.remove("PatientID")
    listOfFeatures.remove("DoctorInCharge")
    
    numericalFeatures: list = []
    binaryFeatures: list = []
    categoricalFeatures: list = []
    targetFeature: list = []
    identifier: list = ["PatientID"]
    
    for eachFeature in listOfFeatures:

        if eachFeature == "Diagnosis":
            targetFeature.append("Diagnosis")
            
        elif set(pd.unique(dataset[eachFeature])).issubset({0, 1}):
            binaryFeatures.append(eachFeature)

        elif dataset[eachFeature].dtype == 'int64' or dataset[eachFeature].dtype == 'float64':

            if set(pd.unique(dataset[eachFeature])).issubset({0, 1, 2, 3}):
                categoricalFeatures.append(eachFeature)

            else:
                numericalFeatures.append(eachFeature)

        else:
            categoricalFeatures.append(eachFeature)

    return numericalFeatures, binaryFeatures, categoricalFeatures, targetFeature, identifier


def numerical_feature_importance_p_value(dataset: pd.DataFrame, numericalFeatures: list, targetFeature: str) -> dict:

    pValueDict: dict = {}

    for eachFeature in numericalFeatures:

        eachFeatureValue = dataset[eachFeature]
        targetFeatureValue = dataset[targetFeature]

        pValue = ttest_ind(eachFeatureValue[targetFeatureValue == 0], eachFeatureValue[targetFeatureValue == 1], equal_var=True, nan_policy='omit', random_state=6)[1]

        pValueDict[eachFeature] = float(pValue)
        
    return pValueDict


def numerical_feature_importance_random_forest(dataset: pd.DataFrame, numericalFeatures: list, targetFeature: str) -> pd.DataFrame:

    featureImportanceDict:dict = {}

    randomForest = RandomForestRegressor(n_estimators=100, random_state=6)
    randomForest.fit(dataset[numericalFeatures].values, dataset[targetFeature].values)

    featureImportance = randomForest.feature_importances_

    for eachFeature, eachImportance in zip(numericalFeatures, featureImportance):

        featureImportanceDict[eachFeature] = float(eachImportance)

    return featureImportanceDict


def numerical_feature_high_importance(numericalFeatures: list, featureImportanceDict: dict, minimumIndex: float) -> list:
    
    currentNumericalFeatures:list = []

    for feature in numericalFeatures:

        featureImportance = featureImportanceDict.get(feature)

        if featureImportance >= minimumIndex:
            currentNumericalFeatures.append(feature)
    
    return currentNumericalFeatures


def binary_feature_importance_random_forest(dataset: pd.DataFrame, binaryFeatures: list, targetFeature: str) -> dict:

    featureImportanceDict: dict = {}

    randomForest = RandomForestClassifier(n_estimators=100, random_state=6)
    randomForest.fit(dataset[binaryFeatures].values, dataset[targetFeature].values)

    featureImportance = randomForest.feature_importances_

    for eachFeature, eachImportance in zip(binaryFeatures, featureImportance):
        featureImportanceDict[eachFeature] = float(eachImportance)

    return featureImportanceDict


def binary_feature_high_importance(binaryFeatures: list, featureImportanceDict: dict, minimumIndex: float) -> list:

    currentBinaryFeatures: list = []

    for feature in binaryFeatures:
        featureImportance = featureImportanceDict.get(feature)

        if featureImportance >= minimumIndex:
            currentBinaryFeatures.append(feature)
    
    return currentBinaryFeatures


def categorical_feature_importance_random_forest(dataset: pd.DataFrame, categoricalFeatures: list, targetFeature: str) -> dict:

    featureImportanceDict: dict = {}

    randomForest = RandomForestClassifier(n_estimators=100, random_state=6)
    randomForest.fit(dataset[categoricalFeatures].values, dataset[targetFeature].values)

    featureImportance = randomForest.feature_importances_

    for eachFeature, eachImportance in zip(categoricalFeatures, featureImportance):

        featureImportanceDict[eachFeature] = float(eachImportance)

    return featureImportanceDict

def categorical_feature_high_importance(categoricalFeatures: list, featureImportanceDict: dict, minimumIndex: float) -> list:

    currentCategoricalFeatures: list = []

    for feature in categoricalFeatures:
        featureImportance = featureImportanceDict.get(feature)

        if featureImportance >= minimumIndex:
            currentCategoricalFeatures.append(feature)
    
    return currentCategoricalFeatures


def dataset_splitting(dataset: pd.DataFrame, testSize: float, trainSize: float, randomState: int, isShuffle: bool, targetFeature: list) -> tuple:

    splittedTrainSet: pd.DataFrame
    validationSet: pd.DataFrame
    
    splittedTrainSet, validationSet = train_test_split(dataset, test_size=testSize, train_size=trainSize, random_state=randomState,shuffle=isShuffle, stratify=dataset[targetFeature] if isShuffle else None)
    
    return splittedTrainSet, validationSet


def accuracy_prediction(trueValue: pd.Series, predictedValue: pd.Series) -> float:
        
    correctPrediction = np.sum(trueValue == predictedValue)
    totalPrediction = len(trueValue)
    accuracy = correctPrediction / totalPrediction

    return accuracy


def k_fold_MSE(dataset: pd.DataFrame, eachModel, modelClass, features: list, targetFeature: str, splitNumber: int, randomState: int) -> float:

    mseValueList: list = []

    kf = KFold(n_splits=splitNumber, shuffle=True, random_state=randomState)

    for trainIndex, validationIndex in kf.split(dataset):
        splittedTrainSet = dataset.iloc[trainIndex]
        validationSet = dataset.iloc[validationIndex]

        model = modelClass(splittedTrainSet, validationSet, features, targetFeature)
        if eachModel == "Random Forest":
            model.hyperparameter_tuning()
        model.fit_model()
        mse = model.validation_mean_squared_error()
        mseValueList.append(mse)

    return np.mean(mseValueList)


def k_fold_accuracy(dataset: pd.DataFrame, eachModel, modelClass, features: list, targetFeature: str, splitNumber: int, randomState: int) -> float:

    accuracyValueList: list = []

    kf = KFold(n_splits=splitNumber, shuffle=True, random_state=randomState)

    for train_index, val_index in kf.split(dataset):
        splittedTrainSet: pd.DataFrame = dataset.iloc[train_index]
        validationSet: pd.DataFrame = dataset.iloc[val_index]

        model = modelClass(splittedTrainSet, validationSet, features, targetFeature)
        if eachModel == "Random Forest":
            model.hyperparameter_tuning()
        model.fit_model()
        accuracy: float = model.validation_accuracy_model()
        accuracyValueList.append(accuracy)

    return np.mean(accuracyValueList)

def categorical_features_encoding(dataset: pd.DataFrame, categoricalFeatures: list, targetFeature: str) -> pd.DataFrame:

    target_encoder = ce.TargetEncoder(cols=categoricalFeatures)

    encoded_dataset = dataset.copy()
    
    encoded_dataset[categoricalFeatures] = target_encoder.fit_transform(dataset[categoricalFeatures], dataset[targetFeature])

    return encoded_dataset

def k_fold_cross_validator(splitNumber: int, shuffle: bool) -> KFold:

    kFoldCrossValidator = KFold(n_splits=splitNumber, shuffle=shuffle, random_state=6)

    return kFoldCrossValidator

def f1_score_prediction(trueValue: pd.Series, predictedValue: pd.Series) -> float:
    return f1_score(trueValue, predictedValue)