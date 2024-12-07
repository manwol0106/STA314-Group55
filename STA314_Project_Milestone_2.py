import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier

# NOTE IMPORTANT: The current .py file is our second milestone.
# NOTE IMPORTANT: We also have two .py files that contain functions and classes.


# Importing all classes from STA314_Project_Class.py.
from STA314_Project_Class import BinaryFeatureLDAModel, BinaryFeatureLogisticRegressionModel, BinaryFeatureRandomForestModel, BinaryFeatureSVMModel
from STA314_Project_Class import CategoricalFeatureLDAModel, CategoricalFeatureLogisticRegressionModel, CategoricalFeatureRandomForestModel, CategoricalFeatureSVMModel, xgb
from STA314_Project_Class import  NumericalFeatureBayesianRegressionModel, NumericalFeatureLinearRegressionModel, NumericalFeaturePolynomialRegressionModel, NumericalFeatureRandomForestModel


# Importing all functions from STA314_Project_Function.py.
from STA314_Project_Function import binary_feature_high_importance, binary_feature_importance_random_forest
from STA314_Project_Function import categorical_feature_high_importance, categorical_feature_importance_random_forest
from STA314_Project_Function import categorical_features_encoding, feature_categorization, f1_score_prediction, k_fold_accuracy, k_fold_cross_validator, k_fold_MSE
from STA314_Project_Function import numerical_feature_high_importance, numerical_feature_importance_p_value, numerical_feature_importance_random_forest

# Load the dataset.
trainingDataset = pd.read_csv("train.csv")

# Defining feature categories.
numericalFeatures, binaryFeatures, categoricalFeatures, targetFeature, identifier = feature_categorization(trainingDataset)

# Analyzing the significance of each feature in numericalFeatures by p value.
print(numerical_feature_importance_p_value(trainingDataset, numericalFeatures, targetFeature[0]))

# Analyzing the significance of each feature in numericalFeatures by Random Forest.
print(numerical_feature_importance_random_forest(trainingDataset, numericalFeatures, targetFeature[0]))

# Analyzing the significance of each feature in binaryFeatures by Random Forest.
print(binary_feature_importance_random_forest(trainingDataset, binaryFeatures, targetFeature[0]))

# Analyzing the significance of each feature in categoricalFeatures by Random Forest.
print(categorical_feature_importance_random_forest(trainingDataset, categoricalFeatures, targetFeature[0]))


## Fitting numerical features using different models (Bayesian Regression, Linear Regression, Polynomial Regression, Random Forest).
## Evaluating performance using Mean Squared Error (MSE) with K-Fold Cross-Validation.
print("\nFitting numerical features using different models (Bayesian Regression, Linear Regression, Polynomial Regression, Random Forest).")
print("Evaluating performance using Mean Squared Error (MSE) with K-Fold Cross-Validation.")

# Iterate different minimumIndex combinations of numerical features.
bestNumericalIndex: float = 0
bestNumericalModel = None
lowestMse: float = float('inf')
splitNumber: int = 5

for minimumIndex in np.arange(0, 0.05, 0.005):

    currentNumericalFeatures = numerical_feature_high_importance(numericalFeatures, numerical_feature_importance_random_forest(trainingDataset, numericalFeatures, targetFeature[0]), minimumIndex=minimumIndex)

    allNumericalModels = [
        ("Bayesian Regression", NumericalFeatureBayesianRegressionModel),
        ("Linear Regression", NumericalFeatureLinearRegressionModel),
        ("Polynomial Regression", NumericalFeaturePolynomialRegressionModel),
        ("Random Forest", NumericalFeatureRandomForestModel),
    ]

    for eachModel, ModelClass in allNumericalModels:
        
        mse = k_fold_MSE(trainingDataset, eachModel, ModelClass, currentNumericalFeatures, targetFeature[0], splitNumber, 6)
        print(f"The current model is: {eachModel} with average MSE: {mse:.5f}, and the minimum index: {minimumIndex:.5f}")

        if mse < lowestMse:
            lowestMse = mse
            bestNumericalModel = eachModel
            bestNumericalIndex = minimumIndex

print(f"\nThe best model is {bestNumericalModel} with average MSE {lowestMse:.5f}, and the minimum index is {bestNumericalIndex:.5f}")

numericalFeatures = numerical_feature_high_importance(numericalFeatures, numerical_feature_importance_random_forest(trainingDataset, numericalFeatures, targetFeature[0]), minimumIndex=bestNumericalIndex)


## Fitting binary features using different models (LDA, Logistic Regression, Random Forest, SVM).
## Evaluating performance using Accuracy with K-Fold Cross-Validation.
print("\nFitting binary features using different models (LDA, Logistic Regression, Random Forest, SVM).")
print("Evaluating performance using Accuracy with K-Fold Cross-Validation.")

# Iterate different minimumIndex combinations of binary features.
bestBinaryIndex: float = 0
bestBinaryModel = None
highestBinaryAccuracy: float = 0.0
splitNumber: int = 5

for minimumIndex in np.arange(0, 0.06, 0.005):

    currentBinaryFeatures = binary_feature_high_importance(binaryFeatures, binary_feature_importance_random_forest(trainingDataset, binaryFeatures, targetFeature[0]), minimumIndex=minimumIndex)

    allBinaryModels = [
        ("LDA", BinaryFeatureLDAModel),
        ("Logistic Regression", BinaryFeatureLogisticRegressionModel),
        ("Random Forest", BinaryFeatureRandomForestModel),
        ("SVM", BinaryFeatureSVMModel)
    ]

    for eachModel, ModelClass in allBinaryModels:
        accuracy = k_fold_accuracy(trainingDataset, eachModel, ModelClass, currentBinaryFeatures, targetFeature[0], splitNumber, 6)
        print(f"The current binary model is: {eachModel} with average Accuracy: {accuracy:.5f}, and the minimum index: {minimumIndex:.5f}")

        if accuracy > highestBinaryAccuracy:
            highestBinaryAccuracy = accuracy
            bestBinaryModel = eachModel
            bestBinaryIndex = minimumIndex

print(f"\nThe best binary model is {bestBinaryModel} with Accuracy {highestBinaryAccuracy:.5f}, and the minimum index is {bestBinaryIndex:.5f}")

binaryFeatures = binary_feature_high_importance(binaryFeatures, binary_feature_importance_random_forest(trainingDataset, binaryFeatures, targetFeature[0]), minimumIndex=bestBinaryIndex)


## Fitting categorical features using different models (LDA, Logistic Regression, Random Forest, SVM) with Target Encoding.
## Evaluating performance using Accuracy with K-Fold Cross-Validation.
print("\nFitting categorical features using different models (LDA, Logistic Regression, Random Forest, SVM) with Target Encoding.")
print("Evaluating performance using Accuracy with K-Fold Cross-Validation.")

# Target Encoding for categorical features.
EncodingCategoricalFeatures = categorical_features_encoding(trainingDataset, categoricalFeatures, targetFeature[0])

# Iterate different minimumIndex combinations of categorical features.
bestCategoricalIndex: float = 0
bestCategoricalModel = None
highestCategoricalAccuracy: float = 0.0
splitNumber: int = 5

for minimumIndex in np.arange(0, 0.6, 0.05):

    currentCategoricalFeatures = categorical_feature_high_importance(categoricalFeatures, categorical_feature_importance_random_forest(trainingDataset, categoricalFeatures, targetFeature[0]), minimumIndex=minimumIndex)

    allCategoricalModels = [
        ("LDA", CategoricalFeatureLDAModel),
        ("Logistic Regression", CategoricalFeatureLogisticRegressionModel),
        ("Random Forest", CategoricalFeatureRandomForestModel),
        ("SVM", CategoricalFeatureSVMModel)
    ]

    for eachModel, ModelClass in allCategoricalModels:
        accuracy = k_fold_accuracy(EncodingCategoricalFeatures, eachModel, ModelClass, currentCategoricalFeatures, targetFeature[0], splitNumber, 6)
        print(f"The current categorical model is: {eachModel} with average Accuracy: {accuracy:.5f}, and the minimum index: {minimumIndex:.5f}")

        if accuracy > highestCategoricalAccuracy:
            highestCategoricalAccuracy = accuracy
            bestCategoricalModel = eachModel
            bestCategoricalIndex = minimumIndex

print(f"\nThe best categorical model is: {bestCategoricalModel} with Accuracy: {highestCategoricalAccuracy:.5f}, and the minimum index: {bestCategoricalIndex:.5f}")

categoricalFeatures = categorical_feature_high_importance(categoricalFeatures, categorical_feature_importance_random_forest(trainingDataset, categoricalFeatures, targetFeature[0]), minimumIndex=bestCategoricalIndex)


## Combining and training all three models.
kFoldCrossValidator = k_fold_cross_validator(splitNumber=5, shuffle=True)

bestBaggingClassifier = None
highestValidationAccuracy = 0.0
highestTrainingAccuracy = 0.0

for trainIndex, valIndex in kFoldCrossValidator.split(trainingDataset):
    splittedTrainSet = trainingDataset.iloc[trainIndex]
    validationSet = trainingDataset.iloc[valIndex]

    numericalModel = NumericalFeatureRandomForestModel(splittedTrainSet, validationSet, numericalFeatures, targetFeature[0])
    numericalModel.hyperparameter_tuning()
    numericalModel.fit_model()

    binaryModel = BinaryFeatureLDAModel(splittedTrainSet, validationSet, binaryFeatures, targetFeature[0])
    binaryModel.fit_model()

    categoricalModel = CategoricalFeatureRandomForestModel(splittedTrainSet, validationSet, categoricalFeatures, targetFeature[0])
    categoricalModel.hyperparameter_tuning()
    categoricalModel.fit_model()

    baggingClassifier = BaggingClassifier(estimator=xgb.XGBClassifier(objective='reg:pseudohubererror', random_state=6, eval_metric='logloss'), n_estimators=100, random_state=6)

    baggingClassifier.fit(splittedTrainSet[numericalFeatures + binaryFeatures + categoricalFeatures], splittedTrainSet[targetFeature[0]])

    validationFeatures = validationSet[numericalFeatures + binaryFeatures + categoricalFeatures]
    validationPrediction = baggingClassifier.predict(validationFeatures)

    validationAccuracy = f1_score_prediction(validationSet[targetFeature[0]].values, validationPrediction)
    trainingPrediction = baggingClassifier.predict(splittedTrainSet[numericalFeatures + binaryFeatures + categoricalFeatures])
    trainingAccuracy = f1_score_prediction(splittedTrainSet[targetFeature[0]].values, trainingPrediction)

    print(f"Current fold - Training Accuracy: {trainingAccuracy:.5f}, Validation Accuracy: {validationAccuracy:.5f}")

    if validationAccuracy > highestValidationAccuracy or (
        validationAccuracy == highestValidationAccuracy and trainingAccuracy > highestTrainingAccuracy):
        highestValidationAccuracy = validationAccuracy
        highestTrainingAccuracy = trainingAccuracy
        bestBaggingClassifier = clone(baggingClassifier)

bestBaggingClassifier.fit(trainingDataset[numericalFeatures + binaryFeatures + categoricalFeatures], trainingDataset[targetFeature[0]])

print(f"\nThe best model has Validation Accuracy: {highestValidationAccuracy:.5f} and Training Accuracy: {highestTrainingAccuracy:.5f}")




## Combining and training only two models.
kFoldCrossValidator = k_fold_cross_validator(splitNumber=25, shuffle=True)

bestBaggingClassifier = None
highestValidationAccuracy = 0.0
highestTrainingAccuracy = 0.0

for trainIndex, valIndex in kFoldCrossValidator.split(trainingDataset):
    splittedTrainSet = trainingDataset.iloc[trainIndex]
    validationSet = trainingDataset.iloc[valIndex]

    numericalModel = NumericalFeatureRandomForestModel(splittedTrainSet, validationSet, numericalFeatures, targetFeature[0])
    numericalModel.hyperparameter_tuning()
    numericalModel.fit_model()

    binaryModel = BinaryFeatureLDAModel(splittedTrainSet, validationSet, binaryFeatures, targetFeature[0])
    binaryModel.fit_model()

    baggingClassifier = BaggingClassifier(estimator=xgb.XGBClassifier(objective='reg:pseudohubererror', random_state=6, eval_metric='logloss'), n_estimators=100, random_state=6)

    baggingClassifier.fit(splittedTrainSet[numericalFeatures + binaryFeatures], splittedTrainSet[targetFeature[0]])

    validationFeatures = validationSet[numericalFeatures + binaryFeatures]
    validationPrediction = baggingClassifier.predict(validationFeatures)

    validationAccuracy = f1_score_prediction(validationSet[targetFeature[0]].values, validationPrediction)
    trainingPrediction = baggingClassifier.predict(splittedTrainSet[numericalFeatures + binaryFeatures])
    trainingAccuracy = f1_score_prediction(splittedTrainSet[targetFeature[0]].values, trainingPrediction)

    print(f"Current fold - Training Accuracy: {trainingAccuracy:.5f}, Validation Accuracy: {validationAccuracy:.5f}")

    if validationAccuracy > highestValidationAccuracy or (
        validationAccuracy == highestValidationAccuracy and trainingAccuracy > highestTrainingAccuracy):
        highestValidationAccuracy = validationAccuracy
        highestTrainingAccuracy = trainingAccuracy
        bestBaggingClassifier = clone(baggingClassifier)

bestBaggingClassifier.fit(trainingDataset[numericalFeatures + binaryFeatures], trainingDataset[targetFeature[0]])

print(f"\nThe best model has Validation Accuracy: {highestValidationAccuracy:.5f} and Training Accuracy: {highestTrainingAccuracy:.5f}")



# Predicting on test.csv
testDataset = pd.read_csv("test.csv")


combinedTestFeatures = testDataset[numericalFeatures + binaryFeatures].values

finalPrediction = bestBaggingClassifier.predict(combinedTestFeatures)


output_df = pd.DataFrame({"PatientID": testDataset[identifier[0]], "Diagnosis": finalPrediction})


output_df.to_csv("Prediction_Milestone_2.csv", index=False)