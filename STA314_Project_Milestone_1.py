import numpy as np
import pandas as pd

# NOTE IMPORTANT: The current .py file is our first milestone.
# NOTE IMPORTANT: We also have two .py files that contain functions and classes.


# Importing all classes from STA314_Project_Class.py.
from STA314_Project_Class import BinaryFeatureLDAModel, BinaryFeatureLogisticRegressionModel, BinaryFeatureRandomForestModel, BinaryFeatureSVMModel
from STA314_Project_Class import CategoricalFeatureLDAModel, CategoricalFeatureLogisticRegressionModel, CategoricalFeatureRandomForestModel, CategoricalFeatureSVMModel
from STA314_Project_Class import NumericalFeatureBayesianRegressionModel, NumericalFeatureLinearRegressionModel, NumericalFeaturePolynomialRegressionModel, NumericalFeatureRandomForestModel


# Importing all functions from STA314_Project_Function.py.
from STA314_Project_Function import binary_feature_high_importance, binary_feature_importance_random_forest
from STA314_Project_Function import categorical_feature_high_importance, categorical_feature_importance_random_forest
from STA314_Project_Function import accuracy_prediction, dataset_splitting, feature_categorization
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

# Splitting the training dataset into a training dataset and a validation dataset in the ratio of 80/20 using the function dataset_splitting.
splittedTrainSet, validationSet = dataset_splitting(trainingDataset, 0.2, 0.8, 6, True, targetFeature)


## Fitting numerical features using different models (Bayesian Regression, Linear Regression, Polynomial Regression, Random Forest).
## Evaluating performance using Mean Squared Error (MSE).
print("\nFitting numerical features using different models (Bayesian Regression, Linear Regression, Polynomial Regression, Random Forest).")
print("Evaluating performance using Mean Squared Error (MSE).")

# Iterate different minimumIndex combinations of numerical features.
bestNumericalIndex: float = 0
bestNumericalModel = None
lowestMse: float = float('inf')

for minimumIndex in np.arange(0, 0.04, 0.005):

    currentNumericalFeatures = numerical_feature_high_importance(numericalFeatures, numerical_feature_importance_random_forest(trainingDataset, numericalFeatures, targetFeature[0]), minimumIndex=minimumIndex)

    allNumericalModels = [
        ("Bayesian Regression", NumericalFeatureBayesianRegressionModel(splittedTrainSet, validationSet, currentNumericalFeatures, targetFeature[0])),
        ("Linear Regression", NumericalFeatureLinearRegressionModel(splittedTrainSet, validationSet, currentNumericalFeatures, targetFeature[0])),
        ("Polynomial Regression", NumericalFeaturePolynomialRegressionModel(splittedTrainSet, validationSet, currentNumericalFeatures, targetFeature[0])),
        ("Random Forest", NumericalFeatureRandomForestModel(splittedTrainSet, validationSet, currentNumericalFeatures, targetFeature[0]))
    ]

    for eachModel, model in allNumericalModels:
        model.fit_model()
        mse = model.validation_mean_squared_error()
        print(f"The current model is: {eachModel} with MSE: {mse:.5f}, and the minimum index: {minimumIndex:.5f}")

        if mse < lowestMse:
            lowestMse = mse
            bestNumericalModel = eachModel
            bestNumericalIndex = minimumIndex

print(f"The best model is {bestNumericalModel} with MSE {lowestMse:.5f}, and the minimum index is {bestNumericalIndex:.5f}")

numericalFeatures = numerical_feature_high_importance(numericalFeatures, numerical_feature_importance_random_forest(trainingDataset, numericalFeatures, targetFeature[0]), minimumIndex=bestNumericalIndex)


## Fitting binary features using different models (LDA, Logistic Regression, Random Forest, SVM).
## Evaluating performance using Accuracy.
print("\nFitting binary features using different models (LDA, Logistic Regression, Random Forest, SVM).")
print("Evaluating performance using Accuracy.")

# Iterate different minimumIndex combinations of binary features.
bestBinaryIndex: float = 0
bestBinaryModel = None
highestBinaryAccuracy: float = 0

for minimumIndex in np.arange(0, 0.06, 0.01):

    currentBinaryFeatures = binary_feature_high_importance(binaryFeatures, binary_feature_importance_random_forest(trainingDataset, binaryFeatures, targetFeature[0]), minimumIndex=minimumIndex)

    allBinaryModels = [
        ("LDA", BinaryFeatureLDAModel(splittedTrainSet, validationSet, currentBinaryFeatures, targetFeature[0])),
        ("Logistic Regression", BinaryFeatureLogisticRegressionModel(splittedTrainSet, validationSet, currentBinaryFeatures, targetFeature[0])),
        ("Random Forest", BinaryFeatureRandomForestModel(splittedTrainSet, validationSet, currentBinaryFeatures, targetFeature[0])),
        ("SVM", BinaryFeatureSVMModel(splittedTrainSet, validationSet, currentBinaryFeatures, targetFeature[0]))
    ]

    for eachModel, model in allBinaryModels:
        model.fit_model()
        accuracy = model.validation_accuracy_model()
        print(f"The current binary model is: {eachModel} with Accuracy: {accuracy:.5f}, and the minimum index: {minimumIndex:.5f}")

        if accuracy > highestBinaryAccuracy:
            highestBinaryAccuracy = accuracy
            bestBinaryModel = eachModel
            bestBinaryIndex = minimumIndex

print(f"The best binary model is {bestBinaryModel} with Accuracy {highestBinaryAccuracy:.5f}, and the minimum index is {bestBinaryIndex:.5f}")

binaryFeatures = binary_feature_high_importance(binaryFeatures, binary_feature_importance_random_forest(trainingDataset, binaryFeatures, targetFeature[0]), minimumIndex=bestBinaryIndex)


## Fitting categorical features using different models (LDA, Logistic Regression, Random Forest, SVM). 
## Evaluating performance using Accuracy.
print("\nFitting categorical features using different models (LDA, Logistic Regression, Random Forest, SVM).")
print("Evaluating performance using Accuracy.")

# Iterate different minimumIndex combinations of categorical features.
bestCategoricalIndex: float = 0
bestCategoricalModel = None
highestCategoricalAccuracy: float = 0.0

for minimumIndex in np.arange(0, 0.6, 0.1):

    currentCategoricalFeatures = categorical_feature_high_importance(categoricalFeatures, categorical_feature_importance_random_forest(trainingDataset, categoricalFeatures, targetFeature[0]), minimumIndex=minimumIndex)

    allCategoricalModels = [
        ("LDA", CategoricalFeatureLDAModel(splittedTrainSet, validationSet, currentCategoricalFeatures, targetFeature[0])),
        ("Logistic Regression", CategoricalFeatureLogisticRegressionModel(splittedTrainSet, validationSet, currentCategoricalFeatures, targetFeature[0])),
        ("Random Forest", CategoricalFeatureRandomForestModel(splittedTrainSet, validationSet, currentCategoricalFeatures, targetFeature[0])),
        ("SVM", CategoricalFeatureSVMModel(splittedTrainSet, validationSet, currentCategoricalFeatures, targetFeature[0]))
    ]

    for eachModel, model in allCategoricalModels:
        model.fit_model()
        accuracy = model.validation_accuracy_model()
        print(f"The current categorical model is {eachModel} with Accuracy {accuracy:.5f}, and the minimum index is {minimumIndex:.5f}")

        if accuracy > highestCategoricalAccuracy:
            highestCategoricalAccuracy = accuracy
            bestCategoricalModel = eachModel
            bestCategoricalIndex = minimumIndex

print(f"The best categorical model is: {bestCategoricalModel} with Accuracy: {highestCategoricalAccuracy:.5f}, and the minimum index: {bestCategoricalIndex:.5f}")

categoricalFeatures = categorical_feature_high_importance(categoricalFeatures, categorical_feature_importance_random_forest(trainingDataset, categoricalFeatures, targetFeature[0]), minimumIndex=bestCategoricalIndex)


## Fit numericalFeatures by RandomForestModel.
numericalFeatureRandomForest = NumericalFeatureRandomForestModel(splittedTrainSet, validationSet, numericalFeatures, targetFeature[0])
numericalFeatureRandomForest.fit_model()
trainingNumericalPrediction = numericalFeatureRandomForest.model.predict(numericalFeatureRandomForest.trainFeatureValue)
validationNumericalPrediction = numericalFeatureRandomForest.model.predict(numericalFeatureRandomForest.validationFeatureValue)

## Fit binaryFeatures by LogisticRegressionModel.
binaryFeatureLogisticRegression = BinaryFeatureLogisticRegressionModel(splittedTrainSet, validationSet, binaryFeatures, targetFeature[0])
binaryFeatureLogisticRegression.fit_model()
trainingBinaryPrediction = binaryFeatureLogisticRegression.binaryFeatureLogisticRegression.predict(binaryFeatureLogisticRegression.trainFeatureValue)
validationBinaryPrediction = binaryFeatureLogisticRegression.binaryFeatureLogisticRegression.predict(binaryFeatureLogisticRegression.validationFeatureValue)

## Fit categoricalFeatures by RandomForestModel.
categoricalFeatureRandomForest = CategoricalFeatureRandomForestModel(splittedTrainSet, validationSet, categoricalFeatures, targetFeature[0])
categoricalFeatureRandomForest.fit_model()
trainingCategoricalPrediction = categoricalFeatureRandomForest.model.predict(categoricalFeatureRandomForest.trainFeatureValue)
validationCategoricalPrediction = categoricalFeatureRandomForest.model.predict(categoricalFeatureRandomForest.validationFeatureValue)


## Looping different combinations of weights from the three models and combining the predictions.
bestWeightCombination: tuple = None
highestValidationAccuracyThreeModel: float = 0

for numericalWeight in np.arange(0, 1.1, 0.1):

    for binaryWeight in np.arange(0, 1.1 - numericalWeight, 0.1):

        categoricalWeight = 1 - numericalWeight - binaryWeight

        if not categoricalWeight < 0:

            combinedPrediction = ( numericalWeight * validationNumericalPrediction + binaryWeight * validationBinaryPrediction + categoricalWeight * validationCategoricalPrediction) > 0.5

            currentValidationAccuracy = accuracy_prediction(validationSet[targetFeature[0]].values, combinedPrediction)

            print(f"Weights - Numerical: {numericalWeight:.1f}, Binary: {binaryWeight:.1f}, Categorical: {categoricalWeight:.1f}, Validation Accuracy: {currentValidationAccuracy :.5f}")

            if currentValidationAccuracy  > highestValidationAccuracyThreeModel:
                highestValidationAccuracyThreeModel = currentValidationAccuracy
                bestWeightCombination = (numericalWeight, binaryWeight, categoricalWeight)

print(f"\nThe best weight combination is Numerical: {bestWeightCombination[0]:.1f}, Binary: {bestWeightCombination[1]:.1f}, Categorical: {bestWeightCombination[2]:.1f}, and the highest validation accuracy is {highestValidationAccuracyThreeModel:.5f}")


## Looping different combinations of weights from numerical and binary models and combining the predictions.
bestWeightCombination: tuple = None
highestValidationAccuracyTwoModel: float = 0

for numericalWeight in np.arange(0, 1.1, 0.1):

    binaryWeight = 1 - numericalWeight

    combinedPrediction = (numericalWeight * validationNumericalPrediction + binaryWeight * validationBinaryPrediction) > 0.5

    currentValidationAccuracy = accuracy_prediction(validationSet[targetFeature[0]].values, combinedPrediction)

    print(f"Weights - Numerical: {numericalWeight:.1f}, Binary: {binaryWeight:.1f}, Validation Accuracy: {currentValidationAccuracy :.5f}")

    if currentValidationAccuracy > highestValidationAccuracyTwoModel:
        highestValidationAccuracyTwoModel = currentValidationAccuracy
        bestWeightCombination = (numericalWeight, binaryWeight, categoricalWeight)

print(f"\nThe best weight combination is Numerical: {bestWeightCombination[0]:.1f}, Binary: {bestWeightCombination[1]:.1f}, and the highest validation accuracy is {highestValidationAccuracyTwoModel:.5f}")


## Conclusion
Conclusion = """
Conclusion:
1. Individual Model Accuracy:
   Numerical Features by RandomForestModel: Validation MSE is {:.5f}
   Binary Features by LogisticRegressionModel: Validation Accuracy is {:.5f}
   Categorical Features by RandomForestModel: Validation Accuracy is {:.5f}
\n2. Combined Model Accuracy:
   Combined Numerical, Binary, and Categorical Models: Validation Accuracy is {:.5f}
   Combined Numerical and Binary Models: Validation Accuracy is {:.5f}
\nTherefore, based on the results, the highest validation accuracy is the combining of Numerical and Binary models, with validation accuracy {:.5f}.
"""
print(Conclusion.format(numericalFeatureRandomForest.validation_mean_squared_error(), binaryFeatureLogisticRegression.validation_accuracy_model(), categoricalFeatureRandomForest.validation_accuracy_model(), highestValidationAccuracyThreeModel, highestValidationAccuracyTwoModel, highestValidationAccuracyTwoModel))