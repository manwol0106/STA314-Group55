import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import xgboost as xgb


from STA314_Project_Function import accuracy_prediction


## NOTE Models used to fit numerical features.
class NumericalFeatureBayesianRegressionModel:

    def __init__(self, splittedTrainSet, validationSet, numericalFeatures, targetFeature):

        self.splittedTrainSet = splittedTrainSet
        self.validationSet = validationSet
        self.numericalFeatures = numericalFeatures
        self.targetFeature = targetFeature
        
        self.trainFeatureValue = self.splittedTrainSet[self.numericalFeatures].values
        self.trainTargetValue = self.splittedTrainSet[self.targetFeature].values

        self.validationFeatureValue = self.validationSet[self.numericalFeatures].values
        self.validationTargetValue = self.validationSet[self.targetFeature].values

        self.meanTrainFeatureValue = np.mean(self.trainFeatureValue, axis=0)
        self.standardDeviationTrainFeatureValue = np.std(self.trainFeatureValue, axis=0)

        self.model = BayesianRidge()

    def fit_model(self):

        self.model.fit((self.trainFeatureValue - self.meanTrainFeatureValue) / self.standardDeviationTrainFeatureValue, self.trainTargetValue)

    def mean_squared_error(self):

        trainingPrediction = self.model.predict((self.trainFeatureValue - self.meanTrainFeatureValue) / self.standardDeviationTrainFeatureValue)
        validationPrediction = self.model.predict((self.validationFeatureValue - self.meanTrainFeatureValue) / self.standardDeviationTrainFeatureValue)

        trainingMSE = np.mean((self.trainTargetValue - trainingPrediction) ** 2)
        print(f"Mean Squared Error on Training Set Using Bayesian Regression: {trainingMSE:.5f}")

        validationMSE = np.mean((self.validationTargetValue - validationPrediction) ** 2)
        print(f"Mean Squared Error on Validation Set Using Bayesian Regression: {validationMSE:.5f}")

    def training_mean_squared_error(self):

        trainingPrediction = self.model.predict((self.trainFeatureValue - self.meanTrainFeatureValue) / self.standardDeviationTrainFeatureValue)
        trainingMSE = np.mean((self.trainTargetValue - trainingPrediction) ** 2)
        return trainingMSE
    
    def validation_mean_squared_error(self):

        validationPrediction = self.model.predict((self.validationFeatureValue - self.meanTrainFeatureValue) / self.standardDeviationTrainFeatureValue)
        validationMSE = np.mean((self.validationTargetValue - validationPrediction) ** 2)
        return validationMSE
    
class NumericalFeatureDecisionTreeClassifierModel:
    def __init__(self, splittedTrainSet, validationSet, numericalFeatures, targetFeature):
        self.splittedTrainSet = splittedTrainSet
        self.validationSet = validationSet
        self.numericalFeatures = numericalFeatures
        self.targetFeature = targetFeature
        
        self.trainFeatureValue = self.splittedTrainSet[self.numericalFeatures].values
        self.trainTargetValue = self.splittedTrainSet[self.targetFeature].values

        self.validationFeatureValue = self.validationSet[self.numericalFeatures].values
        self.validationTargetValue = self.validationSet[self.targetFeature].values

        self.model = DecisionTreeClassifier(random_state=6)

    def fit_model(self):
        self.model.fit(self.trainFeatureValue, self.trainTargetValue)

    def mean_squared_error(self):
        trainingPrediction = self.model.predict(self.trainFeatureValue)
        validationPrediction = self.model.predict(self.validationFeatureValue)

        trainingMSE = np.mean((self.trainTargetValue - trainingPrediction) ** 2)
        print(f"Mean Squared Error on Training Set Using Decision Tree: {trainingMSE:.5f}")

        validationMSE = np.mean((self.validationTargetValue - validationPrediction) ** 2)
        print(f"Mean Squared Error on Validation Set Using Decision Tree: {validationMSE:.5f}")

    def training_mean_squared_error(self):
        trainingPrediction = self.model.predict(self.trainFeatureValue)
        trainingMSE = np.mean((self.trainTargetValue - trainingPrediction) ** 2)
        return trainingMSE
    
    def validation_mean_squared_error(self):
        validationPrediction = self.model.predict(self.validationFeatureValue)
        validationMSE = np.mean((self.validationTargetValue - validationPrediction) ** 2)
        return validationMSE

class NumericalFeaturePolynomialRegressionModel:

    def __init__(self, splittedTrainSet, validationSet, numericalFeatures, targetFeature, degree=2):

        self.splittedTrainSet = splittedTrainSet
        self.validationSet = validationSet
        self.numericalFeatures = numericalFeatures
        self.targetFeature = targetFeature
        
        self.trainFeatureValue = self.splittedTrainSet[self.numericalFeatures].values
        self.trainTargetValue = self.splittedTrainSet[self.targetFeature].values

        self.validationFeatureValue = self.validationSet[self.numericalFeatures].values
        self.validationTargetValue = self.validationSet[self.targetFeature].values

        self.meanTrainFeatureValue = np.mean(self.trainFeatureValue, axis=0)
        self.standardDeviationTrainFeatureValue = np.std(self.trainFeatureValue, axis=0)

        self.poly = PolynomialFeatures(degree)
        self.model = LinearRegression()

    def fit_model(self):

        train_features_poly = self.poly.fit_transform((self.trainFeatureValue - self.meanTrainFeatureValue) / self.standardDeviationTrainFeatureValue)
        self.model.fit(train_features_poly, self.trainTargetValue)

    def mean_squared_error(self):

        train_features_poly = self.poly.transform((self.trainFeatureValue - self.meanTrainFeatureValue) / self.standardDeviationTrainFeatureValue)
        validation_features_poly = self.poly.transform((self.validationFeatureValue - self.meanTrainFeatureValue) / self.standardDeviationTrainFeatureValue)

        trainingPrediction = self.model.predict(train_features_poly)
        validationPrediction = self.model.predict(validation_features_poly)

        trainingMSE = np.mean((self.trainTargetValue - trainingPrediction) ** 2)
        print(f"Mean Squared Error on Training Set Using Polynomial Regression: {trainingMSE:.5f}")

        validationMSE = np.mean((self.validationTargetValue - validationPrediction) ** 2)
        print(f"Mean Squared Error on Validation Set Using Polynomial Regression: {validationMSE:.5f}")


    def training_mean_squared_error(self):

        train_features_poly = self.poly.transform((self.trainFeatureValue - self.meanTrainFeatureValue) / self.standardDeviationTrainFeatureValue)
        trainingPrediction = self.model.predict(train_features_poly)
        trainingMSE = np.mean((self.trainTargetValue - trainingPrediction) ** 2)
        return trainingMSE
    
    def validation_mean_squared_error(self):

        validation_features_poly = self.poly.transform((self.validationFeatureValue - self.meanTrainFeatureValue) / self.standardDeviationTrainFeatureValue)
        validationPrediction = self.model.predict(validation_features_poly)
        validationMSE = np.mean((self.validationTargetValue - validationPrediction) ** 2)
        return validationMSE

class NumericalFeatureLinearRegressionModel:

    def __init__(self, splittedTrainSet, validationSet, numericalFeatures, targetFeature):

        self.splittedTrainSet = splittedTrainSet
        self.validationSet = validationSet
        self.numericalFeatures = numericalFeatures
        self.targetFeature = targetFeature
        
        self.trainFeatureValue = self.splittedTrainSet[self.numericalFeatures].values
        self.trainTargetValue = self.splittedTrainSet[self.targetFeature].values

        self.validationFeatureValue = self.validationSet[self.numericalFeatures].values
        self.validationTargetValue = self.validationSet[self.targetFeature].values

        self.meanTrainFeatureValue = np.mean(self.trainFeatureValue, axis=0)
        self.standardDeviationTrainFeatureValue = np.std(self.trainFeatureValue, axis=0)

        self.model = LinearRegression()

    def fit_model(self):
        self.model.fit((self.trainFeatureValue - self.meanTrainFeatureValue) / self.standardDeviationTrainFeatureValue, self.trainTargetValue)

    def mean_squared_error(self):
        trainingPrediction = self.model.predict((self.trainFeatureValue - self.meanTrainFeatureValue) / self.standardDeviationTrainFeatureValue)
        validationPrediction = self.model.predict((self.validationFeatureValue - self.meanTrainFeatureValue) / self.standardDeviationTrainFeatureValue)

        trainingMSE = np.mean((self.trainTargetValue - trainingPrediction) ** 2)
        print(f"Mean Squared Error on Training Set Using Linear Regression: {trainingMSE:.5f}")

        validationMSE = np.mean((self.validationTargetValue - validationPrediction) ** 2)
        print(f"Mean Squared Error on Validation Set Using Linear Regression: {validationMSE:.5f}")

    def training_mean_squared_error(self):

        trainingPrediction = self.model.predict((self.trainFeatureValue - self.meanTrainFeatureValue) / self.standardDeviationTrainFeatureValue)
        trainingMSE = np.mean((self.trainTargetValue - trainingPrediction) ** 2)
        return trainingMSE
    
    def validation_mean_squared_error(self):

        validationPrediction = self.model.predict((self.validationFeatureValue - self.meanTrainFeatureValue) / self.standardDeviationTrainFeatureValue)
        validationMSE = np.mean((self.validationTargetValue - validationPrediction) ** 2)
        return validationMSE

class NumericalFeatureRandomForestModel:

    def __init__(self, splittedTrainSet, validationSet, numericalFeatures, targetFeature):

        self.splittedTrainSet = splittedTrainSet
        self.validationSet = validationSet
        self.numericalFeatures = numericalFeatures
        self.targetFeature = targetFeature

        self.trainFeatureValue = self.splittedTrainSet[self.numericalFeatures].values
        self.trainTargetValue = self.splittedTrainSet[self.targetFeature].values

        self.validationFeatureValue = self.validationSet[self.numericalFeatures].values
        self.validationTargetValue = self.validationSet[self.targetFeature].values

        self.model = RandomForestRegressor(random_state=6)

    def hyperparameter_tuning(self):
        paramGrid = {
            'n_estimators': [100],
            'max_depth': [3],
            'min_samples_split': [12],
            'min_samples_leaf': [5]
        }

        gridSearch = GridSearchCV(
            estimator=self.model,
            param_grid=paramGrid,
            scoring='neg_mean_squared_error',
            cv=3,
            verbose=1
        )
        gridSearch.fit(self.trainFeatureValue, self.trainTargetValue)

        print(f"The best parameters are: {gridSearch.best_params_}")
        self.model = gridSearch.best_estimator_

    def fit_model(self):

        self.model.fit(self.trainFeatureValue, self.trainTargetValue)

    def mean_squared_error(self):

        trainingPrediction = self.model.predict(self.trainFeatureValue)
        validationPrediction = self.model.predict(self.validationFeatureValue)

        trainingMSE = np.mean((self.trainTargetValue - trainingPrediction) ** 2)
        print(f"Mean Squared Error on Training Set Using Random Forest: {trainingMSE:.5f}")

        validationMSE = np.mean((self.validationTargetValue - validationPrediction) ** 2)
        print(f"Mean Squared Error on Validation Set Using Random Forest: {validationMSE:.5f}")

    def training_mean_squared_error(self):

        trainingPrediction = self.model.predict(self.trainFeatureValue)
        trainingMSE = np.mean((self.trainTargetValue - trainingPrediction) ** 2)
        return trainingMSE

    def validation_mean_squared_error(self):

        validationPrediction = self.model.predict(self.validationFeatureValue)
        validationMSE = np.mean((self.validationTargetValue - validationPrediction) ** 2)
        return validationMSE


## NOTE Models used to fit binary features.
class BinaryFeatureDecisionTreeClassifierModel:
    def __init__(self, splittedTrainSet, validationSet, binaryFeatures, targetFeature):
        self.splittedTrainSet = splittedTrainSet
        self.validationSet = validationSet
        self.binaryFeatures = binaryFeatures
        self.targetFeature = targetFeature

        self.trainFeatureValue = self.splittedTrainSet[self.binaryFeatures].values
        self.trainTargetValue = self.splittedTrainSet[self.targetFeature].values

        self.validationFeatureValue = self.validationSet[self.binaryFeatures].values
        self.validationTargetValue = self.validationSet[self.targetFeature].values

        self.model = DecisionTreeClassifier(random_state=6)

    def fit_model(self):

        self.model.fit(self.trainFeatureValue, self.trainTargetValue)

    def accuracy_model(self):

        trainingPrediction = self.model.predict(self.trainFeatureValue)
        validationPrediction = self.model.predict(self.validationFeatureValue)

        trainingAccuracy = accuracy_prediction(self.trainTargetValue, trainingPrediction)
        print(f"Accuracy on Training Set Using Decision Tree: {trainingAccuracy:.5f}")

        validationAccuracy = accuracy_prediction(self.validationTargetValue, validationPrediction)
        print(f"Accuracy on Validation Set Using Decision Tree: {validationAccuracy:.5f}")

    def training_accuracy_model(self):

        trainingPrediction = self.model.predict(self.trainFeatureValue)
        trainingAccuracy = accuracy_prediction(self.trainTargetValue, trainingPrediction)
        return trainingAccuracy

    def validation_accuracy_model(self):

        validationPrediction = self.model.predict(self.validationFeatureValue)
        validationAccuracy = accuracy_prediction(self.validationTargetValue, validationPrediction)
        return validationAccuracy

class BinaryFeatureLDAModel:

    def __init__(self, splittedTrainSet, validationSet, binaryFeatures, targetFeature):
        
        self.splittedTrainSet = splittedTrainSet
        self.validationSet = validationSet
        self.binaryFeatures = binaryFeatures
        self.targetFeature = targetFeature

        self.trainFeatureValue = self.splittedTrainSet[self.binaryFeatures].values
        self.trainTargetValue = self.splittedTrainSet[self.targetFeature].values

        self.validationFeatureValue = self.validationSet[self.binaryFeatures].values
        self.validationTargetValue = self.validationSet[self.targetFeature].values

        self.model = LinearDiscriminantAnalysis()

    def fit_model(self):

        self.model.fit(self.trainFeatureValue, self.trainTargetValue)

    def accuracy_model(self):

        trainingPrediction = self.model.predict(self.trainFeatureValue)
        validationPrediction = self.model.predict(self.validationFeatureValue)

        trainingAccuracy = accuracy_prediction(self.trainTargetValue, trainingPrediction)
        print(f"Accuracy on Training Set Using LDA: {trainingAccuracy:.5f}")

        validationAccuracy = accuracy_prediction(self.validationTargetValue, validationPrediction)
        print(f"Accuracy on Validation Set Using LDA: {validationAccuracy:.5f}")

    def training_accuracy_model(self):

        trainingPrediction = self.model.predict(self.trainFeatureValue)
        trainingAccuracy = accuracy_prediction(self.trainTargetValue, trainingPrediction)
        return trainingAccuracy

    def validation_accuracy_model(self):

        validationPrediction = self.model.predict(self.validationFeatureValue)
        validationAccuracy = accuracy_prediction(self.validationTargetValue, validationPrediction)
        return validationAccuracy


class BinaryFeatureLogisticRegressionModel:

    def __init__(self, splittedTrainSet, validationSet, binaryFeatures, targetFeature):
        
        self.splittedTrainSet = splittedTrainSet
        self.validationSet = validationSet
        self.binaryFeatures = binaryFeatures
        self.targetFeature = targetFeature

        self.trainFeatureValue = self.splittedTrainSet[self.binaryFeatures].values
        self.trainTargetValue = self.splittedTrainSet[self.targetFeature].values

        self.validationFeatureValue = self.validationSet[self.binaryFeatures].values
        self.validationTargetValue = self.validationSet[self.targetFeature].values

        self.binaryFeatureLogisticRegression = LogisticRegression(random_state=6)

    def fit_model(self):

        self.binaryFeatureLogisticRegression.fit(self.trainFeatureValue, self.trainTargetValue)

    def accuracy_model(self):

        trainingPrediction = self.binaryFeatureLogisticRegression.predict(self.trainFeatureValue)
        validationPrediction = self.binaryFeatureLogisticRegression.predict(self.validationFeatureValue)

        trainingAccuracy = accuracy_prediction(self.trainTargetValue, trainingPrediction)
        print(f"Accuracy on Training Set Using Logistic Regression: {trainingAccuracy:.5f}")

        validationAccuracy = accuracy_prediction(self.validationTargetValue, validationPrediction)
        print(f"Accuracy on Validation Set Using Logistic Regression: {validationAccuracy:.5f}")

    def training_accuracy_model(self):

        trainingPrediction = self.binaryFeatureLogisticRegression.predict(self.trainFeatureValue)
        trainingAccuracy = accuracy_prediction(self.trainTargetValue, trainingPrediction)
        return trainingAccuracy

    def validation_accuracy_model(self):

        validationPrediction = self.binaryFeatureLogisticRegression.predict(self.validationFeatureValue)
        validationAccuracy = accuracy_prediction(self.validationTargetValue, validationPrediction)
        return validationAccuracy
    
class BinaryFeatureNaiveBayesModel:

    def __init__(self, splittedTrainSet, validationSet, binaryFeatures, targetFeature):
        
        self.splittedTrainSet = splittedTrainSet
        self.validationSet = validationSet
        self.binaryFeatures = binaryFeatures
        self.targetFeature = targetFeature

        self.trainFeatureValue = self.splittedTrainSet[self.binaryFeatures].values
        self.trainTargetValue = self.splittedTrainSet[self.targetFeature].values

        self.validationFeatureValue = self.validationSet[self.binaryFeatures].values
        self.validationTargetValue = self.validationSet[self.targetFeature].values

        self.model = GaussianNB()

    def fit_model(self):

        self.model.fit(self.trainFeatureValue, self.trainTargetValue)

    def accuracy_model(self):

        trainingPrediction = self.model.predict(self.trainFeatureValue)
        validationPrediction = self.model.predict(self.validationFeatureValue)

        trainingAccuracy = accuracy_prediction(self.trainTargetValue, trainingPrediction)
        print(f"Accuracy on Training Set Using Naive Bayes: {trainingAccuracy:.5f}")

        validationAccuracy = accuracy_prediction(self.validationTargetValue, validationPrediction)
        print(f"Accuracy on Validation Set Using Naive Bayes: {validationAccuracy:.5f}")

    def training_accuracy_model(self):

        trainingPrediction = self.model.predict(self.trainFeatureValue)
        trainingAccuracy = accuracy_prediction(self.trainTargetValue, trainingPrediction)
        return trainingAccuracy

    def validation_accuracy_model(self):

        validationPrediction = self.model.predict(self.validationFeatureValue)
        validationAccuracy = accuracy_prediction(self.validationTargetValue, validationPrediction)
        return validationAccuracy
    
class BinaryFeatureRandomForestModel:

    def __init__(self, splittedTrainSet, validationSet, binaryFeatures, targetFeature):
        
        self.splittedTrainSet = splittedTrainSet
        self.validationSet = validationSet
        self.binaryFeatures = binaryFeatures
        self.targetFeature = targetFeature

        self.trainFeatureValue = self.splittedTrainSet[self.binaryFeatures].values
        self.trainTargetValue = self.splittedTrainSet[self.targetFeature].values

        self.validationFeatureValue = self.validationSet[self.binaryFeatures].values
        self.validationTargetValue = self.validationSet[self.targetFeature].values

        self.model = RandomForestClassifier(n_estimators=100, random_state=6)

    def hyperparameter_tuning(self):

        parameterGrid: dict = {
            'n_estimators': [100],
            'max_depth': [3],
            'min_samples_split': [2],
            'min_samples_leaf': [3]
        }
        
        gridSearch = GridSearchCV(RandomForestClassifier(random_state=6), param_grid=parameterGrid)
        gridSearch.fit(self.trainFeatureValue, self.trainTargetValue)
        
        print(f"Best parameters for Random Forest: {gridSearch.best_params_}")
        self.model = gridSearch.best_estimator_

    def fit_model(self):

        self.model.fit(self.trainFeatureValue, self.trainTargetValue)

    def accuracy_model(self):

        trainingPrediction = self.model.predict(self.trainFeatureValue)
        validationPrediction = self.model.predict(self.validationFeatureValue)

        trainingAccuracy = accuracy_prediction(self.trainTargetValue, trainingPrediction)
        print(f"Accuracy on Training Set Using Random Forest: {trainingAccuracy:.5f}")

        validationAccuracy = accuracy_prediction(self.validationTargetValue, validationPrediction)
        print(f"Accuracy on Validation Set Using Random Forest: {validationAccuracy:.5f}")

    def training_accuracy_model(self):

        trainingPrediction = self.model.predict(self.trainFeatureValue)
        trainingAccuracy = accuracy_prediction(self.trainTargetValue, trainingPrediction)
        return trainingAccuracy

    def validation_accuracy_model(self):

        validationPrediction = self.model.predict(self.validationFeatureValue)
        validationAccuracy = accuracy_prediction(self.validationTargetValue, validationPrediction)
        return validationAccuracy

class BinaryFeatureSVMModel:

    def __init__(self, splittedTrainSet, validationSet, binaryFeatures, targetFeature):
        
        self.splittedTrainSet = splittedTrainSet
        self.validationSet = validationSet
        self.binaryFeatures = binaryFeatures
        self.targetFeature = targetFeature

        self.trainFeatureValue = self.splittedTrainSet[self.binaryFeatures].values
        self.trainTargetValue = self.splittedTrainSet[self.targetFeature].values

        self.validationFeatureValue = self.validationSet[self.binaryFeatures].values
        self.validationTargetValue = self.validationSet[self.targetFeature].values

        self.model = svm.SVC(kernel='rbf', random_state=6)

    def fit_model(self):

        self.model.fit(self.trainFeatureValue, self.trainTargetValue)

    def accuracy_model(self):

        trainingPrediction = self.model.predict(self.trainFeatureValue)
        validationPrediction = self.model.predict(self.validationFeatureValue)

        trainingAccuracy = accuracy_prediction(self.trainTargetValue, trainingPrediction)
        print(f"Accuracy on Training Set Using SVM: {trainingAccuracy:.5f}")

        validationAccuracy = accuracy_prediction(self.validationTargetValue, validationPrediction)
        print(f"Accuracy on Validation Set Using SVM: {validationAccuracy:.5f}")

    def training_accuracy_model(self):

        trainingPrediction = self.model.predict(self.trainFeatureValue)
        trainingAccuracy = accuracy_prediction(self.trainTargetValue, trainingPrediction)
        return trainingAccuracy

    def validation_accuracy_model(self):

        validationPrediction = self.model.predict(self.validationFeatureValue)
        validationAccuracy = accuracy_prediction(self.validationTargetValue, validationPrediction)
        return validationAccuracy


## NOTE Models used to fit categorical features.
class CategoricalFeatureDecisionTreeClassifierModel:
    def __init__(self, splittedTrainSet, validationSet, categoricalFeatures, targetFeature):
        self.splittedTrainSet = splittedTrainSet
        self.validationSet = validationSet
        self.categoricalFeatures = categoricalFeatures
        self.targetFeature = targetFeature

        self.trainFeatureValue = self.splittedTrainSet[self.categoricalFeatures].values
        self.trainTargetValue = self.splittedTrainSet[self.targetFeature].values

        self.validationFeatureValue = self.validationSet[self.categoricalFeatures].values
        self.validationTargetValue = self.validationSet[self.targetFeature].values

        self.model = DecisionTreeClassifier(random_state=6)

    def fit_model(self):
        
        self.model.fit(self.trainFeatureValue, self.trainTargetValue)

    def accuracy_model(self):

        trainingPrediction = self.model.predict(self.trainFeatureValue)
        validationPrediction = self.model.predict(self.validationFeatureValue)

        trainingAccuracy = accuracy_prediction(self.trainTargetValue, trainingPrediction)
        print(f"Accuracy on Training Set Using Decision Tree: {trainingAccuracy:.5f}")

        validationAccuracy = accuracy_prediction(self.validationTargetValue, validationPrediction)
        print(f"Accuracy on Validation Set Using Decision Tree: {validationAccuracy:.5f}")

    def training_accuracy_model(self):

        trainingPrediction = self.model.predict(self.trainFeatureValue)
        trainingAccuracy = accuracy_prediction(self.trainTargetValue, trainingPrediction)
        return trainingAccuracy

    def validation_accuracy_model(self):

        validationPrediction = self.model.predict(self.validationFeatureValue)
        validationAccuracy = accuracy_prediction(self.validationTargetValue, validationPrediction)
        return validationAccuracy
    
class CategoricalFeatureLDAModel:

    def __init__(self, splittedTrainSet, validationSet, categoricalFeatures, targetFeature):
        
        self.splittedTrainSet = splittedTrainSet
        self.validationSet = validationSet
        self.categoricalFeatures = categoricalFeatures
        self.targetFeature = targetFeature

        self.trainFeatureValue = self.splittedTrainSet[self.categoricalFeatures].values
        self.trainTargetValue = self.splittedTrainSet[self.targetFeature].values

        self.validationFeatureValue = self.validationSet[self.categoricalFeatures].values
        self.validationTargetValue = self.validationSet[self.targetFeature].values

        self.model = LinearDiscriminantAnalysis()

    def fit_model(self):

        self.model.fit(self.trainFeatureValue, self.trainTargetValue)

    def accuracy_model(self):

        trainingPrediction = self.model.predict(self.trainFeatureValue)
        validationPrediction = self.model.predict(self.validationFeatureValue)

        trainingAccuracy = accuracy_prediction(self.trainTargetValue, trainingPrediction)
        print(f"Accuracy on Training Set Using LDA: {trainingAccuracy:.5f}")

        validationAccuracy = accuracy_prediction(self.validationTargetValue, validationPrediction)
        print(f"Accuracy on Validation Set Using LDA: {validationAccuracy:.5f}")

    def training_accuracy_model(self):

        trainingPrediction = self.model.predict(self.trainFeatureValue)
        trainingAccuracy = accuracy_prediction(self.trainTargetValue, trainingPrediction)
        return trainingAccuracy

    def validation_accuracy_model(self):

        validationPrediction = self.model.predict(self.validationFeatureValue)
        validationAccuracy = accuracy_prediction(self.validationTargetValue, validationPrediction)
        return validationAccuracy

class CategoricalFeatureLogisticRegressionModel:

    def __init__(self, splittedTrainSet, validationSet, categoricalFeatures, targetFeature):
        
        self.splittedTrainSet = splittedTrainSet
        self.validationSet = validationSet
        self.categoricalFeatures = categoricalFeatures
        self.targetFeature = targetFeature

        self.trainFeatureValue = self.splittedTrainSet[self.categoricalFeatures].values
        self.trainTargetValue = self.splittedTrainSet[self.targetFeature].values

        self.validationFeatureValue = self.validationSet[self.categoricalFeatures].values
        self.validationTargetValue = self.validationSet[self.targetFeature].values

        self.categoricalFeatureLogisticRegression = LogisticRegression(random_state=6)

    def fit_model(self):

        self.categoricalFeatureLogisticRegression.fit(self.trainFeatureValue, self.trainTargetValue)

    def accuracy_model(self):

        trainingPrediction = self.categoricalFeatureLogisticRegression.predict(self.trainFeatureValue)
        validationPrediction = self.categoricalFeatureLogisticRegression.predict(self.validationFeatureValue)


        trainingAccuracy = accuracy_prediction(self.trainTargetValue, trainingPrediction)
        print(f"Accuracy on Training Set Using Logistic Regression: {trainingAccuracy:.5f}")

        validationAccuracy = accuracy_prediction(self.validationTargetValue, validationPrediction)
        print(f"Accuracy on Validation Set Using Logistic Regression: {validationAccuracy:.5f}")

    def training_accuracy_model(self):

        trainingPrediction = self.categoricalFeatureLogisticRegression.predict(self.trainFeatureValue)
        trainingAccuracy = accuracy_prediction(self.trainTargetValue, trainingPrediction)
        return trainingAccuracy

    def validation_accuracy_model(self):

        validationPrediction = self.categoricalFeatureLogisticRegression.predict(self.validationFeatureValue)
        validationAccuracy = accuracy_prediction(self.validationTargetValue, validationPrediction)
        return validationAccuracy

class CategoricalFeatureNaiveBayesModel:

    def __init__(self, splittedTrainSet, validationSet, categoricalFeatures, targetFeature):
        
        self.splittedTrainSet = splittedTrainSet
        self.validationSet = validationSet
        self.categoricalFeatures = categoricalFeatures
        self.targetFeature = targetFeature

        self.trainFeatureValue = self.splittedTrainSet[self.categoricalFeatures].values
        self.trainTargetValue = self.splittedTrainSet[self.targetFeature].values

        self.validationFeatureValue = self.validationSet[self.categoricalFeatures].values
        self.validationTargetValue = self.validationSet[self.targetFeature].values

        self.model = GaussianNB()

    def fit_model(self):

        self.model.fit(self.trainFeatureValue, self.trainTargetValue)

    def accuracy_model(self):

        trainingPrediction = self.model.predict(self.trainFeatureValue)
        validationPrediction = self.model.predict(self.validationFeatureValue)

        trainingAccuracy = accuracy_prediction(self.trainTargetValue, trainingPrediction)
        print(f"Accuracy on Training Set Using Naive Bayes: {trainingAccuracy:.5f}")

        validationAccuracy = accuracy_prediction(self.validationTargetValue, validationPrediction)
        print(f"Accuracy on Validation Set Using Naive Bayes: {validationAccuracy:.5f}")

    def training_accuracy_model(self):

        trainingPrediction = self.model.predict(self.trainFeatureValue)
        trainingAccuracy = accuracy_prediction(self.trainTargetValue, trainingPrediction)
        return trainingAccuracy

    def validation_accuracy_model(self):

        validationPrediction = self.model.predict(self.validationFeatureValue)
        validationAccuracy = accuracy_prediction(self.validationTargetValue, validationPrediction)
        return validationAccuracy
    
class CategoricalFeatureRandomForestModel:

    def __init__(self, splittedTrainSet, validationSet, categoricalFeatures, targetFeature):
        
        self.splittedTrainSet = splittedTrainSet
        self.validationSet = validationSet
        self.categoricalFeatures = categoricalFeatures
        self.targetFeature = targetFeature

        self.trainFeatureValue = self.splittedTrainSet[self.categoricalFeatures].values
        self.trainTargetValue = self.splittedTrainSet[self.targetFeature].values

        self.validationFeatureValue = self.validationSet[self.categoricalFeatures].values
        self.validationTargetValue = self.validationSet[self.targetFeature].values

        self.model = RandomForestClassifier(random_state=6)

    def hyperparameter_tuning(self):

        parameterGrid: dict = {
            'n_estimators': [100],
            'max_depth': [2],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'bootstrap': [True, False]
        }
        
        gridSearch = GridSearchCV(RandomForestClassifier(random_state=6), param_grid=parameterGrid)
        gridSearch.fit(self.trainFeatureValue, self.trainTargetValue)
        
        print(f"Best parameters for Random Forest: {gridSearch.best_params_}")
        self.model = gridSearch.best_estimator_

    def fit_model(self):

        self.model.fit(self.trainFeatureValue, self.trainTargetValue)

    def accuracy_model(self):

        trainingPrediction = self.model.predict(self.trainFeatureValue)
        validationPrediction = self.model.predict(self.validationFeatureValue)

        trainingAccuracy = accuracy_prediction(self.trainTargetValue, trainingPrediction)
        print(f"Accuracy on Training Set Using Random Forest: {trainingAccuracy:.5f}")

        validationAccuracy = accuracy_prediction(self.validationTargetValue, validationPrediction)
        print(f"Accuracy on Validation Set Using Random Forest: {validationAccuracy:.5f}")

    def training_accuracy_model(self):

        trainingPrediction = self.model.predict(self.trainFeatureValue)
        trainingAccuracy = accuracy_prediction(self.trainTargetValue, trainingPrediction)
        return trainingAccuracy

    def validation_accuracy_model(self):

        validationPrediction = self.model.predict(self.validationFeatureValue)
        validationAccuracy = accuracy_prediction(self.validationTargetValue, validationPrediction)
        return validationAccuracy
    
class CategoricalFeatureSVMModel:

    def __init__(self, splittedTrainSet, validationSet, categoricalFeatures, targetFeature):
        
        self.splittedTrainSet = splittedTrainSet
        self.validationSet = validationSet
        self.categoricalFeatures = categoricalFeatures
        self.targetFeature = targetFeature

        self.trainFeatureValue = self.splittedTrainSet[self.categoricalFeatures].values
        self.trainTargetValue = self.splittedTrainSet[self.targetFeature].values

        self.validationFeatureValue = self.validationSet[self.categoricalFeatures].values
        self.validationTargetValue = self.validationSet[self.targetFeature].values

        self.model = svm.SVC(kernel='linear', random_state=6)

    def fit_model(self):

        self.model.fit(self.trainFeatureValue, self.trainTargetValue)

    def accuracy_model(self):
        trainingPrediction = self.model.predict(self.trainFeatureValue)
        validationPrediction = self.model.predict(self.validationFeatureValue)

        trainingAccuracy = accuracy_prediction(self.trainTargetValue, trainingPrediction)
        print(f"Accuracy on Training Set Using SVM: {trainingAccuracy:.5f}")

        validationAccuracy = accuracy_prediction(self.validationTargetValue, validationPrediction)
        print(f"Accuracy on Validation Set Using SVM: {validationAccuracy:.5f}")

    def training_accuracy_model(self):

        trainingPrediction = self.model.predict(self.trainFeatureValue)
        trainingAccuracy = accuracy_prediction(self.trainTargetValue, trainingPrediction)
        return trainingAccuracy

    def validation_accuracy_model(self):

        validationPrediction = self.model.predict(self.validationFeatureValue)
        validationAccuracy = accuracy_prediction(self.validationTargetValue, validationPrediction)
        return validationAccuracy