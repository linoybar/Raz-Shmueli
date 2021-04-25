# Import libraries:
import pandas as pd
from sklearn import metrics
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from recordclass import recordclass
from sklearn.metrics import f1_score, confusion_matrix
import re
from typing import List
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
import json
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression



rcParams['figure.figsize'] = 12, 4


class ProductionTraining(object):

    @staticmethod
    def modelfit(x, y, predictors, alg=GradientBoostingClassifier, performCV=True, printFeatureImportance=True,
                 cv_folds=5):
        # Fit the algorithm on the data
        alg.fit(x[predictors], y)

        # Predict training set:
        dtrain_predictions = alg.predict(x[predictors])

        # Perform cross-validation:
        if performCV:
            cv_score = cross_validate(alg, x[predictors], y, cv=cv_folds, scoring='f1')

        # Print model report:
        print("\nModel Report")
        print("f1 : %.4g" % metrics.f1_score(y.values, dtrain_predictions))

        if performCV:
            t_score = cv_score['test_score']
            print(f"CV Score:Mean {np.mean(t_score):.2}|Std {np.std(t_score):.2} |Min {np.min(t_score):.2}|Max {np.max( t_score): .2}")

        # Print Feature Importance:
        if printFeatureImportance:
            feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
            feat_imp.plot(kind='bar', title='Feature Importances')
            plt.ylabel('Feature Importance Score')

        return cv_score

    @staticmethod
    def read_json_to_class(json_path: str):
        with open(json_path, encoding='utf-8', errors='ignore') as json_file:
            configs = json.load(json_file, object_hook=lambda d: recordclass('X', d.keys())(*d.values()))
        return configs

    @staticmethod
    def evaluate(y_true, y_pred):
        d = {"f1": f1_score(y_true=y_true, y_pred=y_pred),
             "confusion_matrix": confusion_matrix(y_true=y_true, y_pred=y_pred)}
        return d

    @staticmethod
    def remove_characters(feature_value: str):
        '''
        The following function clean characters from strings
        :param feature_value: str
        :return: int

        example :remove_characters(feature_value ='8x') --> 8
        '''
        if isinstance(feature_value, str):
            value = int(re.search(r'\d+', feature_value).group())
            return value
        if isinstance(feature_value, int):
            return feature_value


    @staticmethod
    def run_gridsearchCV_with_pipline(X_train: pd.DataFrame,
                                      y_train: pd.Series,
                                      numerical_cols: List[str],
                                      categorical_cols: List[str],
                                      bool_cols: List[str],
                                      fixed_params: dict,
                                      gs_params: dict):
        '''
        Run grid search with cross validation pipeline
        :param X_train: pd.DataFrame - feature matrix
        :param y_train: pd.Series - target vactore
        :param numerical_cols: List[str] - numerical features
        :param categorical_cols: List[str] - Categorical features
        :param bool_cols: List[str] - Boolean features
        :param fixed_params: dict - not for grid search
        :param gs_params: dict -  for grid search
        :return: GridSearchCV model (contain best model params)
        '''
        categorical_transformer = Pipeline(steps=
                                           [('imputer', SimpleImputer(strategy='most_frequent')

                                             ),
                                            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        continuous_transformer = Pipeline(steps=
                                          [('imputer', SimpleImputer(strategy='median')),
                                           ('scaler', StandardScaler())])

        boll_transformer = Pipeline(steps=
                                    [('imputer', SimpleImputer(strategy='most_frequent'))]
                                    )

        preprocessor = ColumnTransformer(transformers=
                                         [('num', continuous_transformer, numerical_cols),
                                          ('cat', categorical_transformer, categorical_cols),
                                          ('bool', boll_transformer, bool_cols)]
                                         )

        model = GradientBoostingClassifier(**fixed_params)
        #model = LogisticRegression(**fixed_params)

        pipeline = Pipeline(steps=
                            [('preprocess', preprocessor),
                             ('model', model)])

        grid = GridSearchCV(pipeline,
                            param_grid=gs_params, scoring='f1', cv=3, n_jobs=4, iid=False
                            )
        grid = grid.fit(X_train * 1, y_train)
        return grid

    @staticmethod
    def remove_characters(feature_value: str):
        '''
        The following function clean characters from strings
        :param feature_value: str
        :return: int

        example :remove_characters(feature_value ='8x') --> 8
        '''
        if isinstance(feature_value, str):
            value = int(re.search(r'\d+', feature_value).group())
            return value
        if isinstance(feature_value, int):
            return feature_value