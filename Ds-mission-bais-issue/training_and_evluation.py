import pandas as pd
from sklearn.model_selection import train_test_split
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


class TrainingAndEvaluation(object):

    @staticmethod
    def train_test_split(df: pd, labeled_col_name: str, test_size: float, random_state: int):
        '''
        Train test random split
        :param df: pd.df
        :param labeled_col_name: str
        :param test_size: float
        :param random_state: int
        :return: tuple of 4 indexes - X_train, X_test, y_train, y_test
        '''
        X = df.drop(columns=[labeled_col_name])
        y = df[labeled_col_name]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    @staticmethod
    def generate_f1_and_confusion_matrix(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                                         y_test: pd.Series, df_col_types: pd.DataFrame):
        '''
        Train 5 separate models on test set and test on test set, return data frame of f1 and confusion matrices
        :param X_train:pd.DataFrame
        :param X_test:c
        :param y_train:pd.Series
        :param y_test:pd.Series
        :return:pd.Series - f1 and confusion matricess

        '''
        # Convert bool to int
        X_train *= 1
        X_test *= 1
        y_train *= 1
        y_test *= 1

        # Map columns to types : Categorical, Numeric, Boolean
        df = df_col_types[:]
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
        numerical_cols = [col for col in df.columns if ((df[col].dtype == 'int64') or (df[col].dtype == 'float64'))]
        bool_cols = [col for col in df.columns if df[col].dtype == 'bool' if 'label' != col]

        # Assign pipline step for each column type
        categorical_transformer = Pipeline(steps=
                                           [('imputer', SimpleImputer(strategy='most_frequent')),
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

        # Define params for each classifier type
        grid_params_lr = [{'clf__penalty': ['l1', 'l2'],
                           'clf__C': [0.5, 1.5],
                           'clf__solver': ['liblinear']}]

        grid_params_svm = [{'clf__kernel': ['linear', 'rbf'],
                            'clf__C': [0.5, 1.5]}]

        grid_params_gb = [{'clf__n_estimators': [150],
                           'clf__learning_rate': [0.5, 1.5],
                           'clf__max_features': ['auto', 'sqrt'],
                           'clf__loss': ['exponential', 'deviance']}]

        cls_params = [(LogisticRegression(), grid_params_lr),
                      (SVC(), grid_params_svm),
                      (GradientBoostingClassifier(), grid_params_gb)]

        d = {}
        models = {}
        # Run Grid search for each classifier
        for classifier, grid_params in cls_params:
            pipeline = Pipeline(steps=
                                [('preprocess', preprocessor),
                                 ('clf', classifier)])

            model = GridSearchCV(pipeline, param_grid=grid_params,
                                 scoring='f1', cv=3, n_jobs=4
                                 )

            model = model.fit(X_train, y_train)

            y_test_pred, y_train_pred = model.predict(X_test), model.predict(X_train)

            d[str(classifier)] = [f1_score(y_true=y_test, y_pred=y_test_pred),
                                  f1_score(y_true=y_train, y_pred=y_train_pred),
                                  confusion_matrix(y_true=y_test, y_pred=y_test_pred)]
            models[str(classifier)] = model

        # Summarizing results into pandas dataFrame
        df_r = pd.DataFrame.from_dict(d, orient='index', columns=['f1_test', 'f1_train', 'confusion_matrix_test'])
        df_r['train_test_ratio'] = df_r.f1_test / df_r.f1_train
        df_r = df_r[['f1_test', 'f1_train', 'train_test_ratio', 'confusion_matrix_test']]
        return df_r, models

    @staticmethod
    def read_list(path: str):
        # define an empty list
        l = []
        # open file and read the content in a list
        with open(path, 'r') as filehandle:
            for line in filehandle:
                currentPlace = line[:-1]
                l.append(currentPlace)

        return l


