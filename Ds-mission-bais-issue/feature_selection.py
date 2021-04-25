from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix


class FeatureSelection(object):

    def __init__(self):
        pass

    @staticmethod
    def scan_feature_sensitivity(df, labeled_col_name: str, N_number_of_splits: int, test_size: float):
        '''
        Run N svm model for each feature. each i's model split randomly test and train.
        Returning features that return 1 on the test set.
        Make sure categorical features are out.
        :param N_number_of_splits: int - number of models per feature
        :param df: pd.DataFrame
        :param labeled_col_name: str - name of tearget column
        :return:dict
        '''

        X = df.drop(columns=[labeled_col_name])
        y = df[labeled_col_name]

        d = {}
        for i in range(N_number_of_splits):
            if i % 5 == 0:
                print(f'{N_number_of_splits - i} iterations left ...')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
            for col in X_train.columns:
                df_i_train = X_train[[col]]
                df_i_test = X_test[[col]]
                clf = SVC(gamma='auto', kernel='rbf')
                clf.fit(df_i_train, y_train)
                predicted = clf.predict(df_i_test)
                s = sum(predicted == 1)
                if s > 0:
                    if col not in d:
                        d[col] = 1
                    else:
                        d[col] += 1
        return d

    @staticmethod
    def find_best_comb(df: pd, ordered_dict: dict, test_size: float, labeled_col_name: str = 'label'):
        '''
        Step wise feature addition. for each comb calculate confusion matrix and recall
        :param df:pd.DataFrame
        :param ordered_dict:dict - dictionary orderd by the features we want to add first
        :param test_size: float
        :param labeled_col_name: str
        :return: - dict - each key represent features combination each value is dict of confusion matrix and recall
        '''
        ordered_dict = {k: v for k, v in sorted(ordered_dict.items(), key=lambda item: item[1], reverse=True)}
        X = df.drop(columns=[labeled_col_name])
        y = df[labeled_col_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
        d = {}
        all_choswen_columns = list(ordered_dict.keys())
        for i in range(2, len(ordered_dict)):
            cols = all_choswen_columns[0:i]
            df_i_train = X_train[cols]
            df_i_test = X_test[cols]
            clf = SVC(gamma='auto', kernel='rbf')
            clf.fit(df_i_train, y_train)
            predicted = clf.predict(df_i_test)
            s = sum(predicted == 1)
            if s > 0:
                conf_metrix = confusion_matrix(y_test, predicted)
                d[','.join(cols)] = {'f1': f1_score(y_test, predicted),
                                     'conf_matrix': conf_metrix}
        return d

    @staticmethod
    def generate_feature_importance(df: pd, labeled_col_name: str):
        '''
        Run feature importance process using sklearn random forest
        :param df:pd
        :param labeled_col_name:str
        :return:pd.DataFrame, features ordered from the most important to the less
        '''

        X = df.drop(columns=[labeled_col_name])
        y = df[labeled_col_name]

        # Build a forest and compute the impurity-based feature importances
        forest = ExtraTreesClassifier(n_estimators=250,
                                      random_state=0)

        forest.fit(X, y)

        d = {'feature': [], 'importance': []}
        for feat, importance in zip(X.columns, forest.feature_importances_):
            d['feature'].append(feat)
            d['importance'].append(importance)

        df_feature_importence = pd.DataFrame(d)
        df_feature_importence = df_feature_importence.sort_values('importance', ascending=False)
        return df_feature_importence

    @staticmethod
    def run_lasso_lr(df: pd, labeled_col_name: str, C: float = 1, max_iter: int = 100):
        '''
        Run logistic regression with L1 regularization as a part of the feature selection process
        :param df: pd.DataFrame
        :param labeled_col_name: str
        :param C: float - regularization parameters
        :param max_iter: int
        :return: pd.df with feature names and coefficients ordered by coefficients values descending
        '''
        X = df.drop(columns=[labeled_col_name])
        y = df[labeled_col_name]
        clf = LogisticRegression(C=C,
                                 penalty='l1',
                                 solver='liblinear',
                                 random_state=0,
                                 max_iter=max_iter).fit(X, y)
        df_lasso_coeffs = pd.DataFrame({'feature': X.columns, 'coefs': clf.coef_[0]})
        df_lasso_coeffs = df_lasso_coeffs.sort_values(by='coefs', ascending=False)
        return df_lasso_coeffs
