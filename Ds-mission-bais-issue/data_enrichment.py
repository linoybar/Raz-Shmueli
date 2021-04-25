import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.cluster import KMeans
from typing import List
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataEnrichment(object):

    @staticmethod
    def resample_continuous_feautres(df: pd, labeled_col_name: str, n: int, resample_on: bool):
        '''
        Resampeling continuous features and adding normal nois
        :param labeled_col_name:str
        :param n:int - number of resampeling
        :param resample_on: bool - on 1 or 0
        :return: pd - resampeld_df
        '''
        df_t = df[df[labeled_col_name] == resample_on]
        d_i = df_t[:]
        row_n = df_t.shape[0]
        cols = df_t.columns[df_t.dtypes == 'float64']

        for i in range(n):
            d = pd.DataFrame({c: np.random.normal(0, df_t[c].std(), row_n).tolist() for c in cols})
            d_i[cols] = df_t[cols][:] + d[cols][:]
            df = df.append(d_i)

        return df

    @staticmethod
    def resample_discrete_features(df: pd, labeled_col_name: str, n: int, resample_on: bool):
        '''
        Resampeling continuous features and adding random noise from discrete uniform distribution {-2,...,2}
        :param labeled_col_name:str
        :param n:int - number of resampeling
        :param resample_on: bool - on 1 or 0
        :return: pd - resampeld_df
        '''
        df_t = df[df[labeled_col_name] == resample_on]
        d_i = df_t[:]
        row_n = df_t.shape[0]
        cols = df_t.columns[df_t.dtypes == 'float64']

        for i in range(n):
            d = pd.DataFrame({c: np.random.randint(-2, 2, row_n).tolist() for c in cols})
            d_i[cols] = df_t[cols][:] + d[cols][:]
            df = df.append(d_i)

        return df

    @staticmethod
    def resample_and_add_noise(df: pd, labeled_col_name: str, n: int, resample_on: bool):
        '''
        Resampeling continuous features and adding noise
        For continuous faetures adding normal noise with mu = 0 and sigma  = featue standard deviation
        For discrete features adding random noise from discrete uniform distribution {-2,...,2}

        :param labeled_col_name:str
        :param n:int - number of resampeling
        :param resample_on: bool - on 1 or 0
        :return: pd - resampeld_df
        '''
        from copy import deepcopy

        df_t = deepcopy(df[df[labeled_col_name] == resample_on])
        d_i = deepcopy(df_t[:])
        row_n = df_t.shape[0]
        cols_cont = df_t.columns[df_t.dtypes == 'float64']
        cols_discrete = df_t.columns[df_t.dtypes == 'int64']

        for i in range(n):
            d_cont = pd.DataFrame({c: np.random.normal(0, df_t[c].std(), row_n).tolist() for c in cols_cont},
                                  index=df_t.index)
            d_i[cols_cont] = df_t[cols_cont][:] + d_cont[cols_cont][:]

            d_discrete = pd.DataFrame({c: np.random.randint(-2, 2, row_n).tolist() for c in cols_discrete},
                                      index=df_t.index)
            d_i[cols_discrete] = df_t[cols_discrete][:] + d_discrete[cols_discrete][:]

            df = df.append(d_i)

        return df

    @staticmethod
    def resample_df(df: pd, number_of_resamples: int):
        '''
        Resampling data frame by union itself for given number of times
        :param df: pd
        :param number_of_resamples: int number of times for union
        :return: pd
        '''
        df_i = df[:]
        for i in range(number_of_resamples):
            df = df.append(df_i)[:]
        return df

    @staticmethod
    def shuffle_features(df: pd, columns_for_shuffeling: list):
        '''
        Rndom shuffeling of features in each raw
        :param df: pd
        :param columns_for_shuffeling: :List[str] columns we interested to shuffle
        :return: pd - shuffled
        '''
        df_i = deepcopy(df)
        for col in columns_for_shuffeling:
            df_i[col] = np.random.choice(df_i[col], size=df_i.shape[0]).tolist()
        return df_i

    @staticmethod
    def generate_k_means_clusters(df, cols_to_ignore: list, k_groups: int):
        kmeans = KMeans(n_clusters=k_groups, random_state=0).fit(df.drop(columns=cols_to_ignore))
        df['k_m_label'] = kmeans.predict(df.drop(columns=cols_to_ignore))
        return df

    @staticmethod
    def clustering_with_Kmeans(data: pd, cols_to_ignore: List[str], k_clusters: int):
        '''
        Cluster given data with pipelined K-means algorithm
        :param data:pd.DatafRAME
        :param cols_to_ignore:List[str] - Not to use columns
        :param k_clusters:int - number of K-means clusters
        :return:pandas DataFrame with additional k_m_label label
        '''
        # Convet bools to integers
        data = data * 1

        # Categoricla and numeric
        categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
        naumerical_cols = [col for col in data.columns if (data[col].dtype == 'int64' or data[col].dtype == 'float64')]

        # Pipe lin steps
        categorical_transformer = Pipeline(steps=
                                           [('imputer', SimpleImputer(strategy='most_frequent')),
                                            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        continuous_transformer = Pipeline(steps=
                                          [('imputer', SimpleImputer(strategy='median')),
                                           ('scaler', MinMaxScaler())])

        preprocessor = ColumnTransformer(transformers=
                                         [('num', continuous_transformer, naumerical_cols),
                                          ('cat', categorical_transformer, categorical_cols),
                                          ])
        clusterer = Pipeline(
            [
                (
                    "kmeans",
                    KMeans(
                        n_clusters=k_clusters,
                        init="k-means++",
                        n_init=50,
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        )
        pipe = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("clusterer", clusterer)
            ]
        )
        k_means_model = pipe.fit(data)
        data['k_m_label'] = k_means_model.predict(data.drop(columns=cols_to_ignore))
        return data
