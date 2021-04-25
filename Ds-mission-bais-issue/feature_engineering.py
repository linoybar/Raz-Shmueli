import pandas as pd
import json
import re


class FeatureEngineering(object):

    @staticmethod
    def map_col(map_path: str, df: pd, col_to_map: str, new_mapped_col_name: str):
        '''
        Map values in col
        :param map_path: str
        :param df: pd.DataFrame
        :param col_to_map: str
        :param new_mapped_col_name: str
        :return: same pd.DataFrame but with mapped column
        '''
        with open(map_path) as f:
            data = json.load(f)
        df[new_mapped_col_name] = df[col_to_map].map(data)
        return df

    @staticmethod
    def remove_duplicated_columns(df: pd):
        '''
        Remove duplicates columns in case of evidently there is
        :param df: pd.DataFrame
        :return: pd.DataFrame
        '''
        duplicated_columns_name = df.columns[df.T.duplicated().tolist()]
        df = df.drop(duplicated_columns_name, axis=1)
        print(f'The following duplicated columns droped {duplicated_columns_name.tolist()}')
        return df

    @staticmethod
    def convert_nans_by_column(df: pd, nan_col: str, reference_col: str, agg_type: str, threshold: float = None):
        '''
        Fill numeric column with na with respect to other columns (mimic the results of sklearn imputer)
        :param df:pd.DataFrame
        :param nan_col:str
        :param reference_col:str
        :param agg_type: str - mean/median/nax/min ....
        :param threshold: for convert to 1
        :return: pd.DataFrame
        '''
        new_col_name = f'{nan_col}_{agg_type}'
        df_postal_code_median_ft = df[[nan_col, reference_col]].groupby([reference_col]).agg(agg_type)
        df_postal_code_median_ft = df_postal_code_median_ft[~df_postal_code_median_ft[nan_col].isna()].rename(
            columns={nan_col: new_col_name})
        if threshold:
            df_postal_code_median_ft[new_col_name] = df_postal_code_median_ft[new_col_name] > threshold

        d = df[[reference_col, nan_col]].merge(df_postal_code_median_ft, how='left', on=reference_col).set_index(
            df.index)
        d[new_col_name] = d[new_col_name].astype(float)

        d.loc[:, nan_col][d[nan_col].isna()] = d[d[nan_col].isna()][new_col_name]
        return d[[nan_col]]

    @staticmethod
    def fill_nans_by_highest_reference_frequency(df: pd, col_to_fill: str, refernece_col: str):
        '''
        Fill numeric column with na with respect frequency appearance  of other columns (mimic the results of sklearn imputer)
        :param df: pd.DataFrame
        :param col_to_fill: str
        :param refernece_col: str
        :return: pd.DataFrame
        '''
        df_card = df[[col_to_fill, refernece_col]]
        df_agg = df[[refernece_col, col_to_fill, 'id']].groupby([refernece_col, col_to_fill]).count()

        g = df_agg['id'].groupby(refernece_col, group_keys=False).nlargest(1)
        dct = {k: v for k, v in g.index}
        states_with_na = df_card[df_card[col_to_fill].isna()][refernece_col].unique()
        for s in states_with_na:
            if s in dct.keys():
                df_card.loc[(df_card[refernece_col] == s) & (df_card[col_to_fill].isna()), :] = dct[s]

        return df_card

    @staticmethod
    def pull_integers_from_str_column(df: pd, col_to_convert: str):
        '''
        Convert str column to integers by removing characters
        :param df: pd.DataFrame
        :param col_to_convert: str
        :return: pd.DataFrame
        '''
        df_fire_housing_proximity = df[[col_to_convert]][:]
        df_fire_housing_proximity[col_to_convert] = [re.search(r'\d+', i).group() for i in df.fire_housing_proximity]
        df_fire_housing_proximity[col_to_convert] = df_fire_housing_proximity[col_to_convert].astype(int)
        return df_fire_housing_proximity
