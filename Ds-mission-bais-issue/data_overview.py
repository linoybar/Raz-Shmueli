import pandas as pd


class DataOverview(object):

    @staticmethod
    def nan_counter(df: pd):
        '''
        Shows count and percentage of nan for columns with nan
        :param df: pd.DataFrame
        :return: pd.DataFrame example
        	                     nan_count	 an_percenatge
            square_ft	             12118   	0.977495
            portable_electronics     407	    0.032831
            coast	                 1155   	0.093168
            user_age    	          82	    0.006615
            card_type   	          52	    0.004195
        '''
        # Count number of nan
        df_na = pd.DataFrame(df.isna().sum(), columns=['nan_count'])

        # Calculate percentage of nans
        df_na['nan_percenatge'] = df_na.nan_count / df.shape[0]

        # Filter when there is nan
        df_na = df_na[df_na.nan_count > 0]
        return df_na

    @staticmethod
    def value_counts_and_percentage(df: pd, column: str):
        return pd.DataFrame({'count': df[column].value_counts(), '%': df[column].value_counts(normalize=True)})
