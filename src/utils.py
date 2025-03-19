import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class RawData:
    casas_dev : pd.DataFrame = pd.read_csv('/home/nazar/UDESA/5toCUATRIMESTRE/ML/TPs/TP01/TP01_datasets/casas_dev.csv')
    casas_test : pd.DataFrame = pd.read_csv('/home/nazar/UDESA/5toCUATRIMESTRE/ML/TPs/TP01/TP01_datasets/casas_test.csv')
    vivienda_amanda : pd.DataFrame = pd.read_csv('/home/nazar/UDESA/5toCUATRIMESTRE/ML/TPs/TP01/TP01_datasets/vivienda_amanda.csv')

class Utils:
    @staticmethod
    def get_null_rows(df : pd.DataFrame) -> int:
        return df.isna().T.any().sum()
    
    @staticmethod
    def correct_data_types(df : pd.DataFrame, remove_na_rows : bool = False) -> pd.DataFrame:
        _df = df.copy()
        if remove_na_rows:
            df.dropna(inplace=True)                                              # Se eliminan las filas con elementos nulos
        # Convierte al mejor tipo.
        _df = _df.convert_dtypes()
        # Convierte 0s y 1s a valores booleanos.
        _df['is_house'] = _df['is_house'].astype(bool)
        _df['has_pool'] = _df['has_pool'].astype(bool)
        return _df
    
    @staticmethod
    def convert_area_units(df : pd.DataFrame, area_unit : str = 'm2') -> pd.DataFrame:
        _df = df.copy()
        match area_unit:
            case 'm2':
                _df.loc[_df['area_units'] == 'sqft', 'area'] = _df[_df['area_units'] == 'sqft']['area'].apply(lambda x : np.rint(x / 10.7639))
            case 'sqft':
                _df.loc[_df['area_units'] == 'm2', 'area'] = _df[_df['area_units'] == 'm2']['area'].apply(lambda x : np.rint(x * 10.7639))
        return _df
    
    @staticmethod
    def normalize_numeric_columns(df : pd.DataFrame, excluded_columns : set[str] = set()) -> pd.DataFrame:
        numeric_columns : list[str] = ['area', 'age', 'price', 'lat', 'lon', 'rooms']
        _df = df.copy()
        for col in numeric_columns:
            if col in excluded_columns:
                continue
            # print(col, "| ", f"mean: {_df[col].mean()} - std: {_df[col].std()}")
            _df[col] = ((_df[col] - _df[col].mean()) / _df[col].std())
        return _df
    
    @staticmethod
    def get_train_and_validation_sets(df : pd.DataFrame, train_fraction : float = 0.8, seed : int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        train : pd.DataFrame = df.sample(frac=train_fraction,random_state=seed)
        validation : pd.DataFrame = df.drop(train.index)
        return train, validation
