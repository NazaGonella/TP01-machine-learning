import pandas as pd
import numpy as np

def correct_data_types(df : pd.DataFrame) -> pd.DataFrame:
    _df : pd.DataFrame = df.copy()                                            # Se eliminan las filas con elementos nulos
    # Convierte al mejor tipo.
    _df = _df.convert_dtypes()
    # Convierte 0s y 1s a valores booleanos.
    _df['is_house'] = _df['is_house'].astype(bool)
    _df['has_pool'] = _df['has_pool'].astype(bool)
    return _df

def remove_na_rows(df : pd.DataFrame) -> pd.DataFrame:
    _df : pd.DataFrame = df.copy()
    _df = _df.dropna(inplace=False)
    return _df

def convert_area_units(df : pd.DataFrame, area_unit : str = 'm2') -> pd.DataFrame:
    _df : pd.DataFrame = df.copy()
    match area_unit:
        case 'm2':
            _df.loc[_df['area_units'] == 'sqft', 'area'] = _df[_df['area_units'] == 'sqft']['area'].apply(lambda x : np.rint(x / 10.7639))
            _df = _df.replace('sqft', 'm2')
        case 'sqft':
            _df.loc[_df['area_units'] == 'm2', 'area'] = _df[_df['area_units'] == 'm2']['area'].apply(lambda x : np.rint(x * 10.7639))
            _df = _df.replace('m2', 'sqft')
    return _df

def standarize_numeric_columns(df : pd.DataFrame, excluded_columns : set[str] = set()) -> pd.DataFrame:
    numeric_columns : list[str] = ['area', 'age', 'price', 'lat', 'lon', 'rooms']
    _df : pd.DataFrame = df.copy()
    for col in numeric_columns:
        #print(_df)
        if col in excluded_columns:
            continue
        # print(col, "| ", f"mean: {_df[col].mean()} - std: {_df[col].std()}")
        _df[col] = ((_df[col] - _df[col].mean()) / _df[col].std())
    return _df