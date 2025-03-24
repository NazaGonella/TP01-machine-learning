import pandas as pd
import numpy as np
# import src.preprocessing as prepro
import preprocessing as prepro

def get_train_and_validation_sets(df : pd.DataFrame, train_fraction : float = 0.8, seed : int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    train : pd.DataFrame = df.sample(frac=train_fraction,random_state=seed)
    validation : pd.DataFrame = df.drop(train.index)
    return train, validation

def get_null_rows(df : pd.DataFrame) -> int:
    return df.isna().T.any().sum()

def are_data_types_uniform(df : pd.DataFrame) -> bool:
    for col in df.columns.to_list():
        if df['price'].map(type).nunique() != 1:
            return False
    return True

class RawData:
    def __init__(self):
        self.casas_dev : pd.DataFrame = pd.read_csv('/home/nazar/UDESA/5toCUATRIMESTRE/ML/TPs/TP01/data/raw/casas_dev.csv')
        self.casas_test : pd.DataFrame = pd.read_csv('/home/nazar/UDESA/5toCUATRIMESTRE/ML/TPs/TP01/data/raw/casas_test.csv')
        self.vivienda_amanda : pd.DataFrame = pd.read_csv('/home/nazar/UDESA/5toCUATRIMESTRE/ML/TPs/TP01/data/raw/vivienda_amanda.csv')

class ProcessedData:
    def __init__(self, correct_data_types : bool = True, standarize : bool = True, area_units : str = 'm2', remove_na_rows : bool = False):
        self.casas_dev : pd.DataFrame = pd.read_csv('/home/nazar/UDESA/5toCUATRIMESTRE/ML/TPs/TP01/data/raw/casas_dev.csv')
        if remove_na_rows:
            self.casas_dev : pd.DataFrame = prepro.remove_na_rows(self.casas_dev)
        if correct_data_types:
            self.casas_dev : pd.DataFrame = prepro.correct_data_types(self.casas_dev)
        if area_units == 'm2' or area_units == 'sqft':
            self.casas_dev = prepro.convert_area_units(self.casas_dev, area_units)
        if standarize:
            self.casas_dev = prepro.standarize_numeric_columns(self.casas_dev)
    
    def fill_missing_values(self, method : str = 'median') -> None:
        match method:
            case 'median':
                self.casas_dev['age'] = self.casas_dev['age'].fillna(value=self.casas_dev['age'].median())
                self.casas_dev['rooms'] = self.casas_dev['rooms'].fillna(value=self.casas_dev['rooms'].median())
            case 'mean':
                self.casas_dev['age'] = self.casas_dev['age'].fillna(value=int(round(self.casas_dev['age'].mean())))
                self.casas_dev['rooms'] = self.casas_dev['rooms'].fillna(value=int(round(self.casas_dev['rooms'].mean())))
    
    def save_data(self, path : str = 'data/processed', ext : str = ''):
        self.casas_dev.to_csv(f'{path}/casas_dev_processed{'_' + ext if ext else ''}.csv')