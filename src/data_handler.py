import pandas as pd
import src.preprocessing as prepro

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
    def __init__(self, correct_data_types : bool = True, normalize : bool = True, area_units : str = 'm2', remove_na_rows : bool = False):
        self.casas_dev : pd.DataFrame = pd.read_csv('/home/nazar/UDESA/5toCUATRIMESTRE/ML/TPs/TP01/data/raw/casas_dev.csv')
        self.casas_test : pd.DataFrame = pd.read_csv('/home/nazar/UDESA/5toCUATRIMESTRE/ML/TPs/TP01/data/raw/casas_test.csv')
        self.vivienda_amanda : pd.DataFrame = pd.read_csv('/home/nazar/UDESA/5toCUATRIMESTRE/ML/TPs/TP01/data/raw/vivienda_amanda.csv')
        if remove_na_rows:
            self.casas_dev : pd.DataFrame = prepro.remove_na_rows(self.casas_dev)
            self.casas_test : pd.DataFrame = prepro.remove_na_rows(self.casas_test)
            self.vivienda_amanda : pd.DataFrame = prepro.remove_na_rows(self.vivienda_amanda)
        if correct_data_types:
            self.casas_dev : pd.DataFrame = prepro.correct_data_types(self.casas_dev)
            self.casas_test : pd.DataFrame = prepro.correct_data_types(self.casas_test)
            self.vivienda_amanda : pd.DataFrame = prepro.correct_data_types(self.vivienda_amanda)
        if area_units == 'm2' or area_units == 'sqft':
            self.casas_dev = prepro.convert_area_units(self.casas_dev, area_units)
            self.casas_test = prepro.convert_area_units(self.casas_test, area_units)
            self.vivienda_amanda = prepro.convert_area_units(self.vivienda_amanda, area_units)
        if normalize:
            self.casas_dev = prepro.normalize_numeric_columns(self.casas_dev, excluded_columns={'price'})
            self.casas_test = prepro.normalize_numeric_columns(self.casas_test, excluded_columns={'price'})
            self.vivienda_amanda = prepro.normalize_numeric_columns(self.vivienda_amanda, excluded_columns={'price'})
    
    def save_data(self, path : str = 'data/processed'):
        self.casas_dev.to_csv(f'{path}/casas_dev_processed.csv')
        self.casas_test.to_csv(f'{path}/casas_test_processed.csv')
        self.vivienda_amanda.to_csv(f'{path}/vivienda_amanda_processed.csv')