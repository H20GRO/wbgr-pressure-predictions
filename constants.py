import os
import pandas as pd

DATA_DIR = 'Data'
RESULTS_DIR = 'Results'
RAW_DIR = os.path.join(DATA_DIR, 'Raw')
RESAMPLED_DIR = os.path.join(DATA_DIR, 'Resampled')
CONFIG_EXCEL = 'Configuratie.xlsx'
AREAS = {'Provincie':'ProvincieVerbruik', 'Stad':'StadVerbruik','Lauwersoog':'LauwersoogVerbruik'}
MIN_PRESSURE,MAX_PRESSURE = 100,600

def _get_config_wrapper():
    df_config = None
    def get_config_data() -> pd.DataFrame:
        """Get confgiuration data from an excel file"""
        nonlocal df_config
        if df_config is None:
            df_config = pd.read_excel(CONFIG_EXCEL,sheet_name='Vertaling', usecols='C:K',header=0, skiprows=1)
        return df_config         
    return get_config_data

def _get_validation_wrapper():
    df_config = None
    def get_config_data() -> pd.DataFrame:
        """Get validation data from an excel file"""
        nonlocal df_config
        if df_config is None:
            df_config = pd.read_excel(CONFIG_EXCEL,sheet_name='Validatie',skiprows=1,usecols='C:Z',index_col=0) 
        return df_config         
    return get_config_data

DF_CONFIG = _get_config_wrapper()()
DF_VALIDATION = _get_validation_wrapper()()

if __name__ == "__main__":
    print(DF_CONFIG)
    print(DF_VALIDATION)



 