import enum
import math
from typing import List, Tuple
import pandas as pd
import numpy as np
from prepare_data import get_config_data,get_raw_data, get_validation_data, get_resampled_data
import logging as logger
import os
import sys
from constants import *
from logger import get_logger


log = get_logger(__name__)


class Validation(enum.Enum):
    Good = 1
    AboveMax = 2
    BelowMin = 3
    Constant = 4
    GlobalError = 5
    SignalError = 6
    NaN = 7

def validate_elementwise(column:pd.Series):
    ret = pd.Series(np.ones(len(column))*Validation.Good.value)
    if column.name in  DF_CONFIG.query('Type == "Pressure"')['Name'].values:
        filterMax = column.values>MAX_PRESSURE
        ret[filterMax] = Validation.AboveMax.value
        filterMin = column.values<MIN_PRESSURE
        ret[filterMin] = Validation.BelowMin.value       
    ret[column.diff().values == 0] = Validation.Constant.value
    ret[column.isna().values] = Validation.NaN.value 
    return ret

def validate_dataframe(df:pd.DataFrame) -> pd.DataFrame:
    """Validat DF"""   
    df_Validated = df.apply(validate_elementwise)
    df_Validated.index = df.index # needed for next step    
    # time based validation
    #tagMetaData = df_Configuration[df_Configuration['Name'].isin(df.columns)]
    # genericValidations = df_Validation[df_Validation['Signal'].isin(tagMetaData['Type'].unique())]
    # for _, row in genericValidations.reset_index().iterrows():
    #     filter = (df.index >= row['StartDate']) & (df.index <= row['EndDate'])
    #     df_Validated[filter] = Validation.SignalError.value 
    # #and NaN    
    return df_Validated 

def get_validated_data() -> pd.DataFrame:
    """Get validated tags by time"""
    return pd.read_parquet((f'{RESAMPLED_DIR}/validation.parquet'))

def calculate_flow_balance(df_resampled:pd.DataFrame) -> pd.DataFrame:
    """Calculate the flow balance for all DMAs, adds missing data columns as NaN """
    tags_not_available = DF_CONFIG[(~DF_CONFIG['Name'].isin(df_resampled.columns)) & (DF_CONFIG['Type'] == 'Flow')]
    if len(tags_not_available) > 0:
        log.info(f"Following tags not found {tags_not_available['Name'].to_list()}. Will be initialized to NaN")    
        df_resampled[tags_not_available['Name'].to_list()] = np.NAN    
    df_balance = pd.DataFrame()
    df_balance.index = df_resampled.index
    for (key, val) in AREAS.items():
        positive_signals = DF_CONFIG[DF_CONFIG['Destination'] == key]
        negative_signals = DF_CONFIG[DF_CONFIG['Source'] == key]
        df_balance[val] = df_resampled[positive_signals['Name']].sum(axis=1) - df_resampled[negative_signals['Name']].sum(axis=1)
    return df_balance

def get_balance_data() -> pd.DataFrame:
    """Get DMA flows"""
    return pd.read_parquet(f'{RESAMPLED_DIR}/balance.parquet')

def main() -> None:
    """Validating data and calculating flow balance"""
    df_resampled = get_resampled_data()
    df_validated = validate_dataframe(df_resampled)
    df_balance = calculate_flow_balance(df_resampled)    
    df_validated.to_parquet(f'{RESAMPLED_DIR}/validation.parquet')
    log.info("Succesfully stored validation data")
    df_balance.to_parquet(f'{RESAMPLED_DIR}/balance.parquet')
    log.info("Succesfully stored balance data")


def test() -> None:
    pass

if __name__ == '__main__':
    TEST = False
    if TEST:
        test()
    else:
        main()