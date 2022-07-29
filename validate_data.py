import enum
import math
from typing import List, Tuple
import pandas as pd
import numpy as np
from prepare_data import get_config_data,get_raw_data, get_validation_data, DATADIR
import logging as logger
import os
import sys
from datetime import timedelta

CLEANED_RESAMPLED_DIR = os.path.join(DATADIR,'CLEANED')
LOG_DIR = 'Log'
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
handlers = [
        logger.FileHandler(os.path.join(LOG_DIR,'log.log')),
        logger.StreamHandler(sys.stdout)
    ]
logger.basicConfig(format='%(asctime)s - %(message)s',handlers=handlers, level=logger.INFO)
MIN_PRESSURE = 100
MAX_PRESSURE = 600
AREAS = {'Provincie':'ProvincieVerbruik', 'Stad':'StadVerbruik','Lauwersoog':'LauwersoogVerbruik'}
RESAMPLING_RESOLUTION = timedelta(minutes=5)

class Validation(enum.Enum):
    Good = 1
    AboveMax = 2
    BelowMin = 3
    Constant = 4
    GlobalError = 5
    SignalError = 6
    NaN = 7

def validate_elementwise(column:pd.Series, minVal, maxVal):
    ret = pd.Series(np.ones(len(column))*Validation.Good.value)
    filterMax =column.values>maxVal
    ret[filterMax] = Validation.AboveMax.value
    filterMin =column.values<minVal
    ret[filterMin] = Validation.BelowMin.value       
    return ret

def validate_dataframe(df:pd.DataFrame,df_Configuration:pd.DataFrame,df_Validation:pd.DataFrame, absolute_min:float, absolute_max:float) -> pd.DataFrame:
    """TODO!!!!"""
    # value based validation
    df_Validated = df.apply(validate_elementwise, minVal=absolute_min, maxVal=absolute_max)
    df_Validated.index = df.index # needed for next step
    df_Validated[df.pct_change() == 0] = Validation.Constant.value
    # time based validation
    tagMetaData = df_Configuration[df_Configuration['Name'].isin(df.columns)]
    genericValidations = df_Validation[df_Validation['Signal'].isin(tagMetaData['Type'].unique())]
    for _, row in genericValidations.reset_index().iterrows():
        filter = (df.index >= row['StartDate']) & (df.index <= row['EndDate'])
        df_Validated[filter] = Validation.SignalError.value 
    #and NaN
    df_Validated[df.isna()] = Validation.NaN.value   
    return df_Validated 

def perform_resampling_and_pivoting(df:pd.DataFrame, resolution:timedelta = RESAMPLING_RESOLUTION) -> pd.DataFrame:
    """Return resampled using the last value(close in ohlc) in the 5 min bin and with the signals name as columns!"""
    return df.groupby([df.dsg_tag, pd.Grouper(key='PtimeStamp',freq=RESAMPLING_RESOLUTION)])\
            .ohlc()\
            .reset_index()\
            .pivot(index='PtimeStamp',columns='dsg_tag',values=('fmt_meetwaarde','close'))\
            .ffill()
    # return df.groupby([df.dsg_tag, pd.Grouper(key='PtimeStamp',freq='5T')])\
    #     .mean()\
    #     .reset_index()\
    #     .pivot(index='PtimeStamp',columns='dsg_tag',values='fmt_meetwaarde')
    #maybe use weighted average
    #df2 = df.pivot_table(index='PtimeStamp',columns='dsg_tag',values='fmt_meetwaarde',aggfunc='first')

def calculate_flow_balance(df_config:pd.DataFrame, df_resampled:pd.DataFrame) -> pd.DataFrame:
    """Calculate the flow balance for all DMAs, adds missing data columns as NaN """
    tags_not_available = df_config[(~df_config['Name'].isin(df_resampled.columns)) & (df_config['Type'] == 'Flow')]
    if len(tags_not_available) > 0:
        logger.info(f"Following tags not found {tags_not_available['Name'].to_list()}. Will be initialized to NaN")    
        df_resampled[tags_not_available['Name'].to_list()] = np.NAN    
    
    for (key, val) in AREAS.items():
        positive_signals = df_config[df_config['Destination'] == key]
        negative_signals = df_config[df_config['Source'] == key]
        df_resampled[val] = df_resampled[positive_signals['Name']].sum(axis=1) - df_resampled[negative_signals['Name']].sum(axis=1)
    return df_resampled

def translate_columns_names(df_config:pd.DataFrame, column_names:List[str]) -> List[str]:
    translated_names:List[str] = []
    for column_name in column_names:
        translated_name = df_config[df_config['TagName'] == column_name ]['Name'].values
        if translated_name.size < 1:
            logger.error(f"No translation found in df_config for data {column_name}. Skipping translation.")
            return column_names
        translated_names.append(translated_name[0])
    return translated_names

def publish_raw_by_tag(df_raw:pd.DataFrame, df_config:pd.DataFrame):
    """creates and saves an excel file for all tags"""
    for tag in df_raw['dsg_tag'].unique():
        sub_dff:pd.DataFrame = df_raw[df_raw['dsg_tag'] == tag]  
        key_translated = translate_columns_names(df_config, [tag])
        sub_dff.sort_values(by=['PtimeStamp']).to_excel(f'{CLEANED_RESAMPLED_DIR}/{key_translated[0]}.xlsx')        

def convert_volume_to_flow(df_resampled:pd.DataFrame, df_config:pd.DataFrame, resolution:timedelta = RESAMPLING_RESOLUTION):
    """Converts volume counters to flow"""
    for _, signal in df_config[(df_config['Dimension'] == 'm3') &  (df_config['Type'] == 'Flow')].iterrows():
        df_resampled[signal['Name']] = df_resampled[signal['Name']].diff()*(timedelta(hours=1)/resolution)
        df_resampled.loc[df_resampled[signal['Name']] < 0, signal['Name']] = np.nan
        df_resampled[signal['Name']].interpolate(method='linear', inplace=True)
        df_resampled[signal['Name']] = df_resampled[signal['Name']].rolling(window=3, min_periods=1).mean()

    
def get_validated_resampled_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return a tuple of resampled(1) and labelled data(2)"""
    return pd.read_parquet(f'{CLEANED_RESAMPLED_DIR}/resampled.parquet'), pd.read_parquet(f'{CLEANED_RESAMPLED_DIR}/validated.parquet')

def main() -> None:
    df_config = get_config_data()
    df_validation = get_validation_data()
    df_raw = get_raw_data()
    #publish_raw_by_tag(df_raw, df_config) #takes a long time!!!
    df_resampled = perform_resampling_and_pivoting(df_raw)
    # publish_frequency_statistics(df_raw, df_config)
    df_resampled.columns = translate_columns_names(df_config, df_resampled.columns)
    df_validated = validate_dataframe(df_resampled, df_config,df_validation, MIN_PRESSURE,MAX_PRESSURE)
    convert_volume_to_flow(df_resampled, df_config)
    calculate_flow_balance(df_config=df_config,df_resampled=df_resampled)
    #todo graph it    
    if not os.path.exists(CLEANED_RESAMPLED_DIR):
        os.mkdir(CLEANED_RESAMPLED_DIR)
    df_resampled.to_parquet(f'{CLEANED_RESAMPLED_DIR}/resampled.parquet')
    df_validated.to_parquet(f'{CLEANED_RESAMPLED_DIR}/validated.parquet')

def test() -> None:
    df_config = get_config_data()
    df_validation = get_validation_data()
    test = None
    arraySize = 10
    arr = np.array([np.linspace(0,1000,arraySize),np.ones(arraySize)*100])
    print(arr.shape)
    testDf =  pd.DataFrame(arr.transpose(),columns=['TestCol1','TestCol2'])
    testDf.set_index(pd.date_range(start='1/1/2018',name='Timestamp', periods=arraySize), inplace=True)
    print(testDf)
    valDf = validate_dataframe(testDf,df_config,df_validation,100,400)
    print(valDf)


if __name__ == '__main__':
    TEST = False
    if TEST:
        test()
    else:
        main()