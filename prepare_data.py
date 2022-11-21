from datetime import timedelta,datetime
import math
from typing import Dict, List
import pandas as pd
import wbgr_datawarehouse.db as dwh
import wbgr_datawarehouse.tables as tables
import os
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from constants import *
from logger import get_logger

log = get_logger(__name__)

def get_config_data() -> pd.DataFrame:
    """Get confgiuration data from an excel file"""
    return pd.read_excel(CONFIG_EXCEL,sheet_name='Vertaling', usecols='C:K',header=0, skiprows=1) 

def get_validation_data() -> pd.DataFrame:
    """Get validation data from an excel file"""
    return pd.read_excel(CONFIG_EXCEL,sheet_name='Validatie',skiprows=1,usecols='C:Z',index_col=0)  

def get_data_from_dwh(TagDict:Dict[str, str],StartDate:datetime,EndDate = datetime.now(),StepSize = timedelta(days=2)) -> pd.DataFrame:
    """Gets raw data from from dwh in steps of StepSize"""
    db = dwh.db()
    session = db.get_session()    
    steps = math.ceil((EndDate - StartDate)/StepSize)
    df_total_lists:List[pd.DataFrame] = []
    for step in range(steps):
        start_date = StartDate + StepSize*step
        result = (session
            .query(tables.dim_signaal.dsg_id,tables.dim_signaal.dsg_name,tables.fact_meting.PtimeStamp,tables.fact_meting.fmt_meetwaarde)
            .join(tables.fact_meting,tables.dim_signaal.dsg_id == tables.fact_meting.fmt_dsg_id )
            .filter(tables.fact_meting.fmt_ddt_id.between(start_date, start_date+StepSize))
            .filter(tables.dim_signaal.dsg_name.in_(TagDict.keys()))
            .to_df()
            .pivot_table(index='PtimeStamp',values='fmt_meetwaarde',columns='dsg_name', aggfunc='mean')
            .rename(columns=TagDict))
        df_total_lists.append(result)
    return pd.concat(df_total_lists)

def get_raw_data() -> pd.DataFrame:
    """Gets raw data from local file"""
    return pd.read_parquet(f'{RAW_DIR}/data.parquet')

def create_data_availability_chart(_df: pd.DataFrame, _df_config:pd.DataFrame) -> None:       
    interval = timedelta(hours=1)
    # organize data in a pandas dataframe
    df_available_all = _df.resample(rule=interval).count().reset_index()    
    dfm= pd.melt(df_available_all, id_vars=['PtimeStamp'], value_vars=df_available_all.columns)
    print(dfm.columns)
    print(dfm.head())
    dfm.loc[dfm['value'] > 1 ,"value"] = 1
    dfm['value'] = dfm['value'].astype('category')
    dfm['end'] = dfm['PtimeStamp'] + interval
    fig = px.timeline(dfm, x_start="PtimeStamp", x_end="end",color='value', y="dsg_name")
    fig.update_layout({'title':'<b>Hourly data availability</b>'})
    fig.write_html(f'{RESULTS_DIR}/Validation_gantchart.html')       


def create_box_plot(_df:pd.DataFrame, _df_config:pd.DataFrame) -> None:    
    ax = (_df
    .resample(rule=timedelta(hours=1)).count()
    .reset_index()
    .assign(**{col : 0 for col in np.setdiff1d(_df_config['Name'].values, list(_df.columns))}) # add tags that might not exists in the db
    .assign(DummyTag=0) # add dummy tag for visualization                 
    .boxplot(figsize=(20,10),vert=False)
    )
    ax.axvline(x=60, color='red', linestyle='--')
    ax.text(64,18.6,'gewenst, 1 per minuut', rotation=0)
    ax.set_xlabel('Data punten per uur, meer is beter')
    ax.set_title('Data dichtheid')
    ax.get_figure().tight_layout()
    ax.get_figure().savefig(f'{RESULTS_DIR}/data_dichtheid_plot.png')

def _interpolation_selector(_col:pd.Series, _df_config:pd.DataFrame) -> pd.Series:
    """Interpolation of data based on column name """
    methods = _df_config.query(f'Name == "{_col.name}"')['InterpolationMethod'].values
    method = None
    if len(methods) > 0:
        method = methods[0]  
    else:
        method = 'linear'
    if method != method:
        method = 'linear'        
    return _col.interpolate(method=method)

def resampling(_df:pd.DataFrame, _df_config:pd.DataFrame, frequency=timedelta(minutes=15)) -> pd.DataFrame:
    """Resample and interpolate the dataset"""
    return (_df
    .sort_index()
    .apply(_interpolation_selector, _df_config=_df_config)
    .resample(rule=frequency)
    .mean()
    )    

def get_resampled_data() -> pd.DataFrame:
    return pd.read_parquet(f'{RESAMPLED_DIR}/data.parquet')

def run() -> None:
    """ Run data acquisition from DWH"""
    df_config = get_config_data()
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(RAW_DIR):
        os.mkdir(RAW_DIR)
    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)
    if not os.path.exists(RESAMPLED_DIR):
        os.mkdir(RESAMPLED_DIR)
    
    df_raw = pd.DataFrame
    if True:
        df_raw = get_data_from_dwh(df_config.set_index('TagNameShort').to_dict()['Name'], datetime(year=2022,month=7,day=1),StepSize=timedelta(days=60))    
        df_raw.to_parquet(f'{RAW_DIR}/data.parquet')
    else:
        df_raw = get_raw_data()
    create_data_availability_chart(df_raw, df_config)
    create_box_plot(df_raw, df_config)
    df_resampled = resampling(df_raw, df_config)
    df_resampled.to_parquet(f'{RESAMPLED_DIR}/data.parquet')
    
    pass
    
if __name__ == '__main__':
   run()