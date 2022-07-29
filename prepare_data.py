from datetime import timedelta,datetime
import math
from typing import List
import pandas as pd
import wbgr_datawarehouse.db as dwh
import wbgr_datawarehouse.tables as tables
import os

DATADIR = 'Data'
RAWDIR = os.path.join(DATADIR, 'Raw')
CONFIG_EXCEL = 'Configuratie.xlsx'

def get_config_data() -> pd.DataFrame:
    return pd.read_excel(CONFIG_EXCEL,sheet_name='Vertaling', usecols='C:K',header=0, skiprows=1) 

def get_validation_data() -> pd.DataFrame:
       return pd.read_excel(CONFIG_EXCEL,sheet_name='Validatie',skiprows=1,usecols='C:Z',index_col=0)  

def get_data_from_dwh(TagArray,StartDate:datetime,EndDate = datetime.now(),StepSize = timedelta(days=2)) -> pd.DataFrame:
    """Gets raw data from from dwh in steps of StepSize"""
    db = dwh.db()
    session = db.get_session()    
    steps = math.ceil((EndDate - StartDate)/StepSize)
    df_total_lists:List[pd.DataFrame] = []
    for step in range(steps):
        start_date = StartDate + StepSize*step
        result = session\
            .query(tables.dim_signaal.dsg_id,tables.dim_signaal.dsg_tag,tables.fact_meting.PtimeStamp,tables.fact_meting.fmt_meetwaarde)\
            .join(tables.fact_meting,tables.dim_signaal.dsg_id == tables.fact_meting.fmt_dsg_id )\
            .filter(tables.fact_meting.fmt_ddt_id.between(start_date, start_date+StepSize))\
            .filter(tables.dim_signaal.dsg_tag.in_(TagArray))\
            .to_df()
        df_total_lists.append(result)
    return pd.concat(df_total_lists)

def get_raw_data() -> pd.DataFrame:
    """Gets raw data from local file"""
    return pd.read_parquet(f'{RAWDIR}/data.parquet')



if __name__ == '__main__':
    df_config = get_config_data()
    if not os.path.exists(DATADIR):
        os.mkdir(DATADIR)
    if not os.path.exists(RAWDIR):
        os.mkdir(RAWDIR)
    result = get_data_from_dwh(df_config['TagName'].values, datetime(year=2022,month=4,day=1),StepSize=timedelta(days=5))    
    result.to_parquet(f'{RAWDIR}/data.parquet',index=False)
    print(result)