from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from sklearn import linear_model, preprocessing,model_selection
from sqlalchemy import false
from validate_data import get_validated_resampled_data, AREAS
from prepare_data import get_config_data
import plotly.express as px
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
from sklearn.metrics import mean_squared_error

@dataclass
class Station:
    Name:str
    PressureSignal:str
    FlowSignals:List[str] = None
    MinFlow:float = None

@dataclass
class Point:
    Name:str
    PressureSignal:str
    FlowSignal:str    
    Predictions: Optional[dict[str, pd.DataFrame]] = field(default_factory=dict)
    MinFlow:Optional[float] = None
    @property
    def has_local_flow(self) -> bool:
        return bool(self.FlowSignal)

@dataclass
class Area:
    Name:str
    FlowSignal:str
    Stations:List[Station]
    Points:List[Point]
    MinFlow:float = None


def time_filter_df(df:pd.DataFrame, startdate:datetime,end_date:datetime):
    return df.loc[(df.index > startdate) & (df.index <= end_date)]

def time_filter_dfs(startdate:datetime,end_date:datetime,*dfs):
    if len(dfs) == 1:
        return time_filter_df(dfs[0],startdate,end_date)
    return [time_filter_df(df,startdate, end_date) for df in dfs]


dma_flow = AREAS['Provincie']
point1 = Point(Name='Usquert_In',PressureSignal='P_Usquert_In', FlowSignal='Usquert_In', MinFlow=10)
station1 = Station(Name="Onnen", PressureSignal="P_Onnen_Uit", FlowSignals=["Onnen_Uit"], MinFlow=10)
station2 = Station(Name="Nietap", PressureSignal="P_Nietap_Uit", FlowSignals=["Nietap_Lettelbert","Nietap_Provincie"], MinFlow=10)
#station2 = Station(Name="Nietap", PressureSignal="P_Nietap_Uit", FlowSignal="Nietap_Uit", MinFlow=10) mist totale nietap distributie
area = Area(Name='Provincie',FlowSignal=AREAS['Provincie'],Stations=[station1,station2], Points=[point1])


signals_data, _ = get_validated_resampled_data()
resolution = pd.to_timedelta(signals_data.index.inferred_freq)

#predictions:dict[str, pd.DataFrame] = {}

test_days = 30
start_date_test = datetime(year=2022,month=6, day=1)
end_date = start_date_test + timedelta(days=test_days)
signals_data = time_filter_dfs(start_date_test, end_date, signals_data)
for station in area.Stations: 
    station_pressure = signals_data[station.PressureSignal]
    for point in area.Points:
        y = (station_pressure - signals_data[point.PressureSignal])
        x1 = signals_data[area.FlowSignal]**2
        x2 = (signals_data[station.FlowSignals]).sum(axis=1)**2
        x3 = signals_data[point.FlowSignal]**2        
        test_features = np.array([x1,x2,x3]).transpose()        
        scaler =  preprocessing.MaxAbsScaler()
        X =  scaler.fit_transform(test_features)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.15, shuffle=false) # use the first 15% of the values
        model = linear_model.LinearRegression()
        model.fit(X_train, y_train)                
        prediction = model.predict(X)        
        #mse = mean_squared_error(df['y'], df['y_predicted'])
        prefix = f"{point.Name}-{station.Name}"
        df_prediction = pd.DataFrame()
        df_prediction['station_pressure'] = station_pressure
        df_prediction.index = station_pressure.index
        df_prediction['delta_p_prediction'] = prediction
        df_prediction['delta_p'] = y
        df_prediction['point_p_prediction'] = station_pressure - prediction
        df_prediction['point_p'] = signals_data[point.PressureSignal]
        point.Predictions.update({prefix: df_prediction})
        

       
fig = make_subplots(rows=1, cols=1)

for point in area.Points:
    for prefix, prediction in point.Predictions.items():
        fig.add_trace(go.Scatter(x=prediction.index,y=prediction['point_p'], name=f'{prefix}-p',showlegend=True))
        fig.add_trace(go.Scatter(x=prediction.index,y=prediction['point_p_prediction'], name=f'{prefix}-predicted_p',showlegend=True,line={'dash':'dash'}))

        # fig.add_trace(go.Scatter(x=prediction.index,y=prediction['point_p'], name=f'{prefix}-p',showlegend=True,marker={'color':'#FF9999'}))
        # fig.add_trace(go.Scatter(x=prediction.index,y=prediction['point_p_prediction'], name=f'{prefix}-predicted_p',showlegend=True,marker={'color':'#FF0000'},line={'dash':'dash'}))

title = "Pressure measurement and prediction"
fig.update_layout(
    title = title,
    xaxis_title="Time",
    yaxis_title="Pressure (kPa)")
fig.update_yaxes(range=[-50, 500])
fig.write_html(f'{title.replace('','_')}.html', auto_open=True)
fig.show()


























