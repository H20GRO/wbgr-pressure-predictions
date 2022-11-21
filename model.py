from sklearn.preprocessing import MaxAbsScaler,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
 
def create_model_pipeline() -> Pipeline:    
    return Pipeline(steps=[('scaler', StandardScaler()),
                ('linearmodel', linear_model.LinearRegression())],                
                )
