import lightgbm as lgb
import xgboost as xgb
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from scnts.TS.ts_class import ts



from sklearn.metrics import mean_squared_error as mse
from scnts.forecast_models.ml.MLBase import MLBaseModel


def PolynomialRegression(degree = 2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))


class LGBM(MLBaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kwargs = kwargs
        self.args = args
        self.picked_model = lgb.LGBMRegressor(*args, **kwargs)



class XGBoost(MLBaseModel): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kwargs = kwargs
        self.args = args
        self.picked_model = xgb.XGBRegressor(*args, **kwargs)


class L_Regression(MLBaseModel):


    def __init__(self, degree = 'best', max_degree = 5, cv = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.defined_model = LinearRegression(**kwargs)
        self.degree = degree

        self.max_degree = max_degree
        self.cv = cv

    # Overriding the fit of the MLBasemodel
    def fit(self, lagged_ts, original_ts): 

        self.lagged_ts = lagged_ts
        self.original_ts = original_ts
        
        # Splitting target/features
        x_train = lagged_ts.loc[:, lagged_ts.columns != 'y'].values
        y_train = lagged_ts['y'].values

        if self.degree != 'best':
            if self.degree > 1:
                poly_reg = PolynomialFeatures(degree = self.degree)
                x_train = poly_reg.fit_transform(x_train)
            # Fitting
            self.model = self.defined_model.fit(x_train, y_train)

        # If degree = 'best' we have to find the best degree for our polynomial
        else:
            params = {'polynomialfeatures__degree': np.arange(self.max_degree)}
            # Using mean squared error as the evaluation. This might be tunable
            search = GridSearchCV(PolynomialRegression(), params, cv = self.cv, scoring = 'neg_mean_squared_error')
            # Fitting
            self.model = search.fit(x_train, y_train)
            # Printing the best results
            for key, value in search.best_params_.items():
                print(f'Best degree for the linear regression model: {value}')
