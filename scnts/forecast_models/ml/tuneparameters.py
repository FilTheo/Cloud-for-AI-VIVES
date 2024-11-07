
#from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from scnts.forecast_models.ml.mlmodels import LGBM, XGBoost
from sklearn.model_selection import GridSearchCV
import warnings

# converts var to a skopt type
def convert_var(var):
  names = [int,float,str,bool]
  vars = [Integer, Real, Categorical, Categorical]
  d = dict(zip(names,vars))
  return d[var]


# converts parameter space to skopt type
def converst_search_space(search_space):
  
  vals = list(search_space.values())
  keys = list(search_space.keys())
  # Get types
  types = [type(val[0]) for val in vals ]

  # Initialize dict
  d = dict()
  for key, val in zip(keys,vals):
    # If we have str or bool variable we just pass them 
    if type(val[0]) == bool or type(val[0]) == str:
        d[key] = Categorical(val)
    else:
        # Converting to either float or int
        t = convert_var(type(val[0]))
        # Sorting the vals in case they are not provided in the right way
        val = sorted(val)
        # Getting the boudnaries
        low = val[0]
        up = val[-1]
        # appending
        d[key] = t(low,up)
  
  return d


def hypertuner(time_series,  model, method = 'bayesian',
               parameter_list = None, max_degree = 10, # If we have an LR model we only give the max_degree, else the params
                               cv = 5, itterations = 32, 
                                evaluation = 'neg_mean_squared_error',
                               return_best = True, print_results = True):
    
    # Assert model in the list 
    if type(model) not in [XGBoost, LGBM]: 
        # removed random forest and linear regression for now
        # rf is useless and linear regression has its own function 
        raise AttributeError('Model not supported. Currently supporting LGBM and XGBoost')
    
    
    # Splitting x_train, y_train
    # y_train is always the 'y' column.
    # If it doesnt exist an error is raised
    try:
        y_train = time_series['y'].values
    except:
        raise AttributeError('No target column is found. Target column should be named y.')
    
    # taking all columns except the target
    x_train = time_series.loc[:, time_series.columns != 'y'].values
    
    # Getting the model
    model_to_fit = model.picked_model

    

    # Two methods:
    # 1st: Grid search
    if method == 'gridsearch':
        # n_jobs set to -1 uses all available cpus
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            search = GridSearchCV(model_to_fit, parameter_list, 
                                    cv = cv, scoring = evaluation,
                                    n_jobs = -1, verbose = False)
    elif method == 'bayesian':
        # Transforming the parameters
        parameter_list = converst_search_space(parameter_list)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            search =  BayesSearchCV(model_to_fit, parameter_list, 
                                    scoring = evaluation,
                                    cv = cv, n_iter = itterations,
                                    n_jobs = -1, verbose = False)

    else:
        raise AttributeError('Optimization method not supported. We currently support only grid and bayesian methods')

    #Fitting
    fitted = search.fit(x_train, y_train, eval_metric = evaluation)

    # Printing
    if print_results == True:
        print(f'Best parameters:')
        print('')
        for key, value in search.best_params_.items():
            print(f'{key}: {value}')
        print('')
        print(f'Highest accuracy: {search.best_score_}')

    # If returns best estimator (in the correct class)
    if return_best:
        # Getting the best parameters
        best_params = search.best_params_
        # Checking what type of model we have
        if type(model) == XGBoost:
            new_model = XGBoost(**best_params)
        elif type(model) == LGBM:
            new_model = LGBM(**best_params)
        # Returns a new model: Not fitted!!!!
        return new_model