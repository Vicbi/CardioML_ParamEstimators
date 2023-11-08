import pandas as pd
import numpy as np
import pickle
from scipy import stats
import os
from sklearn import metrics
from prettytable import PrettyTable
from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV  

def remove_outliers(dataset, confidence_level=3.5):
    """
    Remove outliers from the dataset based on the specified confidence level.

    Parameters:
        dataset (pandas.DataFrame): The input dataset.
        confidence_level (float): The number of standard deviations for confidence (default is 3.5).

    Returns:
        pandas.DataFrame: Dataset with outliers removed.
    """
    # Calculate mean and standard deviation
    mean = dataset.mean(axis=0)
    std_dev = dataset.std(axis=0)
    
    # Identify outliers and remove them
    mask_low = dataset > mean - confidence_level * std_dev
    mask_high = dataset < mean + confidence_level * std_dev
    mask = mask_low & mask_high
    row_mask = mask.all(axis=1)
    
    dataset = dataset[row_mask]
    return dataset

def select_variable_to_predict(dataset, prediction_variable):
    """
    Generate X, y pairs for a specific prediction variable.

    Parameters:
        dataset (pandas.DataFrame): Input DataFrame.
        prediction_variable (str): Variable to be predicted (e.g., 'CO', 'aSBP', 'Ees').

    Returns:
        X (numpy.ndarray): Input features.
        y (numpy.ndarray): Prediction variable.
    """
    # Dictionary to map prediction variables to corresponding rear and prediction columns
    variable_mapping = {
        'CO': ('EF', 'CO'),
        'aSBP': ('EF', 'aSBP'),
        'Ees': ('aSBP', 'Ees')
    }

    if prediction_variable not in variable_mapping:
        raise ValueError(f"Invalid prediction_variable: {prediction_variable}. Use 'CO', 'aSBP', or 'Ees'.")

    rear, prediction = variable_mapping[prediction_variable]
    
    # Get column indices
    start_idx = dataset.columns.get_loc('brSBP')
    rear_idx = dataset.columns.get_loc(rear)
    prediction_idx = dataset.columns.get_loc(prediction)

    # Generate X, y pairs
    X = dataset.iloc[:, start_idx:rear_idx + 1].values  # +1 to include 'rear' column
    y = dataset.iloc[:, prediction_idx].values

    return X, y


def save_input_output_pairs(X, y):
    """
    Save input features (X) and prediction variable (y) using pickle.

    Parameters:
        X (object): Input features.
        y (object): Prediction variable.

    Returns:
        None
    """
    with open('X.pkl', 'wb') as f:
        pickle.dump(X, f)
    with open('y.pkl', 'wb') as f:
        pickle.dump(y, f)

        
def load_input_output_pairs():
    """
    Load input features (X) and prediction variable (y) from pickle files.

    Returns:
        X (object): Input features.
        y (object): Prediction variable.
    """
    with open('X.pkl', 'rb') as f:
        X = pickle.load(f)
    with open('y.pkl', 'rb') as f:
        y = pickle.load(f)
    
    return X, y
    
    
def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data, 
                       model, param_grid, cv = 10, scoring_fit = 'neg_mean_squared_error',
                       do_probabilities = False):

    gs = GridSearchCV(estimator = model,
                      param_grid = param_grid, 
                      cv = cv, 
                      scoring = scoring_fit,
                      verbose = 2
                     )
    fitted_model = gs.fit(X_train_data, y_train_data)
    
    if do_probabilities:
        pred = fitted_model.predict_proba(X_test_data)
    else:
        pred = fitted_model.predict(X_test_data)
    
    return fitted_model, pred


def hyperparameter_tuning(X_train, X_test, y_train, y_test,regressor):
    
    if regressor == 'RF':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor()

        param_grid = {
            "bootstrap": [True],
            "criterion": ['mse'],
            "max_depth": [5,10,20],
            "n_estimators": [500,700,1000],
            # "min_samples_leaf": [2],
            # "max_features": ['auto'],
            # "n_jobs": [10] 
        }

    if regressor == 'SVR':
        from sklearn.svm import SVR
        model = SVR()

        param_grid = {
            'kernel': ['rbf'],
            'C': [ 1, 10, 100], 
            'gamma': [1, 0.1, 0.01, 0.001], 
            # 'epsilon': [0.1],
            # 'coef0': [1]
        }
        
    if regressor == 'RIDGE':
        from sklearn.linear_model import Ridge
        model = Ridge()

        param_grid = {
            'alpha': [1,10,100,200]
        }
    
    if regressor == 'GB':
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(
                                          # max_depth = 4, 
                                          # min_samples_split = 2,
                                          # min_samples_leaf = 1, 
                                          # subsample = 1,
                                          # max_features = 'sqrt', 
                                          # random_state = 10
        )
        param_grid = {
            'learning_rate':[0.1,0.05,0.01],
            'n_estimators':[100,500,1000,1750]
        }

    
    model, pred = algorithm_pipeline(X_train, X_test, y_train, y_test, model, 
                                     param_grid, cv=10)

    # Root Mean Squared Error
    print(np.sqrt(-model.best_score_))
    print(model.best_params_)

    best_parameters = model.best_params_

    return model, pred, best_parameters

            
def create_folder(path):
    """
    Create a folder at the specified path.

    Parameters:
        path (str): The path to create the folder.

    Returns:
        None
    """
    if not os.path.isabs(path):
        raise ValueError("Please provide an absolute path.")

    if os.path.exists(path):
        print(f"The directory '{path}' already exists.")
        return

    try:
        os.mkdir(path)
        print(f"Successfully created the directory '{path}'.")
    except OSError as e:
        print(f"Creation of the directory '{path}' failed. Error: {e}")
        

def calculate_metrics_for_each_fold(current_fold,y_test,y_pred,metrics_array,output_verbose):
    """
    Calculate and print various regression metrics for a fold.

    Parameters:
        current_fold (int): The current fold number.
        y_test (array-like): True values.
        y_pred (array-like): Predicted values.
        metrics_array (numpy.ndarray): Array to store metrics.
        output_verbose (bool, optional): Whether to print results (default is True). 

    Returns:
        None
    """
    assert len(y_test) == len(y_pred), "y_test and y_pred must have the same length."

    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    nrmse = 100 * rmse / (np.max(y_test) - np.min(y_test))

    if output_verbose:
        print(f'Fold no. ({current_fold})')
        print(f'Mean Absolute Error: {mae:.2f} [unit]')
        print(f'Mean Squared Error: {mse:.2f} [unit]')
        print(f'Root Mean Squared Error: {rmse:.2f} [unit]')
        print(f'Normalized Root Mean Squared Error: {nrmse:.2f}%\n')
    
    metrics_array[current_fold, 0] = mae
    metrics_array[current_fold, 1] = mse
    metrics_array[current_fold, 2] = rmse
    metrics_array[current_fold, 3] = nrmse
    
    return metrics_array


def accumulate_predictions(y_test, y_pred, current_fold, y_test_total, y_pred_total):
    """
    Accumulate y_test and y_pred into y_test_total and y_pred_total across multiple folds.

    Parameters:
        y_test (array-like): True values for the current fold.
        y_pred (array-like): Predicted values for the current fold.
        current_fold (int): The current fold number.
        y_test_total (array-like, optional): Accumulated true values from previous folds.
        y_pred_total (array-like, optional): Accumulated predicted values from previous folds.

    Returns:
        y_test_total (array-like): Accumulated true values.
        y_pred_total (array-like): Accumulated predicted values.
    """
    if y_test_total is None or y_pred_total is None:
        y_test_total = y_test
        y_pred_total = y_pred
    else:
        y_test_total = np.append(y_test_total, y_test, axis=0)
        y_pred_total = np.append(y_pred_total, y_pred, axis=0)
    
    return y_test_total, y_pred_total

    
def calculate_regression_statistics(y_test_total, y_pred_total, metrics_array, regressor):
    """
    Calculate and print various regression statistics.

    Parameters:
        y_test_total (array-like): True values.
        y_pred_total (array-like): Predicted values.
        metrics_array (array-like): Array including MAE, MSE, RMSE, NRMSE for each fold.
        regressor (str): Name of the regression model.

    Returns:
        None
    """
    assert len(y_test_total) == len(y_pred_total), "Input lists must have the same length."
    
    average_MAE = np.round(np.mean(metrics_array[:,0]), 2)
    average_MSE = np.round(np.mean(metrics_array[:,1]), 2)
    average_RMSE = np.round(np.mean(metrics_array[:,2]), 2)
    average_NRMSE = np.round(np.mean(metrics_array[:,3]), 2)
    
    std_MAE = np.round(np.std(metrics_array[:,0]), 2)
    std_MSE = np.round(np.std(metrics_array[:,1]), 2)
    std_RMSE = np.round(np.std(metrics_array[:,2]), 2)
    std_NRMSE = np.round(np.std(metrics_array[:,3]), 2)

    slope, intercept, r_value, p_value, std_err = stats.linregress(y_pred_total, y_test_total)

    mean_diff = np.sum(y_pred_total - y_test_total) / np.size(y_pred_total)
    std_diff = np.std(y_pred_total - y_test_total)

    rmse_m_sd = f'{average_RMSE}±{std_RMSE}'
    nrmse_m_sd = f'{average_NRMSE}±{std_NRMSE}'

    x = PrettyTable()
    x.field_names = ['Model', 'Slope', 'Intercept', 'Correlation Coef', 'p-value', 'RMSE', 'nRMSE', 'Mean difference', 'SD difference']
    x.add_row([regressor, np.round(slope, 2), np.round(intercept, 2), np.round(r_value, 2), p_value, rmse_m_sd, nrmse_m_sd, np.round(mean_diff, 2), np.round(std_diff, 2)])
    print(x)

    print(f'p-value: {p_value}')
    print(f'Average MAE: {average_MAE}')
    print(f'SD MAE: {std_MAE}')
    
        
def calculate_precision_percentages(y_pred_total, y_test_total, prediction_variable):
    """
    Calculate and print precision percentages.

    Parameters:
        y_pred_total (array-like): Predicted values.
        y_test_total (array-like): True values.
        prediction_variable (str): Variable to be predicted ('aSBP', 'CO', 'Ees').

    Returns:
        None
    """
    assert len(y_pred_total) == len(y_test_total), "y_pred_total and y_test_total must have the same length."

    temp = y_pred_total - y_test_total

    thresholds = {
        'aSBP': [1.5, 2.5, 3.5, 5],
        'CO': [0.5, 0.3, 1],
        'Ees': [0.1, 0.2, 0.5]
    }

    if prediction_variable not in thresholds:
        raise ValueError(f"Invalid prediction_variable: {prediction_variable}. Use 'aSBP', 'CO', or 'Ees'.")

    print(f'Precision percentages for {prediction_variable}:')

    for threshold in thresholds[prediction_variable]:
        percentage = 100 * np.round(len(temp[np.abs(temp) < threshold]) / len(temp), 2)
        print(f'Cases with ({prediction_variable}_model - {prediction_variable}_real) < {threshold}: {percentage}%')
    


def run_regressor_with_best_hyperparams(X_train, X_test, y_train, y_test, regressor, prediction_variable):
    """
    Select and train a regression model based on the given regressor type, prediction variable and hyperparameters selected after hyperparameter tuning.

    Parameters:
        X_train (array-like): Training features.
        X_test (array-like): Testing features.
        y_train (array-like): Training target variable.
        y_test (array-like): Testing target variable.
        regressor (str): Type of regressor ('RF', 'SVR', 'RIDGE', 'GB').
        prediction_variable (str): The variable to predict ('aSBP', 'CO', 'Ees').

    Returns:
        model: Trained regression model.
        y_pred: Predicted values.
    """
    # Define best hyperparameters based on prediction_variable
    hyperparameters = {
        'aSBP': {'RF': {'max_depth': 10, 'n_estimators': 700},
                 'SVR': {'C': 100, 'gamma': 0.001},
                 'RIDGE': {'alpha': 10},
                 'GB': {'learning_rate': 0.01, 'n_estimators': 1750}},
        'CO': {'RF': {'max_depth': 10, 'n_estimators': 700},
               'SVR': {'C': 10, 'gamma': 0.001},
               'RIDGE': {'alpha': 100},
               'GB': {'learning_rate': 0.05, 'n_estimators': 500}},
        'Ees': {'RF': {'max_depth': 10, 'n_estimators': 700},
                'SVR': {'C': 10, 'gamma': 0.001},
                'RIDGE': {'alpha': 200},
                'GB': {'learning_rate': 0.05, 'n_estimators': 500}}
    }

    # Input validation
    assert regressor in hyperparameters[prediction_variable], f"Invalid regressor: {regressor}"
    assert prediction_variable in hyperparameters, f"Invalid prediction_variable: {prediction_variable}"

    # Get hyperparameters for the selected regressor and prediction_variable
    selected_hyperparameters = hyperparameters[prediction_variable][regressor]

    if regressor == 'RF':
        model = RandomForestRegressor(**selected_hyperparameters)
    elif regressor == 'SVR':
        model = SVR(**selected_hyperparameters)
    elif regressor == 'RIDGE':
        model = Ridge(**selected_hyperparameters)
    elif regressor == 'GB':
        model = GradientBoostingRegressor(**selected_hyperparameters)

    y_pred = model.fit(X_train, y_train).predict(X_test)
    return model, y_pred