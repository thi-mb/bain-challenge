import pandas as pd
import numpy as np 
import locale
import pickle
import logging

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, mutual_info_regression


# set global options for timezone and pandas chained_assignment
locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
pd.options.mode.chained_assignment = None  # default='warn'


# Set log configurations, and create logging decorator function
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s:%(levelname)s:%(name)s:%(message)s', datefmt='%Y.%m.%d %H:%M:%S')
file_handler = logging.FileHandler('logs/train.log')
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

def logger(original_func):
    '''This function sets up the logger actions. It should be used as a decorator.'''
    def wrapper(*args, **kwargs):
        log.info('{} function executed with args: {}, and kwargs: {}'.format(original_func.__name__, args, kwargs))
        return original_func(*args, **kwargs)
    return wrapper


@logger
def load_data():
    '''This function will load the precipitationes, banco_central, 
    and precio_leche datasets used for training the model, then return 
    those datasets.
    
    Returns:
        precipitaciones: Pandas DataFrame containing the precipitaciones data
        used to train the model
        banco_central: Pandas DataFrame containing the banco_central dataset
        used to train the model
        precio_leche: Pandas DataFrame containing the precio_leche dataset
        used to train the model
    '''

    precipitaciones = pd.read_csv('./data/precipitaciones.csv')
    banco_central = pd.read_csv('./data/banco_central.csv')
    precio_leche = pd.read_csv('./data/precio_leche.csv')

    return precipitaciones, banco_central, precio_leche


@logger
def to_100(x):
    '''This function takes in a string of a number and returns
    a float with 2-3 digits
    
    Parameters:
        x (str): A string containing a number separated by periods 
    
    Returns:
        float: A float of the number rounded to 3 digits (in the hundreds),
        with 3 decimal points'''

    x = x.split('.')
    if x[0].startswith('1'): #es 100+
        if len(x[0]) >2:
            return float(x[0] + '.' + x[1])
        x = x[0]+x[1]
        return float(x[:3] + '.' + x[3:])
    else:
        if len(x[0])>2:
            return float(x[0][:2] + '.' + x[0][-1])
        x = x[0] + x[1]
        return float(x[:2] + '.' + x[2:])


@logger
def convert_int(x):
    '''This function transforms a string into an int with any '.' removed.
    
    Parameters:
        x (str): A string of a number
        
    Returns:
        int: An int of the input with any '.' removed'''

    return int(x.replace('.', ''))
    

@logger
def prep_precipitaciones(precipitaciones):
    '''This function prepares the precipitaciones data to train the model and
    it can also be used to prepare data for predictions
    
    Parameters:
        precipitaciones: Pandas DataFrame of the raw precipitaciones dataset
        
    Returns:
        precipitaciones: Pandas DataFrame of the precipitaciones dataset
        after variable processing'''

    precipitaciones['date'] = pd.to_datetime(precipitaciones['date'], format='%Y-%m-%d')
    precipitaciones = precipitaciones.sort_values(by='date', ascending=True).reset_index(drop=True)
    precipitaciones.dropna(how='any', axis=0)
    precipitaciones.drop_duplicates(subset='date')

    return precipitaciones


@logger
def prep_banco_central(banco_central):
    '''This function prepares the banco_central data to train the model, and
    it can also be used to prepare data for predictions
    
    Parameters:
        banco_central: Pandas DataFrame of the raw banco_central dataset
        
    Returns:
        banco_central: Pandas DataFrame of the banco_central dataset
        after variable processing and feature engineering'''

    banco_central['Periodo'] = banco_central['Periodo'].apply(lambda x: x[:10])
    banco_central['Periodo'] = pd.to_datetime(banco_central['Periodo'], format='%Y-%m-%d', errors='coerce')
    banco_central.drop_duplicates(subset='Periodo', inplace=True)
    banco_central = banco_central[~banco_central.Periodo.isna()]
    
    # Preprocessing PIB columns: 
    # 1. create dataframe slice
    cols_pib = [x for x in list(banco_central.columns) if 'PIB' in x]
    cols_pib.extend(['Periodo'])
    banco_central_pib = banco_central[cols_pib]
    banco_central_pib = banco_central_pib.dropna(how = 'any', axis = 0)
  
    # 2. convert to int
    for col in cols_pib:
        if col != 'Periodo':
            banco_central_pib[col] = banco_central_pib[col].apply(lambda x: convert_int(x))

    # Preprocessing Imacec columns: 
    # 1. create dataframe slice
    cols_imacec = [x for x in list(banco_central.columns) if 'Imacec' in x]
    cols_imacec.extend(['Periodo'])
    banco_central_imacec = banco_central[cols_imacec]
    banco_central_imacec = banco_central_imacec.dropna(how = 'any', axis = 0)
    # 2. remove periods and transform to float
    for col in cols_imacec:
        if col != 'Periodo': 
            banco_central_imacec[col] = banco_central_imacec[col].apply(lambda x: to_100(x))

    # Preprocessing IVCM columns: 
    # 1. create slice; 2. drop NaNs; 3. create new feature 'num'
    banco_central_iv = banco_central[['Indice_de_ventas_comercio_real_no_durables_IVCM', 'Periodo']]
    banco_central_iv = banco_central_iv.dropna() # -unidades? #parte 
    banco_central_iv.sort_values(by = 'Periodo', ascending=True)
    banco_central_iv['num'] = banco_central_iv['Indice_de_ventas_comercio_real_no_durables_IVCM'].apply(lambda x: to_100(x))

    # Merge slices together, then return final dataframe
    banco_central_num = pd.merge(banco_central_pib, banco_central_imacec, on = 'Periodo', how = 'inner')
    banco_central_num = pd.merge(banco_central_num, banco_central_iv, on = 'Periodo', how = 'inner')

    return banco_central_num


@logger
def prep_leche(precio_leche):
    '''This function prepares the precio_leche data to train the model.
    This data contains the target variable
        
    Parameters:
        precio_leche: Pandas DataFrame of the raw precio_leche dataset
        
    Returns:
        precio_leche: Pandas DataFrame of the precio_leche dataset
        after variable processing'''

    precio_leche.rename(columns = {'Anio': 'ano', 'Mes': 'mes_pal'}, inplace = True) # precio = nominal, sin iva en clp/litro
    precio_leche['mes'] = pd.to_datetime(precio_leche['mes_pal'], format = '%b')
    precio_leche['mes'] = precio_leche['mes'].apply(lambda x: x.month)
    precio_leche['mes-ano'] = precio_leche.apply(lambda x: f'{x.mes}-{x.ano}', axis = 1)

    return precio_leche


@logger
def merge_data(precipitaciones, banco_central, precio_leche):
    '''This function merges the three datasets that will be used to train the model.
    
    Parameters:
        precipitaciones: Pandas DataFrame of the precipitaciones dataset 
        after being processed (ready for the model)
        banco_central: Pandas DataFrame of the banco_central dataset 
        after being processed (ready for the model)
        precio_leche: Pandas DataFrame of the precio_leche dataset 
        after being processed (ready for the model)
    
    Returns:
        precio_leche_pp_pib: Pandas DataFrame of the inputs merged, this is
        the data that will be used to train the model
        '''

    precipitaciones['mes'] = precipitaciones.date.apply(lambda x: x.month)
    precipitaciones['ano'] = precipitaciones.date.apply(lambda x: x.year)

    precio_leche_pp = pd.merge(precio_leche, precipitaciones, on = ['mes', 'ano'], how = 'inner')
    precio_leche_pp.drop('date', axis = 1, inplace = True)

    banco_central['mes'] = banco_central['Periodo'].apply(lambda x: x.month)
    banco_central['ano'] = banco_central['Periodo'].apply(lambda x: x.year)
    precio_leche_pp_pib = pd.merge(precio_leche_pp, banco_central, on = ['mes', 'ano'], how = 'inner')
    precio_leche_pp_pib.drop(['Periodo', 'Indice_de_ventas_comercio_real_no_durables_IVCM', 'mes-ano', 'mes_pal'], axis =1, inplace = True)

    return precio_leche_pp_pib


@logger
def preprocess(precipitaciones, banco_central, precio_leche):
    '''This function is used to orchestrate the data processing, feature
    engineering and merging.
    
    Parameters:
        precipitaciones: Pandas DataFrame of the precipitaciones dataset 
        before being processed (ready for the model)
        banco_central: Pandas DataFrame of the banco_central dataset 
        before being processed (ready for the model)
        precio_leche: Pandas DataFrame of the precio_leche dataset 
        before being processed (ready for the model)
    
    Returns:
        DataFrame: a DataFrame of the merged data, ready to train the model'''
    
    return merge_data(prep_precipitaciones(precipitaciones), 
                      prep_banco_central(banco_central), 
                      prep_leche(precio_leche))


@logger
def train_model(data):
    '''This function uses the training data (after processing) to train and
    optimize the model. Then, returns the best performing pipeline (based on
    r-squared score).
    
    Parameters:
        data: Pandas DataFrame of the merged datasets (ready for the model)
    
    Returns:
        pipeline (sklearn): an optimized pipeline that will be used to
        predict data
        '''

    np.random.seed(0)
    X = data.drop(['Precio_leche'], axis = 1)
    y = data['Precio_leche']

    pipe = Pipeline([('scale', StandardScaler()),
                    ('selector', SelectKBest(mutual_info_regression)),
                    ('poly', PolynomialFeatures()),
                    ('model', Ridge())])

    params = {'selector__k': [3, 4, 5, 6, 7, 10],
              'poly__degree': [1, 2, 3, 5, 7],
              'model__alpha': [1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]}

    grid = GridSearchCV(estimator = pipe, param_grid = params, cv = 3, scoring = 'r2')
    grid.fit(X, y)

    logging.info('Successfully created model.')

    return grid.best_estimator_


if __name__ == '__main__':
    # 1. Load data
    precipitaciones, banco_central, precio_leche = load_data()
    # 2. Process and merge data
    data = preprocess(precipitaciones, banco_central, precio_leche)
    # 3. Train the model pipeline
    model = train_model(data)
    # 4. Serialize the pipeline as a pickle file
    pickle.dump(model, open('model/leche_predictor.pkl', 'wb'))