import pickle
import pandas as pd
import logging

import train


# This list contains only the variables essential to the model, used to check if there are any 
# missing variables in the data submitted.
cols = ['date', 'Periodo', 'Coquimbo', 'Valparaiso', 'Metropolitana_de_Santiago', 'Libertador_Gral__Bernardo_O_Higgins', 'Maule', 'Biobio', 'La_Araucania', 'Los_Rios', 'PIB_Agropecuario_silvicola', 'PIB_Pesca', 'PIB_Mineria', 'PIB_Mineria_del_cobre', 'PIB_Otras_actividades_mineras', 'PIB_Industria_Manufacturera', 'PIB_Alimentos', 'PIB_Bebidas_y_tabaco', 'PIB_Textil', 'PIB_Maderas_y_muebles', 'PIB_Celulosa', 'PIB_Refinacion_de_petroleo', 'PIB_Quimica', 'PIB_Minerales_no_metalicos_y_metalica_basica', 'PIB_Productos_metalicos', 'PIB_Electricidad', 'PIB_Construccion', 'PIB_Comercio', 'PIB_Restaurantes_y_hoteles', 'PIB_Transporte', 'PIB_Comunicaciones', 'PIB_Servicios_financieros', 'PIB_Servicios_empresariales', 'PIB_Servicios_de_vivienda', 'PIB_Servicios_personales', 'PIB_Administracion_publica', 'PIB_a_costo_de_factores', 'PIB', 'Imacec_empalmado', 'Imacec_produccion_de_bienes', 'Imacec_minero', 'Imacec_industria', 'Imacec_resto_de_bienes', 'Imacec_comercio', 'Imacec_servicios', 'Imacec_a_costo_de_factores', 'Imacec_no_minero']

# The lists below contain all variables pertinent to each dataset, used to separate the precipitaciones and 
# banco_central data, since they are processed separately.
cols_precipitaciones = ['date', 'Coquimbo', 'Valparaiso', 'Metropolitana_de_Santiago', 'Libertador_Gral__Bernardo_O_Higgins', 'Maule', 'Biobio', 'La_Araucania', 'Los_Rios']
cols_banco_central = ['Periodo', 'Imacec_empalmado','Imacec_produccion_de_bienes','Imacec_minero','Imacec_industria','Imacec_resto_de_bienes','Imacec_comercio','Imacec_servicios','Imacec_a_costo_de_factores','Imacec_no_minero','PIB_Agropecuario_silvicola','PIB_Pesca','PIB_Mineria','PIB_Mineria_del_cobre','PIB_Otras_actividades_mineras','PIB_Industria_Manufacturera','PIB_Alimentos','PIB_Bebidas_y_tabaco','PIB_Textil','PIB_Maderas_y_muebles','PIB_Celulosa','PIB_Refinacion_de_petroleo','PIB_Quimica','PIB_Minerales_no_metalicos_y_metalica_basica','PIB_Productos_metalicos','PIB_Electricidad','PIB_Construccion','PIB_Comercio','PIB_Restaurantes_y_hoteles','PIB_Transporte','PIB_Comunicaciones','PIB_Servicios_financieros','PIB_Servicios_empresariales','PIB_Servicios_de_vivienda','PIB_Servicios_personales','PIB_Administracion_publica', 'PIB_a_costo_de_factores', 'Impuesto_al_valor_agregado', 'Derechos_de_Importacion', 'PIB', 'Precio_de_la_gasolina_en_EEUU_dolaresm3', 'Precio_de_la_onza_troy_de_oro_dolaresoz', 'Precio_de_la_onza_troy_de_plata_dolaresoz', 'Precio_del_cobre_refinado_BML_dolareslibra', 'Precio_del_diesel_centavos_de_dolargalon', 'Precio_del_gas_natural_dolaresmillon_de_unidades_termicas_britanicas', 'Precio_del_petroleo_Brent_dolaresbarril', 'Precio_del_kerosene_dolaresm3', 'Precio_del_petroleo_WTI_dolaresbarril', 'Precio_del_propano_centavos_de_dolargalon_DTN', 'Tipo_de_cambio_del_dolar_observado_diario', 'Ocupados', 'Ocupacion_en_Agricultura_INE', 'Ocupacion_en_Explotacion_de_minas_y_canteras_INE', 'Ocupacion_en_Industrias_manufactureras_INE', 'Ocupacion_en_Suministro_de_electricidad_INE', 'Ocupacion_en_Actividades_de_servicios_administrativos_y_de_apoyo_INE', 'Ocupacion_en_Actividades_profesionales_INE', 'Ocupacion_en_Actividades_inmobiliarias_INE', 'Ocupacion_en_Actividades_financieras_y_de_seguros_INE', 'Ocupacion_en_Informacion_y_comunicaciones_INE', 'Ocupacion_en_Transporte_y_almacenamiento_INE', 'Ocupacion_en_Actividades_de_alojamiento_y_de_servicio_de_comidas_INE', 'Ocupacion_en_Construccion_INE', 'Ocupacion_en_Comercio_INE', 'Ocupacion_en_Suministro_de_agua_evacuacion_de_aguas_residuales_INE', 'Ocupacion_en_Administracion_publica_y_defensa_INE', 'Ocupacion_en_Enseanza_INE', 'Ocupacion_en_Actividades_de_atencion_de_la_salud_humana_y_de_asistencia_social_INE', 'Ocupacion_en_Actividades_artisticas_INE', 'Ocupacion_en_Otras_actividades_de_servicios_INE', 'Ocupacion_en_Actividades_de_los_hogares_como_empleadores_INE', 'Ocupacion_en_Actividades_de_organizaciones_y_organos_extraterritoriales_INE', 'No_sabe__No_responde_Miles_de_personas', 'Tipo_de_cambio_nominal_multilateral___TCM', 'Indice_de_tipo_de_cambio_real___TCR_promedio_1986_100','Indice_de_produccion_industrial', 'Indice_de_produccion_industrial__mineria', 'Indice_de_produccion_industrial_electricidad__gas_y_agua', 'Indice_de_produccion_industrial__manufacturera', 'Generacion_de_energia_electrica_CDEC_GWh', 'Indice_de_ventas_comercio_real_IVCM', 'Indice_de_ventas_comercio_real_no_durables_IVCM', 'Indice_de_ventas_comercio_real_durables_IVCM', 'Ventas_autos_nuevos']

# Set log configurations, and create logging decorator function
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s:%(levelname)s:%(name)s:%(message)s', datefmt='%Y.%m.%d %H:%M:%S')
file_handler = logging.FileHandler('logs/predict.log')
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

def logger(original_func):
    '''This function sets up the logger actions. It should be used as a decorator.'''

    def wrapper(*args, **kwargs):
        log.info('{} function executed with args: {}, and kwargs: {}'.format(original_func.__name__, args, kwargs))
        return original_func(*args, **kwargs)
    return wrapper


class LechePredictor:
    '''This class is used to create the precictor object.
    
    Attributes:
        model: sklearn pipeline used to make a prediction

    Methods:
        find_missing_cols: asserts that all columns needed for prediction are present in the dataset
        separate_new_data: splits input data into precipitaciones and banco_central datasets before processing
        prep_new_data: prepares data for prediction by calling functions from train.py
        find_cols_all_na: last check before the data is used for prediction. This method will make sure none of the columns are missing all values
        make_prediction: the method used to orchestrate prediction. It will call methods for preparing data and then output predictions
    '''

    @classmethod
    @logger
    def __init__(cls):
        '''Parameters: None'''
        cls.model = pickle.load(open('model/leche_predictor.pkl', 'rb')) 
    

    @classmethod
    @logger
    def find_missing_cols(cls, data):
        '''This function will parse throught the data submitted and find if 
        any columns essential to the model for prediction are missing. An
        error will be raised if missing columns are found.
        
        Parameters:
            data: a Pandas DataFrame that will be checked for missing columns'''

        missing = [col for col in cols if col not in data.columns]

        for col in cols:
            assert col in data.columns, \
                f'The following columns were missing in the request: \n\n {missing}' 

    
    @classmethod
    @logger
    def separate_new_data(cls, data):
        '''This function will separate a single dataset into the precipitaciones
        and banco_central sets. This is needed because the established processing
        of these datasets happens separately.
        
        Parameters:
            data: a DataFrame that will be split
            
        Returns:
            precipitaciones: a DataFrame with the columns from the precipitaciones
            training dataset
            banco_central: a DataFrame with the columns from the banco_central
            training dataset'''

        precipitaciones = pd.DataFrame(data=data[[col for col in cols_precipitaciones if col in cols_precipitaciones]])
        banco_central = pd.DataFrame(data=data[[col for col in cols_banco_central if col in cols_banco_central]])
        
        return precipitaciones, banco_central


    @classmethod
    @logger
    def prep_new_data(cls, precipitaciones, banco_central):
        '''This function is used to preprocess all of the data used for prediction, 
        performing the same treatment as when data is prepared to train the model.
        
        Parameters:
            precipitaciones: a DataFrame containing the columns from the precipitaciones
            training dataset
            banco_central: a DataFrame containing the columns from the banco_central
            training dataset
        
        Returns:
            data: a DataFrame that is ready to be used by the model (pipeline)
            for prediction. Variables have been processed, features engineered,
            and datasets merged'''
        
        precipitaciones = train.prep_precipitaciones(precipitaciones)
        banco_central = train.prep_banco_central(banco_central)

        # Separates the date/Periodo columns into month (mes) and year (ano)
        precipitaciones['mes'] = precipitaciones['date'].apply(lambda x: x.month)
        precipitaciones['ano'] = precipitaciones['date'].apply(lambda x: x.year)
        banco_central['mes'] = banco_central['Periodo'].apply(lambda x: x.month)
        banco_central['ano'] = banco_central['Periodo'].apply(lambda x: x.year)

        # Merges the precipitaciones and banco_central data
        data = pd.merge(banco_central, precipitaciones, on=['mes', 'ano'], how='inner')

        # This line is used to order the columns of the data used for prediction, it was included after a warning from sklearn
        # saying that the prediction data columns should be in the same order as the training data columns. It is also a way
        # of dropping any columns no essential to the model for prediction.
        data = data[['ano', 'mes', 'Coquimbo', 'Valparaiso', 'Metropolitana_de_Santiago', 'Libertador_Gral__Bernardo_O_Higgins', 'Maule', 'Biobio', 'La_Araucania', 
                    'Los_Rios', 'PIB_Agropecuario_silvicola', 'PIB_Pesca', 'PIB_Mineria', 'PIB_Mineria_del_cobre', 'PIB_Otras_actividades_mineras', 'PIB_Industria_Manufacturera', 
                    'PIB_Alimentos', 'PIB_Bebidas_y_tabaco', 'PIB_Textil', 'PIB_Maderas_y_muebles', 'PIB_Celulosa', 'PIB_Refinacion_de_petroleo', 'PIB_Quimica', 
                    'PIB_Minerales_no_metalicos_y_metalica_basica', 'PIB_Productos_metalicos', 'PIB_Electricidad', 'PIB_Construccion', 'PIB_Comercio', 'PIB_Restaurantes_y_hoteles', 
                    'PIB_Transporte', 'PIB_Comunicaciones', 'PIB_Servicios_financieros', 'PIB_Servicios_empresariales', 'PIB_Servicios_de_vivienda', 'PIB_Servicios_personales', 
                    'PIB_Administracion_publica', 'PIB_a_costo_de_factores', 'PIB', 'Imacec_empalmado', 'Imacec_produccion_de_bienes', 'Imacec_minero', 'Imacec_industria', 
                    'Imacec_resto_de_bienes', 'Imacec_comercio', 'Imacec_servicios', 'Imacec_a_costo_de_factores', 'Imacec_no_minero', 'num']]
        
        return data


    @classmethod
    @logger
    def find_cols_all_na(cls, data):
        '''This function will assert that none of the columns in the processed DataFrame are filled with 
        NaNs. It is a last check before using the data to make a prediction. This is mainly a safeguard 
        because it could happen that after processing, the data for prediction is an empty DataFrame.
        
        Parameters:
            data: a DataFrame that is ready to be used for prediction       
        '''

        assert not True in [data[col].dropna().empty for col in data.columns], \
            f'''There are not enough values to make a prediction! 
            Make sure sure there aren\'t null values for the following variables: {cols}'''


    @classmethod
    @logger
    def make_prediction(cls, precipitaciones, banco_central):
        '''This is the function used for predicting, it orchestrates all data preparation steps calling 
        previous methods to check and prepare the data, then output predictions. Note: datasets have to
        be split (using separate_new_data method) before using this method.

        Parameters:
            precipitaciones: a DataFrame containing the raw weather data to be used for prediction
            banco_central: a DataFrame containing the raw banco_central data to be used for prediction
        
        Returns:
            data: a DataFrame containing the variables after processing, and their associated
            predictions
        '''
        
        data = cls.prep_new_data(precipitaciones, banco_central)
        cls.find_cols_all_na(data)      
        data['prediction'] = cls.model.predict(data)

        return data