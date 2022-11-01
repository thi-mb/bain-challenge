# Bain Challenge: Machine Learning Engineer
## Predicting Milk Price in Chile 

For this challenge, a [Jupyter notebook with the development of a machine-learning model was provided](https://github.com/SpikeLab-CL/ml-engineer-challenge). This model uses multiple economical and weather variables to predict the price of milk in Chile. 

The goal is to create an simple Python API that exposes an endpoint to obtain predictions via  HTTP requests. This was achieved by extracting the variable preprocessing and feature engineering code from the notebook and converting it into a series of functions. These functions can then be used to process prediction data. The machine-learning pipeline is trained and serialized as a pickle file whenever the API is started, which means it can dynamically adjust if the client chooses to expand the training data (beware, this may affect the model performance negatively).

The API was created using the Flask micro-framework, and predictions can be obtained via GET or POST HTML requests. GET requests are limited to a single prediction at a time. POST requests accept JSON as input, and both GET and POST respond with another JSON including the variables provided and their associated prediction. Details for the input requirements is outlined later in this guide.

Each python file has an associated log, which describes when and how each of their functions are used. Besides the log, there is some error handling, but more work is needed in that regard. Additionally, a basic home page is included which has information on how to use the API. Documentation of the code was a high priority for this project. It was assumed that preserving the original code for the model was a priority as well, since that was established as a good performer.

***

## Installation with Docker
 1. **Prerequisites**

The only prerequisite is Docker, you can download it for free on [their website](https://www.docker.com).

 2.  **Building the image**

To build the image, clone or download this repo and unzip (if necessary), then open terminal and navigate to the 'bain-challenge' folder using `cd`.

Build the image with the following command (may take up to 2 minutes):

`docker build --tag bain-challenge .`

3. **Running the image as container**

Now that the image is built you can run it as a container, which will create all the dependencies and start the API. To do that, run the following command on terminal:

`docker run -d -p 8000:5000 --name bain-challenge-test bain-challenge`

Remember that you can monitor the application using the following command:

`docker logs -f bain-challenge-test`

## Obtaining predictions
Once the API is running, the address should be [http://localhost:8000/](http://localhost:8000/)
* **Home page** (`http://localhost:8000/`)

This contains some basic information on how to use the API. I am not a front-end developer but felt it should be included. It can be accessed via the link above.

* **Health check** (`http://localhost:8000/health/`)

This is a simple endpoint to check if the service is operational. If the response code is 200, that means the system is healthy. It can also be used via a browser and if the system is healthy it will return a message saying 'Sytem is operational!'.

* **GET request** (`http://localhost:8000/get_predict/`)

To obtain a prediction using GET requests, all the variables must be included in the url using the following format:

http://localhost:8000/get_request/?var1=value1&var2=value2&var3=value3 ... so on

This kind if request is limited to a single prediction but can be easily used by the client's developer team. Two caveats are that the url gets really long, and that GET requests are less secure than POST requests. Once the API is running on your computer, [check this link for an example of its usage](http://localhost:8000/get_predict/?date=2016-09-01&Coquimbo=0.0&Valparaiso=0.0358603897069085&Metropolitana_de_Santiago=1.06697986523102&Libertador_Gral__Bernardo_O_Higgins=3.09096124396306&Maule=18.7369966790725&Biobio=45.6468018902109&La_Araucania=74.4993070194218&Los_Rios=119.11441859164&Periodo=2016-09-01&Imacec_empalmado=101.421.423&Imacec_produccion_de_bienes=925.256.728&Imacec_minero=970.200.728&Imacec_industria=94.362.066&Imacec_resto_de_bienes=877.740.435&Imacec_comercio=965.660.644&Imacec_servicios=107.875.178&Imacec_a_costo_de_factores=100.964.252&Imacec_no_minero=10.191.375&PIB_Agropecuario_silvicola=178.797.615&PIB_Pesca=342.580.723&PIB_Mineria=122.437.133&PIB_Mineria_del_cobre=10.851.499&PIB_Otras_actividades_mineras=140.700.266&PIB_Industria_Manufacturera=120.513.577&PIB_Alimentos=296.788.403&PIB_Bebidas_y_tabaco=152.565.286&PIB_Textil=292.274.656&PIB_Maderas_y_muebles=69.059.951&PIB_Celulosa=98.786.742&PIB_Refinacion_de_petroleo=950.226.102&PIB_Quimica=189.741.256&PIB_Minerales_no_metalicos_y_metalica_basica=724.921.833&PIB_Productos_metalicos=205.288.522&PIB_Electricidad=311.474.945&PIB_Construccion=734.734.113&PIB_Comercio=100.716.927&PIB_Restaurantes_y_hoteles=23.782.948&PIB_Transporte=604.998.911&PIB_Comunicaciones=39.756.745&PIB_Servicios_financieros=629.657.787&PIB_Servicios_empresariales=118.878.478&PIB_Servicios_de_vivienda=88.551.452&PIB_Servicios_personales=141.309.491&PIB_Administracion_publica=575.846.354&PIB_a_costo_de_factores=106.169.997&Impuesto_al_valor_agregado=967.771.147&Derechos_de_Importacion=662.808.578&PIB=116.530.017&Precio_de_la_gasolina_en_EEUU_dolaresm3=36.766.072&Precio_de_la_onza_troy_de_oro_dolaresoz=1326.51&Precio_de_la_onza_troy_de_plata_dolaresoz=193.171&Precio_del_cobre_refinado_BML_dolareslibra=213.516.284&Precio_del_diesel_centavos_de_dolargalon=139.84&Precio_del_gas_natural_dolaresmillon_de_unidades_termicas_britanicas=2.9689&Precio_del_petroleo_Brent_dolaresbarril=46.19&Precio_del_kerosene_dolaresm3=3.331.562&Precio_del_petroleo_WTI_dolaresbarril=45.2&Precio_del_propano_centavos_de_dolargalon_DTN=49.804&Tipo_de_cambio_del_dolar_observado_diario=668.632.381&Ocupados=837.838.138&Ocupacion_en_Agricultura_INE=632.257.018&Ocupacion_en_Explotacion_de_minas_y_canteras_INE=200.039.869&Ocupacion_en_Industrias_manufactureras_INE=921.206.282&Ocupacion_en_Suministro_de_electricidad_INE=425.622.028&Ocupacion_en_Actividades_de_servicios_administrativos_y_de_apoyo_INE=220.891.154&Ocupacion_en_Actividades_profesionales_INE=294.301.769&Ocupacion_en_Actividades_inmobiliarias_INE=792.124.762&Ocupacion_en_Actividades_financieras_y_de_seguros_INE=160.525.842&Ocupacion_en_Informacion_y_comunicaciones_INE=151.141.074&Ocupacion_en_Transporte_y_almacenamiento_INE=549.378.623&Ocupacion_en_Actividades_de_alojamiento_y_de_servicio_de_comidas_INE=377.935.684&Ocupacion_en_Construccion_INE=750.005.699&Ocupacion_en_Comercio_INE=162.151.515&Ocupacion_en_Suministro_de_agua_evacuacion_de_aguas_residuales_INE=483.566.473&Ocupacion_en_Administracion_publica_y_defensa_INE=441.176.553&Ocupacion_en_Enseanza_INE=731.519.771&Ocupacion_en_Actividades_de_atencion_de_la_salud_humana_y_de_asistencia_social_INE=455.114.233&Ocupacion_en_Actividades_artisticas_INE=107.299.657&Ocupacion_en_Otras_actividades_de_servicios_INE=229.643.139&Ocupacion_en_Actividades_de_los_hogares_como_empleadores_INE=36.385.222&Ocupacion_en_Actividades_de_organizaciones_y_organos_extraterritoriales_INE=44.630.936&No_sabe__No_responde_Miles_de_personas=nan&Tipo_de_cambio_nominal_multilateral___TCM=11.030.619&Indice_de_tipo_de_cambio_real___TCR_promedio_1986_100=942.942.416&Indice_de_produccion_industrial=978.319.404&Indice_de_produccion_industrial__mineria=969.847.105&Indice_de_produccion_industrial_electricidad__gas_y_agua=985.785.658&Indice_de_produccion_industrial__manufacturera=98.496.455&Generacion_de_energia_electrica_CDEC_GWh=584.741.823&Indice_de_ventas_comercio_real_IVCM=102.585.502&Indice_de_ventas_comercio_real_no_durables_IVCM=100.746.059&Indice_de_ventas_comercio_real_durables_IVCM=110.349.817&Ventas_autos_nuevos=32377.0&)

If the variables submitted have a formatting error, or if any variables are missing, a JSON with an error message will be returned with details on what problem the pipeline has run into.

* **POST request** (`http://localhost:8000/post_predict/`)

This is the main endpoint for obtaining predictions. To obtain a prediction, submit a JSON using your preferred requests manager. In this guide, python requests (plus json and pandas) will be used.

***POST request requires the JSON input to contain all of the data***, so both of the weather (precipitaciones) and economical (banco_central) datasets must be included in the JSON file. The datasets do not have to be merged in any specific way (e.g. matching on dates), they just have to be on the same file. The datasets will be split, processed, then re-merged on dates before obtaining the prediction. Ideally, a conversation with the client would happen so that the input specifications could be outlined to avoid this assumption, but the current way should still provide the client with some flexibility.

Here is an example of how to obtain predictions using python requests, pandas, and json. First the two datasets are merged, then transformed to JSON and submitted. The response contains a JSON with the variables and their associated prediction, which gets converted to a DataFrame and saved as csv.

```python
import pandas as pd
import json
import requests

precipitaciones = pd.read_csv('data/precipitaciones.csv')
banco_central = pd.read_csv('data/banco_central.csv')
pred_data = pd.concat([precipitaciones, banco_central], axis=1)

pred_json = json.loads(pred_data.to_json())

URL = 'http://localhost:8000/post_predict/'
response = requests.post(url=URL, json=pred_json)

response_df = pd.DataFrame(response.json())
response_df.to_csv('predictions.csv')
```

If the variables submitted have a formatting error, or if any variables are missing, a JSON with an error message will be returned with details on what problem the pipeline has run into.
