import logging
from flask import Flask, jsonify, request, render_template
import pandas as pd

from predict import LechePredictor


# create Flask app instance
app = Flask(__name__)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True

# create and train model. This happens once whenever the app is started
predictor = LechePredictor()

# Set log configurations, and create logging decorator function
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s:%(levelname)s:%(name)s:%(message)s', datefmt='%Y.%m.%d %H:%M:%S')
file_handler = logging.FileHandler('logs/app.log')
file_handler.setFormatter(formatter)
log.addHandler(file_handler)


def logger(original_func):
    '''This function sets up the logger actions. It should be used as a decorator.'''
    def wrapper(*args, **kwargs):
        log.info('{} function executed with args: {}, and kwargs: {}'.format(original_func.__name__, args, kwargs))
        return original_func(*args, **kwargs)
    return wrapper

@logger
@app.route('/health/')
def health():
    '''This is a very simple check to make sure the app is operational. If
    the response code is 200, the app is functional.'''

    return '<h2>Service is operational!</h2>'


@logger
@app.route('/')
def index():
    '''This renders the home page, which contains some information on how
    to use the app.'''

    return render_template('index.html')


@logger
@app.route('/get_predict/', methods=['GET'])
def get_predict():
    '''This function creates an endpoint that can be used to make a prediction using a GET request. 
    It outputs a JSON file with the variables and their associated prediction.
    To use it, values for the variables must be passed in the url, in the following format:

    /get_predict/?variable1=value1&variable2=value2&variable3=value3&

    There is an example link for how this can be used in the home page.'''

    try:
        data_df = pd.DataFrame(dict(request.args), index=[0])
        predictor.find_missing_cols(data_df)
        precipitaciones, banco_central = predictor.separate_new_data(data_df)

        return predictor.make_prediction(precipitaciones, banco_central).to_json(orient='records')
    
    except Exception as e:

        return jsonify({'error': str(e)})


@logger
@app.route('/post_predict/', methods=['POST'])
def post_predict():
    '''This function creates an endpoint that can be used to make a series of 
    predictions using a POST request. It outputs a JSON file with the variables 
    and their associated predictions. To use it, submit a JSON file in the POST request.'''

    try:
        content_type = request.headers.get('Content-Type')
        if content_type != 'application/json':
            return 'Content-Type not supported! Please submit JSON files only.'

        data_json = request.get_json()
        data_df = pd.DataFrame(data_json)

        predictor.find_missing_cols(data_df)
        precipitaciones, banco_central = predictor.separate_new_data(data_df)
    
        return predictor.make_prediction(precipitaciones, banco_central).to_json(orient='records')
    
    except Exception as e:
        
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run()