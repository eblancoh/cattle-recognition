from flask import Flask, redirect, url_for, request, render_template, jsonify, session
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from testing import ModelLoad, ImageScore
import operator
import configparser
from keras import backend as K 


# Configuration loading
config = configparser.ConfigParser()
config.read("config.ini")

# Define a flask app
app = Flask(__name__)
app.secret_key = config["app"]["app-secret-key"]

def get_file_path_and_save(request):
    # Get the file from post request
    f = request.files['file']

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'static/img', secure_filename(f.filename))
    f.save(file_path)
    return file_path

def loading_model(facility):
    """Cargamos el modelo fuera del testeo"""
    model_loaded = False
    global model
    global graph
    graph = tf.get_default_graph()
    model = ModelLoad(filepath=os.path.join('./checkpoints', facility, 'chckpt.best.h5')).model_loader()

    model_loaded = True

@app.route('/')
def index():
    # Por defecto limpiamos sesión por si no lanzamos testeo tras cargar el modelo al 
    # volver a la ruto "/" desde index.html.
    K.clear_session()
    # Leemos todas las granjas habilitadas en el sistema
    farms = next(os.walk(os.path.join(os.getcwd(), 'checkpoints')))[1]
    # Sólo nos quedamos con aquellas que tienen un modelo entrenado:
    # Nos ahorramos problemas en la vista choose.html
    data = list()
    for i in farms:
        for fname in os.listdir(os.path.join(os.getcwd(), 'checkpoints', i)):
            if fname.endswith('.h5'):
                data.append({"granja": i})
    return render_template('chooser.html', data=data)

@app.route('/load_model', methods=['GET', 'POST'])
def load_model():
    facility = request.form.get('comp_select')

    # Pasamos la granja elegida como elemento de sesión y 
    session['facility'] = facility
    # Cargamos el modelo para esa granja
    graph = tf.get_default_graph()
    loading_model(facility)
    
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file_path = get_file_path_and_save(request)

        # Necesitamos limpiar sesión para que no crashee en una segunda intentona de testeo
        with graph.as_default():
            preds = ImageScore(model=model, 
                            img=file_path, 
                            farm=session['facility'],
                            version=1).scores()
        
        # Eliminamos la imagen que hemos cargado desde el navegador.
        os.remove(file_path)
        # Devolvemos las predicciones
        print(preds)
        return jsonify(preds)
        
def to_farm():
    session.pop('facility')
    return redirect(url_for('index'))

# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(host='localhost', debug=True)
    # app.run(debug=True)

# TODO: atrás para otra granja
# TODO: Página de despedida
