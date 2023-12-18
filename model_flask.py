from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

chemin_fichier = 'parkinsons_updrs.data'
data = pd.read_csv(chemin_fichier, sep=',', header=0)
X = data.drop(columns=['motor_UPDRS', 'total_UPDRS'])
y_updrs = data[['motor_UPDRS', 'total_UPDRS']]
X_normalized = scaler_X.fit_transform(X)
y_normalized_motor_UPDRS = scaler_y.fit_transform(y_updrs)

loaded_model = tf.keras.models.load_model('my_model.h5')

@app.route('/')
def landing_page():
    return render_template('landing_page.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    columns = ['subject', 'age', 'sex', 'test_time',
               'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
               'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
               'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']
    if request.method == 'POST':
        values = [float(request.form[column]) for column in columns]
        array = np.array(values).reshape(1, -1)

        X_normalized_array = scaler_X.transform(array)
        ss = loaded_model.predict(X_normalized_array)
        ss = scaler_y.inverse_transform(ss)
        print('my ss is : ', ss)

        return redirect(url_for('prediction', data=ss))

    return render_template('index.html', columns=columns)

@app.route('/prediction')
def prediction():
    # Retrieve the data parameter from the URL
    data_param = request.args.get('data', '')

    # Split the string into individual values
    values = data_param.replace('[', '').replace(']', '').split()

    data_list = [float(val) for val in values]

    return render_template('prediction.html', data=data_list)

if __name__ == '__main__':
    app.run(debug=True)
