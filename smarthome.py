from flask import Flask,request,render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

def load_model():
    with open('smarthome.pkl','rb') as file:
        data = pickle.load(file)
    return data

objects = load_model()
model = objects['model']
preprocessor = objects['preprocessor']

@app.route('/')
def homepage():
    return render_template('smarthome.html')

@app.route('/analyse',methods=['POST'])
def predict():
    device = request.form.get('device_type')
    usg = request.form.get('usg(hrs)/day')
    energy = request.form.get('energy_consumption')
    user = request.form.get('user_preference')
    inc = request.form.get('malfuncti_incid')
    age = request.form.get('device_age(m)')
    
    columns = ['device_type', 'usg(hrs)/day', 'energy_consumption', 'user_preference', 'malfuncti_incid', 'device_age(m)']
    x = pd.DataFrame([[device, usg, energy, user, inc, age]], columns=columns)
    x = preprocessor.transform(x)
    
    pred = model.predict(x)
    msg = 'Device is efficient' if pred == 1 else "Device is inefficient"
    
    return render_template('smarthome.html',text=msg)

if __name__ == '__main__':
    app.run(host="0.0.0.0")