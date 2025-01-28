from flask import Flask, request, app,render_template
from flask import Response
import pickle
import numpy as ny
import pandas as pd
import os


application = Flask(__name__)
app=application

# scaler_path = os.path.join(r"D:\Pregrad_Course\Aman_Algerian_Forest_Fire\Model","standardScalar.pkl")
scaler =pickle.load(open(r"D:\Pregrad_Course\Aman_Algerian_Forest_Fire\Model\standardScaler.pkl", "rb"))
model  = pickle.load(open(r"D:\Pregrad_Course\Aman_Algerian_Forest_Fire\Model\modelforPrecision.pkl", "rb"))


@app.route('/')
def index():
    return render_template('index.html')


## Route for Single data point prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=""

    if request.method=='POST':

        Year =int(request.form.get("Year"))
        Month = float(request.form.get('Month'))
        Day = float(request.form.get('Day'))
        Temperature = float(request.form.get('Temperature'))
        ISI = float(request.form.get('ISI'))
        FWI = float(request.form.get('FWI'))
        AlgerianForestFireFunction = float(request.form.get('AlgerianForestFireFunction'))
        Rain = float(request.form.get('Rain'))

        new_data=scaler.transform([[Year,Month,Day,Temperature,ISI,FWI,AlgerianForestFireFunction,Rain]])
        predict=model.predict(new_data)
       
        if predict[0] ==1 :
            result = 'Fire'
        else:
            result ='Not-Fire'
            
        return render_template('single_prediction.html',result=result)

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")