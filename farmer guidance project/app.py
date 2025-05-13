from flask import Flask, render_template, url_for, request, jsonify
import joblib
import pandas as pd

km = joblib.load('E:/Project/cropRecommendationSystem/farmer guidance project/kmeans_model.lb')
std = joblib.load('E:/Project/cropRecommendationSystem/farmer guidance project/standard_scaler.lb')
df = pd.read_csv('E:/Project/cropRecommendationSystem/farmer guidance project/data_to_be_filter.csv')


def find_label(grp_no):
    grp = df[df['group_15'] == grp_no]
    return list(grp['label'].value_counts().keys())

app = Flask(__name__)

@app.route('/')

def home():
    return render_template('home.html')



@app.route('/form')

def form():

    return render_template('form.html')

@app.route('/submitdata', methods = ['GET', 'POST'])

def submitdata():

    if request.method == 'POST':
        n = float(request.form['N'])

        p = float(request.form['P'])

        k = float(request.form['K'])

        temperature = float(request.form['temperature'])

        humidity = float(request.form['humidity'])

        rainfall = float(request.form['rainfall'])

        ph = float(request.form['ph'])



        data = [n, p, k, temperature, humidity, ph, rainfall]

        data_transformed = std.transform([data])

        pred = km.predict(data_transformed)

        crops = find_label(pred[0])

        # return jsonify(crops)

        return render_template('output.html', crops=crops)

if __name__ == '__main__':

    app.run('0.0.0.0', port=4747,debug=True)
