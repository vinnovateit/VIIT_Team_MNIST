import numpy as np
from flask import Flask, request, render_template
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI

    '''
    if request.method == 'POST':
        """
        <!-- longitude	latitude	housing_median_age	total_rooms	total_bedrooms	population	households	median_income	avgRooms	avgBedrooms	pop_per_household	<1H OCEAN	INLAND	NEAR BAY	NEAR OCEAN-->
        """
        longitude = request.form['longitude']
        latitude = request.form['latitude']
        housing_median_age = request.form['housing_median_age']
        total_rooms = request.form['total_rooms']
        total_bedrooms = request.form['total_bedrooms']
        population = request.form['population']
        households = request.form['households']
        median_income = request.form['median_income']
        avgRooms = request.form['avgRooms']
        avgBedrooms = request.form['avgBedrooms']
        ocean = request.form['OCEAN']
        inland = request.form['INLAND']
        near_bay = request.form['NEAR BAY']
        near_ocean = request.form['NEAR OCEAN']

        data = [[float(longitude), float(latitude), float(housing_median_age),float(total_rooms), float(total_bedrooms), float(population), float(households), float(median_income), float(avgRooms), float(avgBedrooms), float(ocean), float(inland), float(near_bay), float(near_ocean)]]
        lr = pickle.load(open('model.pkl', 'rb'))
        prediction = lr.predict(data)[0]

        return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)