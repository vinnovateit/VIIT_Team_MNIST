import numpy as np
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


app = Flask(__name__)
model = pickle.load(open(r'C:\Users\Aviral.HP-PAVILION\Desktop\Machine Learning\Projects\California_Housing_Prices\Flask hosting\model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    int_features.append(float(int_features[3])/float(int_features[6]))
    int_features.append(float(int_features[4])/float(int_features[3]))
    int_features.append(float(int_features[4])/float(int_features[6]))
    int_features.append(float(int_features[6])/float(int_features[5]))
    int_features.append(float(int_features[5])/float(int_features[3]))
    int_features.append(float(int_features[5])/float(int_features[4]))
    del (int_features[6])
    del (int_features[4])

    d={"NEAR BAY":[0.0,0.0,0.0,1.0,0.0],"INLAND":[0.0,1.0,0.0,0.0,0.0],'<1H OCEAN':[1.0, 0.0, 0.0, 0.0, 0.0],"ISLAND":[0.0, 0.0, 1.0, 0.0, 0.0],"NEAR OCEAN":[0.0, 0.0, 0.0, 0.0, 1.0]}
    
    if int_features[6]=='NEAR BAY':
        del(int_features[6])
        int_features=d['NEAR BAY']+int_features
    elif int_features[6]=='INLAND':
        del(int_features[6])
        int_features=d['INLAND']+int_features
    elif int_features[6]=='NEAR OCEAN':
        del(int_features[6])
        int_features=d['NEAR OCEAN']+int_features
    elif int_features[6]=='<1H OCEAN':
        del(int_features[6])
        int_features=d['<1H OCEAN']+int_features
    elif int_features[6]=='ISLAND':
        del(int_features[6])
        int_features=d['ISLAND']+int_features
        
    final_features = [np.array(int_features)]
    final_features=np.array(final_features,dtype='object')
    final_features[:,5:]=scaler.fit_transform(final_features[:,5:])
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Estimated price of House is $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
    