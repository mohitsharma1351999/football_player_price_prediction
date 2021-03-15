from flask import Flask,render_template, request
import pickle
import pandas as pd 
import numpy as np 

app = Flask(__name__)

# Loading the models
gbr_model = pickle.load(open('gbr_model.pkl', 'rb'))
min_max = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def predict():
    data = {}
    data['age'] = [int(request.form.get('age'))]
    data['position_cat'] = [int(request.form.get('position'))]
    data['page_views'] = [int(request.form.get('page_views'))]
    data['fpl_value'] = [float(request.form.get('fpl_value'))]
    data['fpl_points'] = [int(request.form.get('fpl_points'))]
    data['region'] = [int(request.form.get('region'))]
    data['big_club'] = [int(request.form.get('big_club'))]
    data['fpl_sel'] = [float(request.form.get('fpl_per'))]
    
    algorithm = str(request.form.get('algorithm'))
    data = pd.DataFrame.from_dict(data)
    
    print(algorithm)
    # apply min-max
    # data = min_max.fit_transform(data)
    final_features = data.to_numpy()
    
    prediction = gbr_model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template("results.html", prediction = output)

if __name__ == "__main__":
    app.run(debug=True)    