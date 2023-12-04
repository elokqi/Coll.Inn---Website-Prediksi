# app.py
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

app = Flask(__name__)
data = pd.read_csv('data.csv')

@app.route('/')
def home():
    return render_template('start.html')  # Change here

@app.route('/index')  # Add a new route for index.html
def index():
    return render_template('index.html')
    
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Ambil data dari form
        gaji = float(request.form['gaji'])
        usia = int(request.form['usia'])
        pendidikan = int(request.form['pendidikan'])

        # Pisahkan atribut dan label
        X = data[['parent_salary', 'parent_age', 'parent_was_in_college']]
        y = data['will_go_to_college']


        # Bagi data menjadi data pelatihan dan pengujian
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Inisialisasi model Naive Bayes
        model = GaussianNB()

        # Latih model pada data pelatihan
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)

        # Prediksi menggunakan input dari form
        prediksi = model.predict([[gaji, usia, pendidikan]])

        return render_template('result.html', prediction=prediksi[0])
        

if __name__ == '__main__':
    app.run(debug=True)
