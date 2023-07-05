from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model from the pkl file
model = None
with open('mlp_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/userinput', methods=['GET', 'POST'])
def user_input():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Make the prediction
        input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                  columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        prediction = model.predict(input_data)[0]

        return render_template('prediction.html', prediction=prediction)

    return render_template('user_input.html')

if __name__ == '__main__':
    app.run(debug=True)
