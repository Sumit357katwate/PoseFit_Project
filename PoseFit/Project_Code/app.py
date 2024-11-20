# app.py

from flask import Flask, render_template, request
from F_bicep import count_curls
from F_shoulder import shoulder_training_logic
from F_squats import squats_training_logic

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/static/style.css')
def style():
    return app.send_static_file('style.css')


@app.route('/train_biceps', methods=['POST'])
def train_biceps():
    count_curls()
    return "Training biceps functionality goes here!"

@app.route('/train_shoulders', methods=['POST'])
def train_shoulders():
    shoulder_training_logic()
    return "Training shoulders functionality goes here!"

@app.route('/train_squats', methods=['POST'])
def train_squats():
    squats_training_logic()
    return "Training squats functionality goes here!"

if __name__ == '__main__':
    app.run(debug=True)
