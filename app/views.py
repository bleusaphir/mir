import os
import sys
from flask import Flask, render_template, request, url_for, flash, redirect

app = Flask(__name__)

@app.route('/run_algo', methods=['POST', 'GET'])
def run_algo():
    print('Hello world!', file=sys.stderr)
    des = request.form['des']
    top = int(request.form['top'])
    img = request.form['image']
    import app.run_algo as run_algo
    IMG_LIST = run_algo.search(img, top, des)
    return render_template('images.html', images = IMG_LIST)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/images", methods=['POST', 'GET'])
def images():

    IMG_LIST = [image for image in os.listdir('app/static/')]
    return render_template("images.html", images=IMG_LIST)

    

@app.route("/graphes")
def graphiques():
    return render_template("graphes.html")