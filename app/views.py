import os
import sys
from flask import Flask, render_template,request
import pyrebase


app = Flask(__name__)

firebaseconfig={
  'apiKey': "AIzaSyBoYlBhL8hW_AhVQ64CcQTQsv-Xkx5e2TY",
  'authDomain': "mir-projet.firebaseapp.com",
  'projectId': "mir-projet",
  'storageBucket': "mir-projet.appspot.com",
  'messagingSenderId': "874515404680",
  'appId': "1:874515404680:web:1b262e90d6da5dbab98c3d",
  'databaseURL':""
}

firebase = pyrebase.initialize_app(firebaseconfig)
auth = firebase.auth()

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Obtenir les données de formulaire
        username = request.form['username']
        password = request.form['password']

        # Vérifiez les informations d'identification de l'utilisateur
        try :
            # Redirigez l'utilisateur vers une page protégée
            auth.sign_in_with_email_and_password(username,password)
        except :
            # Affiche un message d'erreur
            return render_template('login.html')
    else:
        # Affichez le formulaire de connexion
        return render_template('login.html')

    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route("/images")
def images():
    if len(IMG_LIST) > 0:
        return render_template("images.html", images=IMG_LIST)
    else:
        return render_template('index.html')


@app.route("/graphes")
def graphiques():
    return render_template("graphes.html")

@app.route('/run_algo', methods=['POST', 'GET'])
def run_algo():
    des = request.form['des']
    top = int(request.form['top'])
    img = request.form['image']
    sim = request.form['similarity']
    import app.run_algo as run_algo
    IMG_LIST, INPUT_IM, GRAPH = run_algo.search(img, top, des, sim)
    return render_template('images.html', input_im = INPUT_IM, images = IMG_LIST, graph = GRAPH)