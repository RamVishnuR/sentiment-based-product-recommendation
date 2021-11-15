from flask import Flask, render_template, request, redirect, url_for, jsonify
import pickle
from model import Recommendation

recommend = Recommendation()
app = Flask(__name__)

import os
from flask import send_from_directory


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/', methods = ['POST', 'GET'])
def home():
    flag = False 
    data = ""
    if request.method == 'POST':
        flag = True
        user = request.form["userid"]
        data=recommend.getTopProductsHTML(user)
    return render_template('index.html', data=data, flag=flag)

@app.route('/userList', methods = ['GET'])
def userList():
    data=recommend.getUsers()
    return data

@app.route('/productList', methods = ['GET'])
def productList():
    user=request.args.get("userid")
    data=recommend.getTopProductsJSON(user)
    return data

@app.route('/analysText', methods = ['GET'])
def analysText():
    text=request.args.get("text")
    data=recommend.analyiseSentiment(text)
    return data

if __name__ == '__main__' :
    app.run(debug=True )
