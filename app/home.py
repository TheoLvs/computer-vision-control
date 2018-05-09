#! /usr/bin/python
# -*- coding:utf-8 -*-

from flask import Flask
from flask import render_template
from flask import url_for
from flask import request
from flask import jsonify
from flask import redirect
from flask import flash
from flask_login import current_user, login_user, logout_user, login_required
from flask_login import LoginManager
import json
import random
import sys

# https://github.com/tensorflow/tfjs-models/tree/master/posenet
# https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5


#=============================================================================================================
# CONFIGURATION
#=============================================================================================================


#------------------------------------------------------------------------------
# APP CONFIGURATION

app = Flask("VisionControl")
app.secret_key = "VisionControl"









#=============================================================================================================
# VIEWS
#=============================================================================================================





#------------------------------------------------------------------------------
# HOMEPAGE VIEW
@app.route('/home',methods=['GET', 'POST'])
@app.route('/',methods=['GET', 'POST'])
def homepage():
    return render_template('home.html')




#------------------------------------------------------------------------------
# 404 VIEW

@app.route('/404',methods=['GET', 'POST'])
# @login_required
def error_view():
    return render_template('404.html')





if __name__ == '__main__':
    app.run(debug=True)


