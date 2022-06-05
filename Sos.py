import json
import os

import requests
from flask import Flask, jsonify
import numpy as np
from sqlalchemy import JSON

import PreProcessingFun as pre
import CISIcleaning as CIclean
import TF_IDF as DD
import QueryAction as QQ
import Evaloation as EE

import Data1 as DDDD

import main as MM

app = Flask(__name__)


@app.route("/")
def home():


    return "Hello, Flask!"


from flask import request


@app.route('/api/add_message', methods=['POST', 'GET'])
def add_message():
    return "username"


from markupsafe import escape






if __name__ == '__main__':
    app.run(debug=False, port=4000)
