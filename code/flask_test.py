# encoding: utf-8
"""
 @project:id_card_recognization
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 6/11/19 9:30 PM
 @desc:
"""
import os
import json
import time
import shutil
from flask import Flask
from flask import request
from flask import redirect
from flask import jsonify

app = Flask(__name__)


@app.route('/hello')
def hello_world():
    return 'hello world!'


@app.route('/recognition', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        path = '../res/pic_input/'
        now_time = int(round(time.time() * 1000))
        path_img = path + now_time + '.png'
        print(path_img)
        f.save(path_img)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8888)
