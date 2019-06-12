# encoding: utf-8
"""
 @project:id_card_recognization
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 6/12/19 9:54 AM
 @desc:
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
from flask import Flask, request
import time
import json

debug = 0  # 决定是否输出处理过程中的结果,0表示不输出.

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)

app.config['SECRET_KEY'] = '123456'
app.config['UPLOAD_FOLDER'] = '../res/uploaded_files/'  # 设置上传文件夹
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def grey_img(img):
    # 转化为灰度图
    gray = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    # otsu 二值化操作
    return_val, gray = cv2.threshold(gray, 120, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    return gray


# 图像预处理
def preprocess(gray):
    # 二值化操作，但与前面grey_img二值化操作中不一样的是要膨胀选定区域所以是反向二值化
    ret, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    ele = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 10))
    # 膨胀操作
    dilation = cv2.dilate(binary, ele, iterations=1)
    if debug:
        cv2.imwrite("../res/pic_output/binary.png", binary)
        cv2.imwrite("../res/pic_output/dilation.png", dilation)
    return dilation


def find_text_region(img):
    region = []
    # 1. 查找轮廓
    image, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)
        # 面积小的都筛选掉
        if area < 300:
            continue
        # 轮廓近似，作用很小
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)
        # 函数 cv2.minAreaRect() 返回一个Box2D结构 rect：（最小外接矩形的中心（x，y），（宽度，高度），旋转角度）。
        if debug:
            print("rect is: ", rect)
        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 筛选那些太细的矩形，留下扁的
        if height > width * 1.2:
            continue
        # 太扁的也不要
        if height * 18 < width:
            continue
        if width > img.shape[1] / 2 and height > img.shape[0] / 20:
            region.append(box)
    return region


def detect(img):
    # cv2.fastNlMeansDenoisingColored 作用为去噪
    gray = cv2.fastNlMeansDenoisingColored(img, None, 10, 3, 3, 3)
    coefficients = [0, 1, 1]
    m = np.array(coefficients).reshape((1, 3))
    gray = cv2.transform(gray, m)
    if debug:
        cv2.imwrite("../res/pic_output/gray.png", gray)
    # 2. 形态学变换的预处理，得到可以查找矩形的图片
    dilation = preprocess(gray)

    # 3. 查找和筛选文字区域
    region = find_text_region(dilation)
    # 4. 用绿线画出这些找到的轮廓
    for box in region:
        h = abs(box[0][1] - box[2][1])
        w = abs(box[0][0] - box[2][0])
        x_s = [i[0] for i in box]
        y_s = [i[1] for i in box]
        x1 = min(x_s)
        y1 = min(y_s)
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
        if w > 0 and h > 0 and x1 < gray.shape[1] / 2:
            id_img = grey_img(img[y1:y1 + h, x1:x1 + w])
            cv2.imwrite("../res/pic_output/id_img.png", id_img)
            cv2.imwrite("../res/pic_output/contours.png", img)
            return id_img


def recognize_id_card(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (428, 270), interpolation=cv2.INTER_CUBIC)
    id_img = detect(img)
    image = Image.fromarray(id_img)
    result = pytesseract.image_to_string(image, lang='eng', config='--psm 7 digits')
    result = result.replace('§', '5')
    return result


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/recognition', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            now_time = int(round(time.time() * 1000))
            filename = str(now_time) + '.' + file.filename.rsplit('.', 1)[1].lower()
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # 保存完图片之后，开始进行身份证号码提取
            number = recognize_id_card(file_path)
            res = {'card_number': number}
            return json.dumps(res)


if __name__ == '__main__':
    app.run()
