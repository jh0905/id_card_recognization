# encoding: utf-8
"""
 @project:id_card_recognization
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 6/19/19 10:48 AM
 @desc: 将id_recognition实现的功能,打包放在flask服务器上,提供API接口
"""
import cv2
import numpy as np
import pytesseract
from flask import Flask, request
import time
import json
import os

# ******************* flask服务器端基本配置 ***************************
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)

app.config['SECRET_KEY'] = '123456'
path = os.getcwd() + '/uploaded_files/'
if not os.path.exists(path):
    os.makedirs(path)
app.config['UPLOAD_FOLDER'] = path  # 设置上传文件夹
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def img_preprocess(pic_path, coefs):
    '''
    图片预处理: resize -> denoise -> transform to grey
    :param pic_path:
    :return: image
    '''
    img = cv2.imread(pic_path, cv2.IMREAD_COLOR)
    img_resize = cv2.resize(img, (428, 270), interpolation=cv2.INTER_CUBIC)  # resize照片为428*270
    img_denoise = cv2.fastNlMeansDenoisingColored(img_resize, None, 10, 10, 7, 21)  # 降噪
    # 转换成灰度图
    # coefficients = [0, 1, 1]
    m = np.array(coefs).reshape((1, 3))
    img_gray = cv2.transform(img_denoise, m)
    # 反向二值化图像
    img_binary_inv = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 3)
    # 自定义的降噪代码,将孤立像素点去除
    img_binary_inv = denoise(img_binary_inv)
    # 膨胀操作,将图像变成一个个矩形框，用于下一步的筛选，找到身份证号码对应的区域
    ele = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))  # 经过测试,(10,5)为水平和垂直方向的膨胀size
    img_dilation = cv2.dilate(img_binary_inv, ele, iterations=1)
    return img_resize, img_dilation


def denoise(binary):
    """
    当某个像素点值为255,但是其上下左右像素值都是0的话,认做为孤立像素点,将其值也设置为0
    :param binary:二值图
    :return:binary
    """
    for i in range(1, len(binary) - 1):
        for j in range(1, len(binary[0]) - 1):
            if binary[i][j] == 255:
                # if条件成立的话,则说明当前像素点为孤立点
                if binary[i - 1][j] == binary[i + 1][j] == binary[i][j - 1] == binary[i][j + 1] == 0:
                    binary[i][j] = 0
    return binary


def find_number_region(img):
    '''
    在膨胀处理后的img中,找到身份证号码所在矩形区域,
    :param img:
    :return: box
    '''
    image, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    card_number_region = []  # 用来保存最终返回的region
    max_area = 0  # 存储最大的矩形面积,用于筛选出身份证边框所在区域
    for i in range(len(contours)):
        # 返回点集cnt的最小外接矩形，(外接矩形中心坐标(x,y),(外接矩形宽，外接矩形高)，旋转角度)
        rect = cv2.minAreaRect(contours[i])
        box = np.int0(cv2.boxPoints(rect))  # box是外接矩形四个点的坐标,np.int0()用来去除小数点之后的数字
        width, height = rect[1]
        if 0 not in box:  # 剔除一些越界的矩形,如果矩阵坐标中包含0,说明该矩阵不是我们想要找的矩阵框
            if 9 < width / height < 16 or 9 < height / width < 16:
                area = width * height
                if area > max_area:
                    max_area = area
                    card_number_region = box
    return card_number_region


def get_number_img(origin_img, region):
    '''
    根据上一步找到的边框,从原始图像中,裁剪出身份证号码区域的图像
    :param origin_img:
    :param region:
    :return: image
    '''
    # 根据四个点的左边裁剪区域
    h = abs(region[0][1] - region[2][1])
    w = abs(region[0][0] - region[2][0])
    x_s = [i[0] for i in region]
    y_s = [i[1] for i in region]
    x1 = min(x_s)
    y1 = min(y_s)
    return origin_img[y1:y1 + h, x1:x1 + w]


# ************* start 身份证号码图像区域水平矫正代码 start *******************


# 度数转换
def degree_trans(theta):
    res = theta / np.pi * 180
    return res


# 逆时针旋转图像degree角度（原尺寸）
def rotate_image(src, degree):
    # 旋转中心为图像中心
    h, w = src.shape[:2]
    # 计算二维旋转的仿射变换矩阵
    rotate_matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degree, 1)
    # 仿射变换，背景色填充为白色
    rotate = cv2.warpAffine(src, rotate_matrix, (w, h), borderValue=(255, 255, 255))
    return rotate


# 通过霍夫变换计算角度
def calc_degree(img):
    mid_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst_image = cv2.Canny(mid_image, 50, 200, 3)
    line_image = img.copy()
    # 通过霍夫变换检测直线
    # 第4个参数就是阈值，阈值越大，检测精度越高
    lines = cv2.HoughLines(dst_image, 1, np.pi / 180, 15)
    # 由于图像不同，阈值不好设定，因为阈值设定过高导致无法检测直线，阈值过低直线太多，速度很慢
    count = 0
    # 依次画出每条线段
    for i in range(len(lines)):
        for rho, theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(round(x0 + 1000 * (-b)))
            y1 = int(round(y0 + 1000 * a))
            x2 = int(round(x0 - 1000 * (-b)))
            y2 = int(round(y0 - 1000 * a))
            # 只选角度最小的作为旋转角度
            count += theta
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)

    # 对所有角度求平均，这样做旋转效果会更好
    average = count / len(lines)
    angle = degree_trans(average) - 90
    return angle


def horizontal_correct(img):
    degree = calc_degree(img)
    # 在测试中发现,如果扭曲角度角度(大于3),则进行水平矫正,否则不进行矫正
    if abs(degree) > 5:
        img_rotate = rotate_image(img, degree)
        return img_rotate
    return img


# ************* end 身份证号码图像区域水平矫正代码 end *******************


def tesseract_ocr(img):
    id_number = pytesseract.image_to_string(img, lang='eng', config='--psm 7 sfz')

    # 手动处理,识别结果中可能出现的错误

    # python清除字符串中非数字字符(xX§除外)
    id_number = ''.join(list(filter(lambda ch: ch in '0123456789xX§', id_number)))
    id_number = id_number.replace('x', 'X')
    #
    if len(id_number) == 19 and '§' in id_number:
        id_number = id_number.replace('§', '')
    else:
        id_number = id_number.replace('§', '5')

    # 在测试中发现,tesseract会把4识别成46,所以这里直接手动替换
    if len(id_number) == 19 and '46' in id_number:
        id_number = id_number.replace('46', '4')
    return id_number


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

            # ********************* begin **************************
            # step 1:preprocess the image
            image_resize, image_preprocessed = img_preprocess(file_path, [0, 1, 1])

            # step 2:find id number_region
            number_region = find_number_region(image_preprocessed)

            # 以[0,1,1]作为灰度化转换参数时,偶尔导致找不到身份证号码所在区域,所以如果出错,用[0,1,0]进行转换
            if len(number_region) == 0:
                image_resize, image_preprocessed = img_preprocess(file_path, [0, 1, 0])
                number_region = find_number_region(image_preprocessed)

            # step 3:get id number image
            image_id_number = get_number_img(image_resize, number_region)

            # step 4:horizontal correct the image, if necessary.
            image_correct = horizontal_correct(image_id_number)

            # step 5:recognize the id number
            number = tesseract_ocr(image_correct)
            # ********************* end **************************

            if number is None:
                res = {'code': -1, 'card_number': '-1', 'info': 'The photo has too low pixel to be identified!'}
            elif len(number) != 18:
                res = {'code': -1, 'card_number': number, 'info': 'failure'}
            else:
                res = {'code': 0, 'card_number': number, 'info': 'success'}
            return json.dumps(res)


if __name__ == '__main__':
    app.run('0.0.0.0', port=8880)
