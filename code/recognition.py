# encoding: utf-8
"""
 @project:id_card_recognization
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 6/2/19 12:06 PM
 @desc:
"""
from PIL import Image
import pytesseract
import time
import matplotlib.pyplot as plt  # plt 用于显示图片

time1 = time.time()


# 二值化算法
def binarizing(img, threshold):
    pix_data = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            if pix_data[x, y] < threshold:
                pix_data[x, y] = 0
            else:
                pix_data[x, y] = 255
    return img


# 去除干扰线算法
def de_point(img):  # input: gray image
    pix_data = img.load()
    w, h = img.size
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            count = 0
            if pix_data[x, y - 1] > 245:
                count = count + 1
            if pix_data[x, y + 1] > 245:
                count = count + 1
            if pix_data[x - 1, y] > 245:
                count = count + 1
            if pix_data[x + 1, y] > 245:
                count = count + 1
            if count > 2:
                pix_data[x, y] = 255
    return img


# 身份证号码识别
def identity_ocr(pic_path):
    # 身份证号码截图
    img_1 = Image.open(pic_path)
    w, h = img_1.size
    # 将身份证放大3倍
    out = img_1.resize((w * 3, h * 3), Image.ANTIALIAS)
    region = (125 * 3, 200 * 3, 370 * 3, 250 * 3)
    # 裁切身份证号码图片
    crop_img = out.crop(region)
    # 转化为灰度图
    img = crop_img.convert('L')
    img = binarizing(img, 100)
    img = de_point(img)
    plt.imshow(img)
    plt.show()
    code = pytesseract.image_to_string(img)
    print("识别该身份证号码是:" + str(code))


if __name__ == '__main__':
    img_path = "../res/2.jpg"
    identity_ocr(img_path)
    time2 = time.time()
    print(u'总共耗时：' + str(time2 - time1) + 's')
