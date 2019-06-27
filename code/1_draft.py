# encoding: utf-8
"""
 @project:id_card_recognization
 @author: Jiang Hui
 @language:Python 3.7.2 [GCC 7.3.0] :: Anaconda, Inc. on linux
 @time: 6/17/19 4:27 PM
 @desc: 本代码是第一版,从头到尾的,一步步地完成身份证号码识别功能,作调试用,
        逻辑比较混乱,仅供想了解每个函数具体功能的人参考
"""
import cv2
import numpy as np
import pytesseract

pic_path = "../res/pic_input/43.jpeg"
img = cv2.imread(pic_path, cv2.IMREAD_COLOR)
print(img.shape)
# 第一步处理，判断图片是竖直还是水平，如果是竖直放置的话，就旋转90度
if img.shape[0] > img.shape[1]:
    img = np.rot90(img)

img = cv2.resize(img, (428, 270), interpolation=cv2.INTER_CUBIC)
cv2.imwrite("../res/pic_output/out1.png", img)
# h：参数决定滤波器强度。较高的h值可以更好地消除噪声，但也会删除图像的细节 (10 is ok)
# hForColorComponents：与h相同，但仅适用于彩色图像。 （通常与h相同）
# templateWindowSize：应该是奇数。 （recommended 7）
# searchWindowSize：应该是奇数。 （recommended 21
# 可以去除身份证照片中的纹理
denoise_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
cv2.imwrite("../res/pic_output/out2.png", denoise_img)

# 转换成灰度图
coefficients = [0, 1, 1]
m = np.array(coefficients).reshape((1, 3))
gray = cv2.transform(denoise_img, m)
# gray = cv2.cvtColor(denoise_img, cv2.COLOR_BAYER_BG2GRAY)
cv2.imwrite("../res/pic_output/out3.png", gray)

# 反向二值化图像
# 第一个原图像，第二个进行分类的阈值，第三个是高于（低于）阈值时赋予的新值，第四个是一个方法选择参数，常用的有：
# cv2.THRESH_BINARY（黑白二值）
# cv2.THRESH_BINARY_INV（黑白二值反转）
# cv2.THRESH_TRUNC （得到的图像为多像素值）
# cv2.THRESH_TOZERO
# cv2.THRESH_TOZERO_INV

# 二值化图像,像素点要么为0,要么为255
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 3)
# ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite("../res/pic_output/out4.png", binary)
print(len(binary))

# 自定义的降噪代码,将孤立像素点去除
for i in range(1, len(binary) - 1):
    for j in range(1, len(binary[0]) - 1):
        if binary[i][j] == 255:
            # if成立,则说明当前像素点为孤立点
            if binary[i - 1][j] == binary[i + 1][j] == binary[i][j - 1] == binary[i][j + 1] == 0:
                binary[i][j] = 0

cv2.imwrite("../res/pic_output/out5.png", binary)

# 膨胀操作，将图像变成一个个矩形框，用于下一步的筛选，找到身份证号码对应的区域,(10,5)为水平方向和垂直方向的膨胀size
ele = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
dilation = cv2.dilate(binary, ele, iterations=1)
cv2.imwrite("../res/pic_output/out6.png", dilation)

# 寻找图像中物体的轮廓，参数解释
# image:输入图像，图像必须为8-bit单通道图像，图像中的非零像素将被视为1，0像素保留其像素值，故加载图像后会自动转换为二值图像
# contours:检测到的轮廓，每个轮廓都是以点向量的形式进行存储即使用point类型的vector表示
# hierarchy:可选的输出向量(std::vector)，包含了图像的拓扑信息，作为轮廓数量的表示hierarchy包含了很多元素
# mode轮廓检索模式：
#   RETR_EXTERNAL:表示只检测最外层轮廓，对所有轮廓设置hierarchy[i][2]=hierarchy[i][3]=-1
#   RETR_LIST:提取所有轮廓，并放置在list中，检测的轮廓不建立等级关系
#   RETR_CCOMP:提取所有轮廓，并将轮廓组织成双层结构(two-level hierarchy),顶层为连通域的外围边界，次层位内层边界
#   RETR_TREE:提取所有轮廓并重新建立网状轮廓结构
#   RETR_FLOODFILL：官网没有介绍，应该是洪水填充法
# method:轮廓近似方法：
# CHAIN_APPROX_NONE：获取每个轮廓的每个像素，相邻的两个点的像素位置差不超过1
# CHAIN_APPROX_SIMPLE：压缩水平方向，垂直方向，对角线方向的元素，值保留该方向的重点坐标，如果一个矩形轮廓只需4个点来保存轮廓信息
# CHAIN_APPROX_TC89_L1和CHAIN_APPROX_TC89_KCOS使用Teh-Chinl链逼近算法中的一种
# offset:轮廓点可选偏移量，有默认值Point()，对ROI图像中找出的轮廓并要在整个图像中进行分析时，使用

image, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
img_temp = img.copy()
for i in range(len(contours)):
    cv2.drawContours(img_temp, contours, i, (0, 255, 0), 1)
cv2.imwrite("../res/pic_output/out7.png", img_temp)

img_temp = img.copy()
for i in range(len(contours)):
    cnt = contours[i]
    # 计算该轮廓的面积
    area = cv2.contourArea(cnt)
    print(area)
    # 面积小的都筛选掉
    if area < 500:
        cv2.drawContours(img_temp, contours, i, (0, 0, 255), 1)
        continue
cv2.imwrite("../res/pic_output/out8.png", img_temp)

# 轮廓周长，也被称为弧长，第二个参数用来指定对象是闭合的(True)还是打开的(False)
# epsilon = 0.1 * cv2.arcLength(contours[5], True)
# 轮廓近似，将轮廓形状近似到另外一种由更少点组成的多边形形状
# InputArray curve:一般是由图像的轮廓点组成的点集
# OutputArray approxCurve：表示输出的多边形点集
# double epsilon：主要表示输出的精度，就是另个轮廓点之间最大距离数，5,6,7，，8，，,,，
# bool closed：表示输出的多边形是否封闭
# approx = cv2.approxPolyDP(contours[5], epsilon, True)
# print('------------------------------------------')
# print(approx)
# img_temp = img.copy()
# cv2.drawContours(img_temp, approx, 3, (0, 0, 255), 1)
# cv2.imwrite("../res/pic_output/out9.png", img_temp)

card_number_region = []

img_temp = img.copy()
max_area = 0  # 用于存储最大的矩形面积,筛选出身份证边框所在区域
for i in range(len(contours)):
    # 返回点集cnt的最小外接矩形，(外接矩形中心坐标(x,y),(外接矩形宽，外接矩形高)，旋转角度)
    rect = cv2.minAreaRect(contours[i])
    # box是外接矩形四个点的坐标，如
    #  [[427.  76.]
    #   [417.  76.]
    #   [417.  67.]
    #   [427.  67.]]
    # np.int0()用来去除小数点之后的数字
    box = np.int0(cv2.boxPoints(rect))
    print('-----------')
    print(rect)
    print(box)
    print('************')
    # 水平轴（x轴）逆时针旋转碰到的矩形的第一条边是width，另一条边是height
    # ((5.616669654846191, 140.7300567626953), (20.895458221435547, 239.635986328125), -2.4470486640930176)
    width, height = rect[1]
    print('width:{},height:{}'.format(width, height))
    # cv2.drawContours(img_temp, [box], 0, (0, 0, 255), 1)
    # cv2.imwrite("../res/pic_output/out9.png", img_temp)
    if 0 not in box:  # 剔除一些越界的矩形,如果矩阵坐标中包含0,说明该矩阵不是我们想要找的矩阵框
        if 9 < width / height < 16 or 9 < height / width < 16:
            print('######## ########')
            print(width / height)
            print(height / width)
            area = width * height
            if area > max_area:
                max_area = area
                card_number_region = box
cv2.drawContours(img_temp, [card_number_region], 0, (0, 0, 255), 1)
cv2.imwrite("../res/pic_output/out9.png", img_temp)

# card_number_region是一个矩形四个点的坐标
print(card_number_region)
# 根据四个点的左边裁剪区域
h = abs(card_number_region[0][1] - card_number_region[2][1])
w = abs(card_number_region[0][0] - card_number_region[2][0])
x_s = [i[0] for i in card_number_region]
y_s = [i[1] for i in card_number_region]
x1 = min(x_s)
y1 = min(y_s)
crop_img = img[y1:y1 + h, x1:x1 + w]
cv2.imwrite("../res/pic_output/out10.png", crop_img)


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
    print("调整角度：", angle)
    return angle


def horizontal_correct(img):
    degree = calc_degree(img)
    img_rotate = rotate_image(img, degree)
    return img_rotate


image_correct = horizontal_correct(crop_img)
cv2.imwrite("../res/pic_output/out11.png", image_correct)

result = pytesseract.image_to_string(image_correct, lang='eng', config='--psm 7 sfz')
print('the number of the card is', result)

# [[157 237]
#  [156 217]
#  [380 212]
#  [381 233]]
