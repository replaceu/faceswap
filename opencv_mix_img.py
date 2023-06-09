# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 11:25:13 2023

@author: carterzhang
"""

import dlib
import cv2
import numpy as np

BLUR_AMOUNT = 51
SWAP = 1


detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor('./model/shape_predictor_68_face_landmarks.dat')

def padding_edge(rectangle, shape, padding=0.2):
    """
    为边界增加padding，比例为padding=0.2，同时处理边界溢出的情况
    :param rectangle: 矩阵边框的上下左右 [top, bottom, left, right]
    :param shape: img.shape 图片的分辨率
    :param padding: 边界填充的比例
    :return: 填充后的矩形
    """
    # padding的高和宽
    height = (rectangle[1] - rectangle[0]) * padding
    length = (rectangle[3] - rectangle[2]) * padding

    # 填充矩形，同时边界处理
    rectangle[0] = rectangle[0] - height if rectangle[0] - height > 0 else 0
    rectangle[1] = rectangle[1] + height if rectangle[1] + height < shape[0] else shape[0]
    rectangle[2] = rectangle[2] - length if rectangle[2] - length > 0 else 0
    rectangle[3] = rectangle[3] + length if rectangle[3] + length < shape[1] else shape[1]

    return np.array(rectangle, dtype=np.int_).tolist()

def get_landmarks(img, padding=0.2):
    """
    利用dlib库函数得到img中人脸的矩形边框和特征点坐标
    :param img: 人脸图片
    :param padding:
    :return: 矩形边框list和人脸特征点坐标list
    """
    # 利用detector模型检测人脸边框rects
    rects = detector(img, 1)
    rect_list = []
    landmark_list = []

    # 判断识别到的人脸个数
    if len(rects) > 5:
        print('[Warning]: Too much face detected...(more than 5)')
    elif 1 < len(rects) < 5:
        print('[Warning]: More than one face in picture(2~5)')
    elif len(rects) == 0:
        print('[Error]: No face detected...')
    elif len(rects) == 1:
        # print("检测到人脸的个数为1个")
        rects = [rects[0]]

    # 遍历所有人脸，添加对应特征点到列表中
    for item in rects:
        # 二维数组landmark_list的元素为[(x, y), ...]
        landmark_list.append([(p.x, p.y) for p in predictor(img, item).parts()])
        # 二维数组rect_list的元素为[(top, bottom, left, right), ...]
        edges = [item.top(), item.bottom(), item.left(), item.right()]
        rect_list.append(padding_edge(edges, img.shape, padding=padding))
        print("人脸方框的坐标位置为(上，下，左，右){}".format(rect_list[0]))
    return landmark_list, rect_list

#img1 = cv2.imread('swp/swp_{}.jpg'.format(background_id)).astype(np.uint8)

img_path = './images/single_face_3.jpg'
img = cv2.imread(img_path)
print("Processing file: {}".format(img_path))
get_landmarks(img)


def img_mix(mask, img1, img2, blur_amount=BLUR_AMOUNT, swap=SWAP):
    """
    图片融合
    :param mask: 脸部mask
    :param img1: 模板底片
    :param img2: 拍摄人脸
    :param blur_amount: 高斯核大小
    :param swap:
    :return:
    """

    # mask：脸部纯白色，背景纯黑色
    mask = cv2.GaussianBlur(mask * 255.0, (blur_amount, blur_amount), 0) / 255.0
    mask[np.where(mask > 0.8)] = 1.0
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)

    # temp1：模板图片，背景保留，脸部去除
    temp1 = img1 * (1.0 - mask)
    # cv2.imshow('temp1', np.clip(temp1, 0, 255).astype(np.uint8))
    # cv2.waitKey(0)

    # temp2：拍摄照片，背景去除，脸部保留
    temp2 = img2 * mask
    # cv2.imshow('temp2', np.clip(temp2, 0, 255).astype(np.uint8))
    # cv2.waitKey(0)

    # temp3：模板照片，背景去除，脸部保留
    temp3 = img1 * mask
    # cv2.imshow('temp3', np.clip(temp3, 0, 255).astype(np.uint8))
    # cv2.waitKey(0)

    res = temp1 + temp2 * swap + temp3 * (1.0 - swap)
    res = np.clip(res, 0, 255).astype(np.uint8)
    return res