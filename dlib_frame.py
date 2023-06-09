# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 14:14:15 2023

@author: carterzhang
"""

import dlib
import cv2


def dlib_detector_face(img_path):
    print("Processing file: {}".format(img_path))
    img = cv2.imread(img_path)
    #dlib正面人脸检测器
    detector = dlib.get_frontal_face_detector()
    #1 表示图像向上采样一次，图像将被放大一倍，这样可以检测更多的人脸
    dets = detector(img, 1)
    print('dets:', dets)  # dets: rectangles[[(118, 139) (304, 325)]]
    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        # 人脸的左上和右下角坐标
        left = d.left()
        top = d.top()
        right = d.right()
        bottom = d.bottom()
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, left, top, right, bottom))
        cv2.rectangle(img, (left, top), (right, bottom), color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    cv2.imshow('detect result', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    
    dlib_detector_face('./images/single_face.jpg')
    #dlib_detector_face('./images/multi_face.jpg')
