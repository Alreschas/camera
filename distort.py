# -*- coding: utf-8 -*-

import pickle
import cv2
import numpy as np
import time

# Read in an image
img = cv2.imread('test_image3.jpg')
# img = cv2.imread('test_image2.jpg')

nx = 6  # the number of inside corners in x
ny = 9  # the number of inside corners in y


def corners_unwarp(img, nx, ny, mtx=[], dist=[]):
    # 歪みをとる
    if(len(mtx) == 0 or len(dist) == 0):
        undist = img
    else:
        undist = cv2.undistort(img, mtx, dist, None, mtx)

    # グレースケールに変換
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # チェスボードのコーナーを取得
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        # コーナーを描画
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)

        # イメージのサイズ
        img_size = (undist.shape[1], undist.shape[0])

        src = np.float32([corners[0], corners[nx - 1], corners[-1], corners[-nx]])

        # 写像する点
        offsetX = img_size[0] / 4
        offsetY = int((img_size[1] - (img_size[0] - 2 * offsetX) * ny / nx) / 2)
        dst = np.float32([
            [img_size[0] - offsetX, img_size[1] - offsetY],
            [offsetX, img_size[1] - offsetY],
            [offsetX, offsetY],
            [img_size[0] - offsetX, offsetY],
        ])

        # 座標変換行列
        M = cv2.getPerspectiveTransform(src, dst)
        # 座標変換
        warped = cv2.warpPerspective(undist, M, img_size)
        return undist, warped, M


undist, top_down, perspective_M = corners_unwarp(img, nx, ny)
cv2.imwrite("test.png", top_down)
