# coding=utf-8

import numpy as np
import cv2
import sys

sys.path.append('F:\python\MNIST')
import mnist

'''通过方差的百分比来计算将数据降到多少维是比较合适的，
函数传入的参数是特征值和百分比percentage，返回需要降到的维度数num'''


def eigValPct(eigVals, percentage):
    sortArray = np.sort(eigVals)  # 使用numpy中的sort()对特征值按照从小到大排序
    sortArray = sortArray[-1::-1]  # 特征值从大到小排序
    arraySum = sum(sortArray)  # 数据全部的方差arraySum
    tempSum = 0
    num = 0
    for i in sortArray:
        tempSum += i
        num += 1
        if tempSum >= arraySum * percentage:
            return num


'''pca函数有两个参数，其中dataMat是已经转换成矩阵matrix形式的数据集，列表示特征；
其中的percentage表示取前多少个特征需要达到的方差占比，默认为0.9'''


def pca(dataMat, percentage=0.9):
    meanVals = np.mean(dataMat, axis=0)  # 对每一列求平均值，因为协方差的计算中需要减去均值
    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved, rowvar=False)  # cov()计算方差
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 利用numpy中寻找特征值和特征向量的模块linalg中的eig()方法
    k = eigValPct(eigVals, percentage)  # 要达到方差的百分比percentage，需要前k个向量
    eigValInd = np.argsort(eigVals)  # 对特征值eigVals从小到大排序
    eigValInd = eigValInd[:-(k + 1):-1]  # 从排好序的特征值，从后往前取k个，这样就实现了特征值的从大到小排列
    redEigVects = eigVects[:, eigValInd]  # 返回排序后特征值对应的特征向量redEigVects（主成分）
    lowDDataMat = meanRemoved * redEigVects  # 将原始数据投影到主成分上得到新的低维数据lowDDataMat
    reconMat = (lowDDataMat * redEigVects.T) + meanVals  # 得到重构数据reconMat
    return lowDDataMat, reconMat


# def get_K(dataMat, percentage):
#     meanVals = np.mean(dataMat, axis=0)  # 对每一列求平均值，因为协方差的计算中需要减去均值
#     meanRemoved = dataMat - meanVals
#     covMat = np.cov(meanRemoved, rowvar=False)  # cov()计算方差
#     eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 利用numpy中寻找特征值和特征向量的模块linalg中的eig()方法
#     k = eigValPct(eigVals, percentage)  # 要达到方差的百分比percentage，需要前k个向量
#     return k


if __name__ == "__main__":
    # train, valid, test = mnist.read_data_sets('F:\python\MNIST')
    # img = train.images
    img = cv2.imread("5.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lowDDataMat, reconMat = pca(np.array(img_gray))
    reconMat = np.real(reconMat)
    lowDDataMat = np.real(lowDDataMat)

    cv2.imwrite("51.jpg", lowDDataMat)
    cv2.imwrite("52.jpg", reconMat)

    print(img_gray)
    print(lowDDataMat)
    print(reconMat)

    print(np.array(img_gray).shape)
    print(np.array(lowDDataMat).shape)
    print(np.array(reconMat).shape)
