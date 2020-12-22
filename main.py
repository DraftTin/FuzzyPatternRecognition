import numpy as np
import pandas as pd

def normalize(data):
    """标准化"""
    data_normalized = np.zeros_like(data)
    rows = data.shape[0]
    cols = data.shape[1]
    for i in range(rows):
        for j in range(cols):
            maxVal = max(data[:, j])
            minVal = min(data[:, j])
            data_normalized[i, j] = (data[i, j] - minVal) / (maxVal - minVal)
    return data_normalized

def innerProduct(A, B):
    """两项内积"""
    return max([min(valA, valB) for valA, valB in zip(A, B)])

def outterProduct(A, B):
    """两项外积"""
    return min([max(valA, valB) for valA, valB in zip(A, B)])

def calcN_A_B(A, B):
    """计算两项的贴近度"""
    return  (1 / 2) * (innerProduct(A, B) + (1 - outterProduct(A, B)))

def getFuzzySimilityMatrix(X_normalized):
    """计算模糊相似矩阵"""
    rows = X_normalized.shape[0]
    cols = X_normalized.shape[1]
    # n * n 矩阵
    similityMatrix = np.zeros((rows, rows))
    for i in range(rows):
        for j in range(rows):
            if i == j:
                similityMatrix[i, j] = 1
            else:
                numA = sum([min(X_normalized[i, k], X_normalized[j, k]) for k in range(cols)])
                numB = sum([X_normalized[i, k] + X_normalized[j, k] for k in range(cols)])
                similityMatrix[i, j] = 2 * numA / numB
    return similityMatrix

def fuzzyMatrixMultiply(X, Y):
    """模糊矩阵相乘, 用于计算传递闭包"""
    rows = X.shape[0]
    cols = Y.shape[1]
    XY = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            XY[i, j] = innerProduct(X[i, :], Y[:, j])
    return XY

def getEquivalenceMatrix(R):
    """计算传递闭包"""
    R_ = fuzzyMatrixMultiply(R, R)
    while not (R == R_).all():
        R = R_
        R_ = fuzzyMatrixMultiply(R, R)
    return R

def cutMatrix(R, thresh):
    """按照λ的值进行截割"""
    R[R > thresh] = 1
    R[R <= thresh] = 0
    return R

def getClusterRes(matrixCut, X):
    """按照截割的结果进行聚类划分, 并对划分各组进行聚类求平均值, 返回分组和聚类的结果"""
    n = matrixCut.shape[0]
    totalSet = set(range(1, n + 1))
    groupSet = list()
    i = 0
    while totalSet != set([]):
        tmp = set([k + 1 for k in range(n) if matrixCut[i, k] == 1])
        i += 1
        if tmp.issubset(totalSet):
            groupSet.append(tmp)
            totalSet -= tmp
    polymerizationRes = np.zeros((len(groupSet), X.shape[1]))
    groupRes = list()
    i = 0
    for group in groupSet:
        groupRes.append(group)
        # 生成每项的和
        for k in group:
            for j in range(X.shape[1]):
                polymerizationRes[i, j] += X[k - 1, j]
        for j in range(X.shape[1]):
            polymerizationRes[i, j] /= len(group)
        i += 1
    return groupRes, polymerizationRes


def getClassfication(A, item):
    """计算item和A中各项的最大贴进度, 从而对item进行分类, 返回最大贴进度和分类结果"""
    maxval = 0
    maxindex = -1
    for i in range(A.shape[0]):
        val = calcN_A_B(A[i, :], item)
        if val > maxval:
            maxval = val
            maxindex = i
    return maxval, maxindex + 1


def cluster(X):
    """对X数据集进行聚合, 返回聚合的分组和对应的聚合后的矩阵"""
    X_normalized = normalize(X)
    similityMatrix = getFuzzySimilityMatrix(X_normalized)
    equivalenceMatrix = getEquivalenceMatrix(similityMatrix)
    matrixCut = cutMatrix(equivalenceMatrix, 0.78)
    return getClusterRes(matrixCut, X)


if __name__ == '__main__':
    X = np.array([
        [1.4, 4, 0.58, 2, 1.67, 6.9],
        [1.38, 5, 0.61, 3, 1.42, 11.9],
        [1.35, 8, 0.59, 3, 1.47, 16.32],
        [3.95, 14, 0.22, 5, 49.79, 63.63],
        [3.95, 6, 0.54, 5, 14.58, 11.11],
        [2, 7, 0.48, 5, 5.28, 14.58],
        [1.45, 14, 0.58, 3, 2.32, 24.13],
        [1.4, 3, 0.51, 3, 1.9, 5.88],
    ])
    np.set_printoptions(precision=3)
    groupRes, plomerizationRes = cluster(X)
    X9 = [[0.95, 10, 0.57, 2, 1.1, 17.5]]
    X10 = [[2.75, 19.1, 0.31, 5, 31.6, 61.6]]
    newX = np.append(plomerizationRes, X9, axis=0)
    newX = np.append(newX, X10, axis=0)
    print(newX)
    newX_normalized = normalize(newX)
    val1, r1 = getClassfication(newX_normalized[0:4], newX_normalized[4])
    val2, r2 = getClassfication(newX_normalized[0:4], newX_normalized[5])
    print(val1, r1)
    print(val2, r2)