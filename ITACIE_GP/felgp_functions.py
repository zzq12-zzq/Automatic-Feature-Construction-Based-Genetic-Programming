import random

import cv2
from scipy.ndimage import median_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import sift_features
import numpy
import pylab
from scipy import ndimage
from skimage.filters import gabor
import skimage
from skimage.feature import local_binary_pattern, canny
from skimage.feature import hog
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from PIL import Image
from skimage import data,exposure
from skimage.filters import roberts
from sklearn.decomposition import PCA

def combine(a,b,c,d):
    output =a*d[0]+b*d[1]+c*d[2]
    # for i in range(1, len(args)):
    #     output += args[i]*
    return output

# def linear_svm(x_train, y_train,array,d, cm=0):
#     #parameters c
#     x_train1=x_train[0]
#     c = 10**(cm)
#     classifier = LinearSVC(C=c)
#     num_train = y_train.shape[0] # num_train = 150
#     if num_train == x_train1.shape[0]:
#         y_labels1 = svm_train_model(classifier, x_train1, y_train)
#     else:
#         y_labels1 = test_function_svm(classifier, x_train1[0:num_train,:], y_train, x_train1[num_train:x_train1.shape[0],:])
#
#     # 低级特征
#     x_train2=x_train[1]
#     c = 10**(cm)
#     classifier = LinearSVC(C=c)
#     num_train = y_train.shape[0] # num_train = 150
#     if num_train == x_train2.shape[0]:
#         y_labels2 = svm_train_model(classifier, x_train2, y_train)
#     else:
#         y_labels2 = test_function_svm(classifier, x_train2[0:num_train,:], y_train, x_train2[num_train:x_train2.shape[0],:])
#
#     # 全局特征
#
#     c = 10**(cm)
#     classifier = LinearSVC(C=c)
#     num_train = y_train.shape[0] # num_train = 150
#     if num_train == array.shape[0]:
#         y_labels3 = svm_train_model(classifier, array, y_train)
#     else:
#         y_labels3 = test_function_svm(classifier, array[0:num_train,:], y_train, array[num_train:array.shape[0],:])
#     output =y_labels1*d[0]+y_labels3*d[1]+y_labels2*d[2]
#     #output = y_labels1 + y_labels3  + y_labels2
#     return output

def arr_pad_align(arr1, arr2):
    arr1_len = len(arr1)  # 取第一个数组的长度
    arr2_len = len(arr2)  # 取第二个数组的长度
    arr_list = [(arr1), (arr2)]  # 将数组放入list容器
    arr_matrix = [[arr1_len, arr2_len]]  # 将输入的长度放入二维数组，方便取索引
    max_index = numpy.argmax(arr_matrix, 1).squeeze(0)  # 取最大长度数组的索引
    min_index = numpy.argmin(arr_matrix, 1).squeeze(0)  # 取最小长度数组的索引
    pad_len = numpy.max([arr1_len, arr2_len]) - numpy.min([arr1_len, arr2_len])  # 拿到填充的长度
    arr_min = arr_list[min_index]  # 拿到填充数组
    pad_arr = numpy.pad(arr_min, (0, pad_len), 'constant', constant_values=0)  # 进行填充，尾部为0
                            # (0, pad_len), 前面填充0个0，后面填充pad_len个0
    return pad_arr, arr_list[max_index]  # 返回填充后的结果

# 功能，将特征和标签连一起。
def svmtool(f,sf):
    # 如果为训练集f，sf的shape为150，否则为200
    slabel = sf[0]
    sjumpf=sf[1]
    deepf=f





def linear_svm2(x_train, y_train):
        # 【1】是特征 【0】是标签 [2]是训练集标签
        # x_train=FeaCon2(x_train,shallowf[0])
        # svmtool(x_train,shallowf)
        # print(x_train.shape, y_train.shape, num_train, x_train[0:num_train,:].shape, x_train[num_train:-1,:].shape)
        classifier = LinearSVC(C=1)
        num_train = y_train.shape[0]
        if num_train == x_train.shape[0]:
            y_labels = svm_train_model(classifier, x_train, y_train)
            return y_labels, x_train
        else:
            y_labels3 = svm_train_model(classifier, x_train[0:num_train, :], y_train)
            y_labels = test_function_svm(classifier, x_train[0:num_train, :], y_train,
                                         x_train[num_train:x_train.shape[0], :])
            return y_labels, x_train, y_labels3

            # 训练集有



def Fusion_SVM(shallowf, x_train, cm, y_train,Filter):
    Value1=Filter[0]
    Value2=Filter[1]
    Value3 = round(1 - Filter[1] - Filter[0], 1)
    num_train = y_train.shape[0]
    # 【1】是特征 【0】是训练集/测试集标签 [2]是训练集标签
    # x_train=FeaCon2(x_train,shallowf[0])
    # svmtool(x_train,shallowf)
    # Data_type = int
    deepf_train_label = np.array(x_train[0])
    deepf_train = np.array(x_train[1])
    shallowf_train_label = np.array(shallowf[0])
    shallowf_train = np.array(shallowf[1])
    classifier = LinearSVC(C=cm)
    if num_train == deepf_train.shape[0]:

        train=FeaCon4(deepf_train, shallowf_train_label,shallowf_train, deepf_train_label)

        y_labels1 = svm_train_model(classifier, train, y_train)
        output=y_labels1*Value1+deepf_train_label*Value2+shallowf_train_label*Value3
        # print(shallowf_train)
        # print(shallowf_train.shape)
        return output
    else:
        # 只是级联标签和训练集。
        # 深层特征1，浅层特征2、标签1、标签二。
        deepf_train_label=x_train[2]
        deepf_test_label=x_train[0]
        shallowf_train_label = shallowf[2]
        shallowf_test_label=shallowf[0]
        shallowf_train_label = np.array(shallowf_train_label)
        shallowf_train=np.array(shallowf_train)
        shallowf_test_label=np.array(shallowf_test_label)
        # print(shallowf_train.shape)
        # print("aaaaa")
        # print(shallowf_train[0:num_train, :])
        x_train1 = FeaCon4(deepf_train[0:num_train, :],shallowf_train[0:num_train, :],deepf_train_label,shallowf_train_label)
        x_test1 = FeaCon4(deepf_train[num_train:deepf_train.shape[0], :],shallowf_train[num_train:deepf_train.shape[0], :],deepf_test_label,shallowf_test_label)
        y_labels1 = test_function_svm(classifier, x_train1, y_train, x_test1)
        return y_labels1*Value1+deepf_test_label*Value2+shallowf_test_label*Value3

        # 训练集有






def Fusion_LR(shallowf, x_train, cm, y_train,Filter):
    Value1=Filter[0]
    Value2=Filter[1]
    Value3 = round(1 - Filter[1] - Filter[0], 1)
    num_train = y_train.shape[0]
    # 【1】是特征 【0】是训练集/测试集标签 [2]是训练集标签
    # x_train=FeaCon2(x_train,shallowf[0])
    # svmtool(x_train,shallowf)
    Data_type = int
    deepf_train_label = np.array(x_train[0])
    deepf_train = np.array(x_train[1])
    shallowf_train_label = np.array(shallowf[0])
    shallowf_train = np.array(shallowf[1],Data_type)
    classifier = LogisticRegression(C=cm, solver='sag', multi_class= 'auto', max_iter=1000)
    if num_train == deepf_train.shape[0]:
        train=FeaCon4(deepf_train, shallowf_train_label,shallowf_train, deepf_train_label)
        y_labels1 = svm_train_model(classifier, train, y_train)
        output=y_labels1*Value1+deepf_train_label*Value2+shallowf_train_label*Value3
        # print(shallowf_train)
        # print(shallowf_train.shape)
        return output
    else:
        # 只是级联标签和训练集。
        # 深层特征1，浅层特征2、标签1、标签二。
        deepf_train_label=x_train[2]
        deepf_test_label=x_train[0]
        shallowf_train_label = shallowf[2]
        shallowf_test_label=shallowf[0]
        shallowf_train_label = np.array(shallowf_train_label)
        shallowf_train=np.array(shallowf_train)
        shallowf_test_label=np.array(shallowf_test_label)
        # print(shallowf_train.shape)
        # print("aaaaa")
        # print(shallowf_train[0:num_train, :])
        x_train1 = FeaCon4(deepf_train[0:num_train, :],shallowf_train[0:num_train, :],deepf_train_label,shallowf_train_label)
        x_test1 = FeaCon4(deepf_train[num_train:deepf_train.shape[0], :],shallowf_train[num_train:deepf_train.shape[0], :],deepf_test_label,shallowf_test_label)
        y_labels1 = test_function_svm(classifier, x_train1, y_train, x_test1)
        return y_labels1*Value1+deepf_test_label*Value2+shallowf_test_label*Value3
def Fusion_ERF(shallowf ,x_train,n_tree, max_dep,y_train,Filter):
    Value1=Filter[0]
    Value2=Filter[1]
    Value3 = round(1 - Filter[1] - Filter[0], 1)
    num_train = y_train.shape[0]
    # 【1】是特征 【0】是训练集/测试集标签 [2]是训练集标签
    # x_train=FeaCon2(x_train,shallowf[0])
    # svmtool(x_train,shallowf)
    Data_type = int
    deepf_train_label = np.array(x_train[0])
    deepf_train = np.array(x_train[1])
    shallowf_train_label = np.array(shallowf[0])
    shallowf_train = np.array(shallowf[1],Data_type)
    classifier = RandomForestClassifier(n_estimators=n_tree, max_depth=max_dep)
    if num_train == deepf_train.shape[0]:

        train=FeaCon4(deepf_train, shallowf_train_label,shallowf_train, deepf_train_label)
        y_labels1 = svm_train_model(classifier, train, y_train)
        output=y_labels1*Value1+deepf_train_label*Value2+shallowf_train_label*Value3
        # print(shallowf_train)
        # print(shallowf_train.shape)
        return output
    else:
        # 只是级联标签和训练集。
        # 深层特征1，浅层特征2、标签1、标签二。
        deepf_train_label=x_train[2]
        deepf_test_label=x_train[0]
        shallowf_train_label = shallowf[2]
        shallowf_test_label=shallowf[0]
        shallowf_train_label = np.array(shallowf_train_label)
        shallowf_train=np.array(shallowf_train)
        shallowf_test_label=np.array(shallowf_test_label)
        # print(shallowf_train.shape)
        # print("aaaaa")
        # print(shallowf_train[0:num_train, :])
        x_train1 = FeaCon4(deepf_train[0:num_train, :],shallowf_train[0:num_train, :],deepf_train_label,shallowf_train_label)
        x_test1 = FeaCon4(deepf_train[num_train:deepf_train.shape[0], :],shallowf_train[num_train:deepf_train.shape[0], :],deepf_test_label,shallowf_test_label)
        y_labels1 = test_function_svm(classifier, x_train1, y_train, x_test1)
        return y_labels1*Value1+deepf_test_label*Value2+shallowf_test_label*Value3

        # 训练集有


def SVM1(shallowf, x_train, cm, y_train,Filter,int10):
    Value1=Filter[0]
    Value2=Filter[1]
    Value3 = round(1 - Filter[1] - Filter[0], 1)
    num_train = y_train.shape[0]
    # 【1】是特征 【0】是训练集/测试集标签 [2]是训练集标签
    # x_train=FeaCon2(x_train,shallowf[0])
    # svmtool(x_train,shallowf)
    # Data_type = int
    deepf_train_label = np.array(x_train[0])
    deepf_train = np.array(x_train[1])
    shallowf_train_label = np.array(shallowf[0])
    shallowf_train = np.array(shallowf[1])
    classifier = LinearSVC(C=cm)
    if num_train == deepf_train.shape[0]:
        if int10==0:
            train=wsFeaCon3(deepf_train, shallowf_train_label, deepf_train_label)
        elif int10==1:
            train = wsFeaCon3(shallowf_train_label, shallowf_train, deepf_train_label)
        else:
            train = FeaCon4(deepf_train, shallowf_train_label, shallowf_train, deepf_train_label)
        y_labels1 = svm_train_model(classifier, train, y_train)
        output=y_labels1*Value1+deepf_train_label*Value2+shallowf_train_label*Value3
        # print(shallowf_train)
        # print(shallowf_train.shape)
        return output
    else:
        # 只是级联标签和训练集。
        # 深层特征1，浅层特征2、标签1、标签二。
        deepf_train_label=x_train[2]
        deepf_test_label=x_train[0]
        shallowf_train_label = shallowf[2]
        shallowf_test_label=shallowf[0]
        shallowf_train_label = np.array(shallowf_train_label)
        shallowf_train=np.array(shallowf_train)
        shallowf_test_label=np.array(shallowf_test_label)
        # print(shallowf_train.shape)
        # print("aaaaa")
        # print(shallowf_train[0:num_train, :])
        x_train1 = FeaCon4(deepf_train[0:num_train, :],shallowf_train[0:num_train, :],deepf_train_label,shallowf_train_label)
        x_test1 = FeaCon4(deepf_train[num_train:deepf_train.shape[0], :],shallowf_train[num_train:deepf_train.shape[0], :],deepf_test_label,shallowf_test_label)
        y_labels1 = test_function_svm(classifier, x_train1, y_train, x_test1)
        return y_labels1*Value1+deepf_test_label*Value2+shallowf_test_label*Value3

        # 训练集有



def addlayer(y_labels1, y_labels2, y_labels3,d):
    a = round(1 - d[1] - d[0], 1)
    output = y_labels1 * d[0] + y_labels2 * d[1] + y_labels3 * a
    print(y_labels3)
    return output

def addlayer2(y_labels1, y_labels2, y_labels3):
    output = y_labels1+ y_labels2 + y_labels3

    return output





def linear_svm3(x_train, y_train,array,array2,d,x):

    #parameters c
    # 构造特征
    a = round(1 - d[1] - d[0], 1)
    #  0是局部 1是全局 a是构造

    classifier = LinearSVC(C=1)
    num_train = y_train.shape[0]  # num_train = 150
    if num_train == x_train.shape[0] and x_train.shape[1]>20:
            y_labels1 = svm_train_model(classifier, x_train, y_train)
    else:
            y_labels1 = test_function_svm(classifier, x_train[0:num_train, :], y_train,x_train[num_train:x_train.shape[0], :])

    # 全局特征2
    classifier = LinearSVC(C=1)
    num_train = y_train.shape[0]  # num_train = 150
    if num_train == array.shape[0] and array.shape[1]>20:
        y_labels2 = svm_train_model(classifier, array, y_train)
    else:
        y_labels2 = test_function_svm(classifier, array[0:num_train, :], y_train,array[num_train:array.shape[0], :])

    feature2 = numpy.asarray(array2)
    if x==0:
            x_train3 = feature2
            classifier = LinearSVC(C=1)
            num_train = y_train.shape[0]  # num_train = 150
            if num_train == x_train3.shape[0] and array.shape[1] > 10:
                y_labels3 = svm_train_model(classifier, x_train3, y_train)
            else:
                y_labels3 = test_function_svm(classifier, x_train3[0:num_train, :], y_train,
                                              x_train3[num_train:x_train3.shape[0], :])
    elif x==1:
            x_train3 = feature2
            classifier = LogisticRegression(solver='sag', multi_class= 'auto', max_iter=1000)
            num_train = y_train.shape[0]  # num_train = 150
            if num_train == x_train3.shape[0] and array.shape[1] > 10:
                y_labels3 = svm_train_model(classifier, x_train3, y_train)
            else:
                y_labels3 = test_function_svm(classifier, x_train3[0:num_train, :], y_train,
                                              x_train3[num_train:x_train3.shape[0], :])
    else:
            x_train3 = feature2
            classifier = ExtraTreesClassifier(n_estimators=500, max_depth=100)
            num_train = y_train.shape[0]
            if num_train == x_train3.shape[0]and array.shape[1] > 10:
                y_labels3 = train_model_prob(classifier, x_train3, y_train)
            else:
                y_labels3 = test_function_prob(classifier, x_train3[0:num_train, :], y_train,
                                              x_train3[num_train:x_train3.shape[0], :])

    output = y_labels1 * d[0] + y_labels2 * d[1] + y_labels3 * a
    return output


def lr(x_train, y_train,array,d,x):
    #parameters c
    # 构造特征
    featuren = []
    for i in range(x_train.shape[0]):
        feature_vector, af = arr_pad_align(x_train[i, :], array[i, :])
        addf = af + feature_vector
        featuren.append(addf)
    feature3 = numpy.asarray(featuren)
    a = round(1 - d[1] - d[0], 1)
    #  0是局部 1是全局 a是构造

    # 如果构造的特征的参数为0，那么只算局部和全局的一个
    if a==0:
        classifier = LogisticRegression(solver='sag', multi_class= 'auto', max_iter=1000)
        if d[0]>d[1]:
            num_train = y_train.shape[0]  # num_train = 150
            if num_train == x_train.shape[0] and array.shape[1] > 10:

                y_labels1 = svm_train_model(classifier, x_train, y_train)
            else:
                y_labels1 = test_function_svm(classifier, x_train[0:num_train, :], y_train,
                                              x_train[num_train:x_train.shape[0], :])
            output = y_labels1
        else:
            num_train = y_train.shape[0] and array.shape[1] > 10  # num_train = 150
            if num_train == array.shape[0]:
                y_labels2 = svm_train_model(classifier, array, y_train)
            else:
                y_labels2 = test_function_svm(classifier, array[0:num_train, :], y_train,
                                              array[num_train:array.shape[0], :])
            output=y_labels2
        return output
    # 如果局部特征为0，那么只算构造和全局的,但是构造特征必须选择分类器
    elif d[0]==0:
         if d[1]>a:
             classifier = LogisticRegression(solver='sag', multi_class='auto', max_iter=1000)
             num_train = y_train.shape[0]  # num_train = 150
             if num_train == x_train.shape[0] and x_train.shape[1] > 10:
                 y_labels1 = svm_train_model(classifier, x_train, y_train)
             else:
                 y_labels1 = test_function_svm(classifier, x_train[0:num_train, :], y_train,
                                               x_train[num_train:x_train.shape[0], :])
             output=y_labels1
         else:
             if x == 0:
                 x_train3 = feature3
                 classifier = LinearSVC(C=1)
                 num_train = y_train.shape[0]  # num_train = 150
                 if num_train == x_train3.shape[0] and array.shape[1] > 10:
                     y_labels3 = svm_train_model(classifier, x_train3, y_train)
                 else:
                     y_labels3 = test_function_svm(classifier, x_train3[0:num_train, :], y_train,
                                                   x_train3[num_train:x_train3.shape[0], :])
             elif x == 1:
                 x_train3 = feature3
                 classifier = LogisticRegression(solver='sag', multi_class='auto', max_iter=1000)
                 num_train = y_train.shape[0]  # num_train = 150
                 if num_train == x_train3.shape[0] and array.shape[1] > 10:
                     y_labels3 = svm_train_model(classifier, x_train3, y_train)
                 else:
                     y_labels3 = test_function_svm(classifier, x_train3[0:num_train, :], y_train,
                                                   x_train3[num_train:x_train3.shape[0], :])
             else:
                 x_train3 = feature3
                 classifier = ExtraTreesClassifier(n_estimators=500, max_depth=100)
                 num_train = y_train.shape[0]
                 if num_train == x_train3.shape[0] and array.shape[1] > 10:
                     y_labels3 = train_model_prob(classifier, x_train3, y_train)
                 else:
                     y_labels3 = test_function_prob(classifier, x_train3[0:num_train, :], y_train,
                                                    x_train3[num_train:x_train3.shape[0], :])
             output=y_labels3
         return output
    # 如果全局特征为0，那么只算构造和全局的,但是构造特征必须选择分类器
    elif d[1]==0:
        if d[0]>a:
            classifier = LogisticRegression(solver='sag', multi_class='auto', max_iter=1000)
            num_train = y_train.shape[0]  # num_train = 150
            if num_train == array.shape[0] and array.shape[1] > 10:
                y_labels2 = svm_train_model(classifier, array, y_train)
            else:
                y_labels2 = test_function_svm(classifier, array[0:num_train, :], y_train,
                                              array[num_train:array.shape[0], :])
            output=y_labels2
        else:
            if x == 0:
                x_train3 = feature3
                classifier = LinearSVC(C=1)
                num_train = y_train.shape[0]  # num_train = 150
                if num_train == x_train3.shape[0] and array.shape[1] > 10:
                    y_labels3 = svm_train_model(classifier, x_train3, y_train)
                else:
                    y_labels3 = test_function_svm(classifier, x_train3[0:num_train, :], y_train,
                                                  x_train3[num_train:x_train3.shape[0], :])
            elif x == 1:
                x_train3 = feature3
                classifier = LogisticRegression(solver='sag', multi_class='auto', max_iter=1000)
                num_train = y_train.shape[0]  # num_train = 150
                if num_train == x_train3.shape[0] and array.shape[1] > 10:
                    y_labels3 = svm_train_model(classifier, x_train3, y_train)
                else:
                    y_labels3 = test_function_svm(classifier, x_train3[0:num_train, :], y_train,
                                                  x_train3[num_train:x_train3.shape[0], :])
            else:
                x_train3 = feature3
                classifier = ExtraTreesClassifier(n_estimators=500, max_depth=100)
                num_train = y_train.shape[0]
                if num_train == x_train3.shape[0] and array.shape[1] > 10:
                    y_labels3 = train_model_prob(classifier, x_train3, y_train)
                else:
                    y_labels3 = test_function_prob(classifier, x_train3[0:num_train, :], y_train,
                                                   x_train3[num_train:x_train3.shape[0], :])
            output = y_labels3
        return output
    else:
        classifier = LogisticRegression(solver='sag', multi_class= 'auto', max_iter=1000)
        num_train = y_train.shape[0]  # num_train = 150
        if num_train == x_train.shape[0] and x_train.shape[1]>20:
            y_labels1 = svm_train_model(classifier, x_train, y_train)
        else:
            y_labels1 = test_function_svm(classifier, x_train[0:num_train, :], y_train,x_train[num_train:x_train.shape[0], :])
        # 全局特征2
        classifier = LogisticRegression(solver='sag', multi_class= 'auto', max_iter=1000)
        num_train = y_train.shape[0]  # num_train = 150
        if num_train == array.shape[0] and array.shape[1]>10:
            y_labels2 = svm_train_model(classifier, array, y_train)
        else:
            print(array.shape)
            y_labels2 = test_function_svm(classifier, array[0:num_train, :], y_train,array[num_train:array.shape[0], :])
        feature1 = []
        for i in range(x_train.shape[0]):
            feature_vector, af = arr_pad_align(x_train[i, :], array[i, :])
            addf = af + feature_vector
            feature1.append(addf)
        feature2 = numpy.asarray(feature1)

        if x == 0:
            x_train3 = feature2
            classifier = LinearSVC(C=1)
            num_train = y_train.shape[0]  # num_train = 150
            if num_train == x_train3.shape[0] and array.shape[1] > 10:
                y_labels3 = svm_train_model(classifier, x_train3, y_train)
            else:
                y_labels3 = test_function_svm(classifier, x_train3[0:num_train, :], y_train,
                                              x_train3[num_train:x_train3.shape[0], :])
        elif x == 1:
            x_train3 = feature2
            classifier = LogisticRegression(solver='sag', multi_class='auto', max_iter=1000)
            num_train = y_train.shape[0]  # num_train = 150
            if num_train == x_train3.shape[0] and array.shape[1] > 10:
                y_labels3 = svm_train_model(classifier, x_train3, y_train)
            else:
                y_labels3 = test_function_svm(classifier, x_train3[0:num_train, :], y_train,
                                              x_train3[num_train:x_train3.shape[0], :])
        else:
            x_train3 = feature2
            classifier = ExtraTreesClassifier(n_estimators=500, max_depth=100)
            num_train = y_train.shape[0]
            if num_train == x_train3.shape[0] and array.shape[1] > 10:
                y_labels3 = train_model_prob(classifier, x_train3, y_train)
            else:
                y_labels3 = test_function_prob(classifier, x_train3[0:num_train, :], y_train,
                                               x_train3[num_train:x_train3.shape[0], :])
        output = y_labels1 * d[0] + y_labels2 * d[1] + y_labels3 * a
        return output

def lr2(x_train, y_train):
    #print(x_train.shape, y_train.shape, num_train, x_train[0:num_train,:].shape, x_train[num_train:-1,:].shape)
    classifier = LogisticRegression(C=1, solver='sag', multi_class= 'auto', max_iter=1000)
    num_train = y_train.shape[0]
    if num_train == x_train.shape[0]:
        y_labels = svm_train_model(classifier, x_train, y_train)
        return [y_labels, x_train]
    else:
        y_labels3 = svm_train_model(classifier, x_train[0:num_train, :], y_train)
        y_labels = test_function_svm(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
        return [y_labels,x_train,y_labels3]

# def random_forest(x_train, y_train):
#     classifier = RandomForestClassifier(n_estimators=500, max_depth=100)
#     num_train = y_train.shape[0]
#     if num_train == x_train.shape[0]:
#         y_labels = train_model_prob(classifier, x_train, y_train)
#         return [y_labels, x_train]
#     else:
#         y_labels3 = train_model_prob(classifier, x_train[0:num_train, :], y_train)
#         y_labels = test_function_prob(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
#         return [y_labels,x_train,y_labels3]]]
def linear_svm(x_train, y_train,cm=0):

    # x_train=np.array(x_train[0])
    c = 10 ** (cm)
    x_train = np.array(x_train)
    #print(x_train.shape, y_train.shape, num_train, x_train[0:num_train,:].shape, x_train[num_train:-1,:].shape)
    classifier = LinearSVC(C=c)
    num_train = y_train.shape[0]
    if num_train == x_train.shape[0]:
        y_labels = svm_train_model(classifier, x_train, y_train)
        return y_labels
    else:

        y_labels = test_function_svm(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])

        return y_labels



def linear_svmtwo(x_train, y_train):
    # x_train=np.array(x_train[0])
    x_train = np.array(x_train)
    classifier = LinearSVC()
    num_train = y_train.shape[0]
    if num_train == x_train.shape[0]:
        y_labels = svm_train_model(classifier, x_train, y_train)
        return y_labels,x_train.shape[1]
    else:
        if x_train.shape[0] < 10:
            return x_train
        y_labels3 = svm_train_model(classifier, x_train[0:num_train, :], y_train)
        y_labels = test_function_svm(classifier, x_train[0:num_train, :], y_train,
                                     x_train[num_train:x_train.shape[0], :])

        zzq = np.concatenate((y_labels, y_labels3), axis=0)
        return y_labels,zzq,x_train.shape[1]
def erandomforesttwo(x_train, y_train):
    #print(x_train.shape, y_train.shape, num_train, x_train[0:num_train,:].shape, x_train[num_train:-1,:].shape)
    x_train = np.array(x_train)
    classifier = ExtraTreesClassifier(n_estimators=500, max_depth=100)
    num_train = y_train.shape[0]
    if num_train == x_train.shape[0]:
        y_labels = train_model_prob(classifier, x_train, y_train)
        return y_labels
    else:
        if x_train.shape[0] < 10:
            return x_train
        y_labels3 = train_model_prob(classifier, x_train[0:num_train, :], y_train)
        y_labels = test_function_prob(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
        zzq = np.concatenate((y_labels, y_labels3), axis=0)
        return y_labels, zzq


def randomforest(x_train, y_train,n_tree,max_dep):

    x_train = np.array(x_train)
    #print(x_train.shape, y_train.shape, num_train, x_train[0:num_train,:].shape, x_train[num_train:-1,:].shape)
    classifier = RandomForestClassifier(n_estimators=n_tree, max_depth=max_dep)
    num_train = y_train.shape[0]
    if num_train == x_train.shape[0]:
        y_labels = train_model_prob(classifier, x_train, y_train)
    else:
        y_labels = test_function_prob(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
    return y_labels

def erandomforest(x_train, y_train,array,d,x):
    # print(x_train.shape, y_train.shape, num_train, x_train[0:num_train,:].shape, x_train[num_train:-1,:].shape)
    # 构造特征
    featuren = []
    for i in range(x_train.shape[0]):
        feature_vector, af = arr_pad_align(x_train[i, :], array[i, :])
        addf = af + feature_vector
        featuren.append(addf)
    feature3 = numpy.asarray(featuren)
    a = round(1 - d[1] - d[0], 1)
    #  0是局部 1是全局 a是构造

    # 如果构造的特征的参数为0，那么只算局部和全局的一个
    if a==0:
        classifier=ExtraTreesClassifier(n_estimators=500, max_depth=100)
        if d[0]>d[1]:
            num_train = y_train.shape[0]  # num_train = 150
            if num_train == x_train.shape[0] and array.shape[1] > 10:

                y_labels1 = train_model_prob(classifier, x_train, y_train)
            else:
                y_labels1 = test_function_prob(classifier, x_train[0:num_train, :], y_train,
                                              x_train[num_train:x_train.shape[0], :])
            output = y_labels1
        else:
            num_train = y_train.shape[0] and array.shape[1] > 10  # num_train = 150
            if num_train == array.shape[0]:
                y_labels2 = train_model_prob(classifier, array, y_train)
            else:
                y_labels2 = test_function_prob(classifier, array[0:num_train, :], y_train,
                                              array[num_train:array.shape[0], :])
            output=y_labels2
        return output
    # 如果局部特征为0，那么只算构造和全局的,但是构造特征必须选择分类器
    elif d[0]==0:
         if d[1]>a:
             classifier=ExtraTreesClassifier(n_estimators=500, max_depth=100)
             num_train = y_train.shape[0]  # num_train = 150
             if num_train == x_train.shape[0] and x_train.shape[1] > 10:
                 y_labels1 = train_model_prob(classifier, x_train, y_train)
             else:
                 y_labels1 = test_function_prob(classifier, x_train[0:num_train, :], y_train,
                                               x_train[num_train:x_train.shape[0], :])
             output=y_labels1
         else:
             if x == 0:
                 x_train3 = feature3
                 classifier = LinearSVC(C=1)
                 num_train = y_train.shape[0]  # num_train = 150
                 if num_train == x_train3.shape[0] and array.shape[1] > 10:
                     y_labels3 = svm_train_model(classifier, x_train3, y_train)
                 else:
                     y_labels3 = test_function_svm(classifier, x_train3[0:num_train, :], y_train,
                                                   x_train3[num_train:x_train3.shape[0], :])
             elif x == 1:
                 x_train3 = feature3
                 classifier = LogisticRegression(solver='sag', multi_class='auto', max_iter=1000)
                 num_train = y_train.shape[0]  # num_train = 150
                 if num_train == x_train3.shape[0] and array.shape[1] > 10:
                     y_labels3 = svm_train_model(classifier, x_train3, y_train)
                 else:
                     y_labels3 = test_function_svm(classifier, x_train3[0:num_train, :], y_train,
                                                   x_train3[num_train:x_train3.shape[0], :])
             else:
                 x_train3 = feature3
                 classifier = ExtraTreesClassifier(n_estimators=500, max_depth=100)
                 num_train = y_train.shape[0]
                 if num_train == x_train3.shape[0] and array.shape[1] > 10:
                     y_labels3 = train_model_prob(classifier, x_train3, y_train)
                 else:
                     y_labels3 = test_function_prob(classifier, x_train3[0:num_train, :], y_train,
                                                    x_train3[num_train:x_train3.shape[0], :])
             output=y_labels3
         return output
         # 如果全局特征为0，那么只算构造和全局的,但是构造特征必须选择分类器
    elif d[1] == 0:
        if d[0] > a:
            classifier = ExtraTreesClassifier(n_estimators=500, max_depth=100)
            num_train = y_train.shape[0]  # num_train = 150
            if num_train == array.shape[0] and array.shape[1] > 10:
                y_labels2 = train_model_prob(classifier, array, y_train)
            else:
                y_labels2 = test_function_prob(classifier, array[0:num_train, :], y_train,
                                               array[num_train:array.shape[0], :])
            output = y_labels2
        else:
            if x == 0:
                x_train3 = feature3
                classifier = LinearSVC(C=1)
                num_train = y_train.shape[0]  # num_train = 150
                if num_train == x_train3.shape[0] and array.shape[1] > 10:
                    y_labels3 = svm_train_model(classifier, x_train3, y_train)
                else:
                    y_labels3 = test_function_svm(classifier, x_train3[0:num_train, :], y_train,
                                                  x_train3[num_train:x_train3.shape[0], :])
            elif x == 1:
                x_train3 = feature3
                classifier = LogisticRegression(solver='sag', multi_class='auto', max_iter=1000)
                num_train = y_train.shape[0]  # num_train = 150
                if num_train == x_train3.shape[0] and array.shape[1] > 10:
                    y_labels3 = svm_train_model(classifier, x_train3, y_train)
                else:
                    y_labels3 = test_function_svm(classifier, x_train3[0:num_train, :], y_train,
                                                  x_train3[num_train:x_train3.shape[0], :])
            else:
                x_train3 = feature3
                classifier = ExtraTreesClassifier(n_estimators=500, max_depth=100)
                num_train = y_train.shape[0]
                if num_train == x_train3.shape[0] and array.shape[1] > 10:
                    y_labels3 = train_model_prob(classifier, x_train3, y_train)
                else:
                    y_labels3 = test_function_prob(classifier, x_train3[0:num_train, :], y_train,
                                                   x_train3[num_train:x_train3.shape[0], :])
            output = y_labels3
        return output
    else:
        classifier=ExtraTreesClassifier(n_estimators=500, max_depth=100)
        num_train = y_train.shape[0]  # num_train = 150
        if num_train == x_train.shape[0] and x_train.shape[1] > 10:
            y_labels1 = train_model_prob(classifier, x_train, y_train)
        else:
            y_labels1 = test_function_prob(classifier, x_train[0:num_train, :], y_train,
                                          x_train[num_train:x_train.shape[0], :])
        # 全局特征2
        classifier=ExtraTreesClassifier(n_estimators=500, max_depth=100)
        num_train = y_train.shape[0]  # num_train = 150
        if num_train == array.shape[0] and array.shape[1] > 10:
            y_labels2 = train_model_prob(classifier, array, y_train)
        else:
            y_labels2 = test_function_prob(classifier, array[0:num_train, :], y_train,
                                          array[num_train:array.shape[0], :])
        feature1 = []
        for i in range(x_train.shape[0]):
            feature_vector, af = arr_pad_align(x_train[i, :], array[i, :])
            addf = af + feature_vector
            feature1.append(addf)
        feature2 = numpy.asarray(feature1)
        if x==0:
            x_train3 = feature2
            classifier = LinearSVC(C=1)
            num_train = y_train.shape[0]  # num_train = 150
            if num_train == x_train3.shape[0] and array.shape[1] > 10:
                y_labels3 = svm_train_model(classifier, x_train3, y_train)
            else:
                y_labels3 = test_function_svm(classifier, x_train3[0:num_train, :], y_train,
                                              x_train3[num_train:x_train3.shape[0], :])
        elif x==1:
            x_train3 = feature2
            classifier = LogisticRegression(solver='sag', multi_class= 'auto', max_iter=1000)
            num_train = y_train.shape[0]  # num_train = 150
            if num_train == x_train3.shape[0] and array.shape[1] > 10:
                y_labels3 = svm_train_model(classifier, x_train3, y_train)
            else:
                y_labels3 = test_function_svm(classifier, x_train3[0:num_train, :], y_train,
                                              x_train3[num_train:x_train3.shape[0], :])
        else:
            x_train3 = feature2
            classifier = ExtraTreesClassifier(n_estimators=500, max_depth=100)
            num_train = y_train.shape[0]
            if num_train == x_train3.shape[0]and array.shape[1] > 10:
                y_labels3 = train_model_prob(classifier, x_train3, y_train)
            else:
                y_labels3 = test_function_prob(classifier, x_train3[0:num_train, :], y_train,
                                              x_train3[num_train:x_train3.shape[0], :])

        output = y_labels1 * d[0] + y_labels2 * d[1] + y_labels3 * a
        return output

def erandomforest2(x_train, y_train,array,d,x=2):
    # print(x_train.shape, y_train.shape, num_train, x_train[0:num_train,:].shape, x_train[num_train:-1,:].shape)
    # 构造特征
    featuren = []
    for i in range(x_train.shape[0]):
        feature_vector, af = arr_pad_align(x_train[i, :], array[i, :])
        addf = af + feature_vector
        featuren.append(addf)
    feature3 = numpy.asarray(featuren)
    a = round(1 - d[1] - d[0], 1)
    #  0是局部 1是全局 a是构造

    # 如果构造的特征的参数为0，那么只算局部和全局的一个
    if a==0:
        classifier=ExtraTreesClassifier(n_estimators=500, max_depth=100)
        if d[0]>d[1]:
            num_train = y_train.shape[0]  # num_train = 150
            if num_train == x_train.shape[0] and array.shape[1] > 10:

                y_labels1 = train_model_prob(classifier, x_train, y_train)
            else:
                y_labels1 = test_function_prob(classifier, x_train[0:num_train, :], y_train,
                                              x_train[num_train:x_train.shape[0], :])
            output = y_labels1
        else:
            num_train = y_train.shape[0] and array.shape[1] > 10  # num_train = 150
            if num_train == array.shape[0]:
                y_labels2 = train_model_prob(classifier, array, y_train)
            else:
                y_labels2 = test_function_prob(classifier, array[0:num_train, :], y_train,
                                              array[num_train:array.shape[0], :])
            output=y_labels2
        return output
    # 如果局部特征为0，那么只算构造和全局的,但是构造特征必须选择分类器
    elif d[0]==0:
         if d[1]>a:
             classifier=ExtraTreesClassifier(n_estimators=500, max_depth=100)
             num_train = y_train.shape[0]  # num_train = 150
             if num_train == x_train.shape[0] and x_train.shape[1] > 10:
                 y_labels1 = train_model_prob(classifier, x_train, y_train)
             else:
                 y_labels1 = test_function_prob(classifier, x_train[0:num_train, :], y_train,
                                               x_train[num_train:x_train.shape[0], :])
             output=y_labels1
         else:
             if x == 0:
                 x_train3 = feature3
                 classifier = LinearSVC(C=1)
                 num_train = y_train.shape[0]  # num_train = 150
                 if num_train == x_train3.shape[0] and array.shape[1] > 10:
                     y_labels3 = svm_train_model(classifier, x_train3, y_train)
                 else:
                     y_labels3 = test_function_svm(classifier, x_train3[0:num_train, :], y_train,
                                                   x_train3[num_train:x_train3.shape[0], :])
             elif x == 1:
                 x_train3 = feature3
                 classifier = LogisticRegression(solver='sag', multi_class='auto', max_iter=1000)
                 num_train = y_train.shape[0]  # num_train = 150
                 if num_train == x_train3.shape[0] and array.shape[1] > 10:
                     y_labels3 = svm_train_model(classifier, x_train3, y_train)
                 else:
                     y_labels3 = test_function_svm(classifier, x_train3[0:num_train, :], y_train,
                                                   x_train3[num_train:x_train3.shape[0], :])
             else:
                 x_train3 = feature3
                 classifier = ExtraTreesClassifier(n_estimators=500, max_depth=100)
                 num_train = y_train.shape[0]
                 if num_train == x_train3.shape[0] and array.shape[1] > 10:
                     y_labels3 = train_model_prob(classifier, x_train3, y_train)
                 else:
                     y_labels3 = test_function_prob(classifier, x_train3[0:num_train, :], y_train,
                                                    x_train3[num_train:x_train3.shape[0], :])
             output=y_labels3
         return output
         # 如果全局特征为0，那么只算构造和全局的,但是构造特征必须选择分类器
    elif d[1] == 0:
        if d[0] > a:
            classifier = ExtraTreesClassifier(n_estimators=500, max_depth=100)
            num_train = y_train.shape[0]  # num_train = 150
            if num_train == array.shape[0] and array.shape[1] > 10:
                y_labels2 = train_model_prob(classifier, array, y_train)
            else:
                y_labels2 = test_function_prob(classifier, array[0:num_train, :], y_train,
                                               array[num_train:array.shape[0], :])
            output = y_labels2
        else:
            if x == 0:
                x_train3 = feature3
                classifier = LinearSVC(C=1)
                num_train = y_train.shape[0]  # num_train = 150
                if num_train == x_train3.shape[0] and array.shape[1] > 10:
                    y_labels3 = svm_train_model(classifier, x_train3, y_train)
                else:
                    y_labels3 = test_function_svm(classifier, x_train3[0:num_train, :], y_train,
                                                  x_train3[num_train:x_train3.shape[0], :])
            elif x == 1:
                x_train3 = feature3
                classifier = LogisticRegression(solver='sag', multi_class='auto', max_iter=1000)
                num_train = y_train.shape[0]  # num_train = 150
                if num_train == x_train3.shape[0] and array.shape[1] > 10:
                    y_labels3 = svm_train_model(classifier, x_train3, y_train)
                else:
                    y_labels3 = test_function_svm(classifier, x_train3[0:num_train, :], y_train,
                                                  x_train3[num_train:x_train3.shape[0], :])
            else:
                x_train3 = feature3
                classifier = ExtraTreesClassifier(n_estimators=500, max_depth=100)
                num_train = y_train.shape[0]
                if num_train == x_train3.shape[0] and array.shape[1] > 10:
                    y_labels3 = train_model_prob(classifier, x_train3, y_train)
                else:
                    y_labels3 = test_function_prob(classifier, x_train3[0:num_train, :], y_train,
                                                   x_train3[num_train:x_train3.shape[0], :])
            output = y_labels3
        return output
    else:
        classifier=ExtraTreesClassifier(n_estimators=500, max_depth=100)
        num_train = y_train.shape[0]  # num_train = 150
        if num_train == x_train.shape[0] and x_train.shape[1] > 10:
            y_labels1 = train_model_prob(classifier, x_train, y_train)
        else:
            y_labels1 = test_function_prob(classifier, x_train[0:num_train, :], y_train,
                                          x_train[num_train:x_train.shape[0], :])
        # 全局特征2
        classifier=ExtraTreesClassifier(n_estimators=500, max_depth=100)
        num_train = y_train.shape[0]  # num_train = 150
        if num_train == array.shape[0] and array.shape[1] > 10:
            y_labels2 = train_model_prob(classifier, array, y_train)
        else:
            y_labels2 = test_function_prob(classifier, array[0:num_train, :], y_train,
                                          array[num_train:array.shape[0], :])
        feature1 = []
        for i in range(x_train.shape[0]):
            feature_vector, af = arr_pad_align(x_train[i, :], array[i, :])
            addf = af + feature_vector
            feature1.append(addf)
        feature2 = numpy.asarray(feature1)
        if x==0:
            x_train3 = feature2
            classifier = LinearSVC(C=1)
            num_train = y_train.shape[0]  # num_train = 150
            if num_train == x_train3.shape[0] and array.shape[1] > 10:
                y_labels3 = svm_train_model(classifier, x_train3, y_train)
            else:
                y_labels3 = test_function_svm(classifier, x_train3[0:num_train, :], y_train,
                                              x_train3[num_train:x_train3.shape[0], :])
        elif x==1:
            x_train3 = feature2
            classifier = LogisticRegression(solver='sag', multi_class= 'auto', max_iter=1000)
            num_train = y_train.shape[0]  # num_train = 150
            if num_train == x_train3.shape[0] and array.shape[1] > 10:
                y_labels3 = svm_train_model(classifier, x_train3, y_train)
            else:
                y_labels3 = test_function_svm(classifier, x_train3[0:num_train, :], y_train,
                                              x_train3[num_train:x_train3.shape[0], :])
        else:
            x_train3 = feature2
            classifier = ExtraTreesClassifier(n_estimators=500, max_depth=100)
            num_train = y_train.shape[0]
            if num_train == x_train3.shape[0]and array.shape[1] > 10:
                y_labels3 = train_model_prob(classifier, x_train3, y_train)
            else:
                y_labels3 = test_function_prob(classifier, x_train3[0:num_train, :], y_train,
                                              x_train3[num_train:x_train3.shape[0], :])

        output = y_labels1 * d[0] + y_labels2 * d[1] + y_labels3 * a
        return output

def erandomforest3(x_train, y_train,n_tree,max_dep):
    #print(x_train.shape, y_train.shape, num_train, x_train[0:num_train,:].shape, x_train[num_train:-1,:].shape)
    classifier = ExtraTreesClassifier(n_estimators=n_tree, max_depth=max_dep)
    num_train = y_train.shape[0]
    if num_train == x_train.shape[0]:
        y_labels = train_model_prob(classifier, x_train, y_train)
    else:
        y_labels = test_function_prob(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
    return y_labels,x_train

def erandomforest4(x_train, y_train):
    #print(x_train.shape, y_train.shape, num_train, x_train[0:num_train,:].shape, x_train[num_train:-1,:].shape)
    classifier = ExtraTreesClassifier(n_estimators=500, max_depth=100)
    num_train = y_train.shape[0]
    if num_train == x_train.shape[0]:
        y_labels = train_model_prob(classifier, x_train, y_train)
        return [y_labels, x_train]
    else:
        y_labels3 = train_model_prob(classifier, x_train[0:num_train, :], y_train)
        y_labels = test_function_prob(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
        return [y_labels,x_train,y_labels3]



def svm_train_model(model, x, y, k=5):
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(np.asarray(x))
    kf = StratifiedKFold(n_splits=k)
    ni = np.unique(y)
    num_class = ni.shape[0]
    y_predict = np.zeros((len(y), num_class))
    for train_index, test_index in kf.split(x,y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        y_label = []
        for i in y_pred:
            binary_label = np.zeros((num_class))
            binary_label[int(i)] = 1
            y_label.append(binary_label)
        y_predict[test_index,:] = np.asarray(y_label)
    return y_predict

def wsFeaCon3(img1, img2, img3):
    x_features = []

    for i in range(img1.shape[0]):
        image1 = conVector(img1[i, :])
        image2 = conVector(img2[i, :])
        image3 = conVector(img3[i, :])
        feature_vector = numpy.concatenate((image1, image2, image3), axis=0)
        x_features.append(feature_vector)
    return numpy.asarray(x_features)

def test_function_svm(model, x_train, y_train, x_test):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_train = min_max_scaler.fit_transform(np.asarray(x_train))
    x_test = min_max_scaler.transform(np.asarray(x_test))
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_label = []
    ni = np.unique(y_train)
    num_class = ni.shape[0]
    for i in y_pred:
        binary_label = np.zeros((num_class))
        binary_label[int(i)] = 1
        y_label.append(binary_label)
    y_predict = np.asarray(y_label)
    return y_predict

def train_model_prob(model, x, y, k=5):
##    min_max_scaler = preprocessing.MinMaxScaler()
##    x = min_max_scaler.fit_transform(np.asarray(x))
    kf = StratifiedKFold(n_splits=k)
    ni = np.unique(y)
    num_class = ni.shape[0]
    y_predict = np.zeros((len(y), num_class))
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train,y_train)
        y_predict[test_index,:] = model.predict_proba(x_test)
    return y_predict

def test_function_prob(model, x_train, y_train, x_test):
    ##min_max_scaler = preprocessing.MinMaxScaler()
    #x_train = min_max_scaler.fit_transform(np.asarray(x_train))
    #x_test = min_max_scaler.transform(np.asarray(x_test))
    model.fit(x_train, y_train)
    y_pred = model.predict_proba(x_test)
    return y_pred

def conVector(img):
    try:
        img_vector=numpy.concatenate((img))
    except:
        img_vector=img
    return img_vector

def FeaCon2(img1, img2):
    x_features = []
    class_labal=img1[1]
    class_labal2 = img2[1]
    a1=img1
    b1=img2
    img1 = np.array(img1)
    img2 = np.array(img2)
    if type(a1)==tuple:
        img1=class_labal
    elif type(b1)==tuple:
        img2=class_labal2
    for i in range(img1.shape[0]):
        image1 = conVector(img1[i, :])
        image2 = conVector(img2[i, :])
        feature_vector = numpy.concatenate((image1, image2), axis=0)
        x_features.append(feature_vector)
    return numpy.asarray(x_features)


def N_FeaCon3(img1, img2, img3):
    x_features = []
    class_labal=img1[1]
    class_labal2 = img2[1]
    class_labal3 = img3[1]
    a1=img1
    b1=img2
    c1=img3
    img1 = np.array(img1)
    img2 = np.array(img2)
    img3 = np.array(img3)
    if type(a1)==tuple:
        img1=class_labal
    elif type(b1)==tuple:
        img2=class_labal2
    elif type(c1)==tuple:
        img3=class_labal3
    for i in range(img1.shape[0]):
        image1 = conVector(img1[i, :])
        image2 = conVector(img2[i, :])
        image3 = conVector(img3[i, :])
        feature_vector = numpy.concatenate((image1, image2, image3), axis=0)
        x_features.append(feature_vector)
    return numpy.asarray(x_features)

def FeaConnew(img1, img2):
    x_features = []
    for i in range(img1.shape[0]):
        image1 = conVector(img1[i, :])
        image2 = conVector(img2[i, :])
        feature_vector = numpy.concatenate((image1, image2), axis=0)
        x_features.append(feature_vector)
    return numpy.asarray(x_features)


def FeaConsvm2(img1, img2):
    x_features = []
    for i in range(img1.shape[0]):
        image1 = conVector(img1[i, :])
        image2 = conVector(img2[i, :])
        feature_vector = numpy.concatenate((image1, image2), axis=0)
        x_features.append(feature_vector)
    return numpy.asarray(x_features)
def pcaa(image):
    pca = PCA(n_components=1)
    newf=pca.fit(image)
    return newf


def pcaFeaCon2(img1, img2):
    x_features = []
    f=FeaCon2(img1[0],img2[0])
    for i in range(img1.shape[0]):
        x_features.append(pcaa(f[i, :]))
    return x_features

def newFeaCon2(img1, img2):
    f=FeaCon2(img1[0],img2[0])
    return f
def newFeaCon3(img1, img2,img3):
    f=FeaCon3(img1[0],img2[0],img3[0])
    return f
def newFeaCon4(img1, img2,img3,img4):
    f=FeaCon4(img1[0],img2[0],img3[0],img4[0])
    return f
def newFeaCon(img1,a):
    if a==0:
        return img1[0]
    else:
        try:
            c=FeaCon2(img1[0],img1[2])
        except:
            c=img1[0]
        return c
def newFeaCon1(img1):
    return img1[0]


def shallowF(img1):
    return img1


def newF(img1):
    f=img1[0]
    return f

def addCon2(img1, img2):
    feature1 = []
    for i in range(img1.shape[0]):
        feature_vector, af = arr_pad_align(img1[i,:],img2[i,:])
        addf = af + feature_vector
        feature1.append(addf)
    return numpy.asarray(feature1)
def addCon3(img1, img2):
    feature1 = []
    for i in range(img1.shape[0]):
        feature_vector, af = arr_pad_align(img1[i],img2[i])
        addf = af + feature_vector
        feature1.append(addf)
    return numpy.asarray(feature1)

def GFeaCon2(img1, img2):
    x_features = []
    for i in range(img1.shape[0]):
        image1 = conVector(img1[i, :])
        image2 = conVector(img2[i, :])
        feature_vector = numpy.concatenate((image1, image2), axis=0)
        x_features.append(feature_vector)
    return numpy.asarray(x_features)
def GGFeaCon2(img1, img2):
    x_features = []
    for i in range(img1.shape[0]):
        image1 = conVector(img1[i, :])
        image2 = conVector(img2[i, :])
        feature_vector = numpy.concatenate((image1, image2), axis=0)
        x_features.append(feature_vector)
    return numpy.asarray(x_features)

def FeaCon3(img1, img2, img3,array):
    f1value=array[0]
    f2value = array[1]
    f3value=1-f1value-f2value
    x_features = []
    x1_features = []
    x2_features = []
    for i in range(img1.shape[0]):
        image1 = img1[i]*f1value
        image2 = img2[i]*f2value
        image3 = img3[i]*f3value
        x_features.append(image1)
        x1_features.append(image2)
        x2_features.append(image3)

    x_features=numpy.asarray(x_features)
    x1_features = numpy.asarray(x1_features)
    x2_features = numpy.asarray(x2_features)
    a=addCon2(x_features,x1_features)
    b=addCon2(a,x2_features)

    return numpy.asarray(b)

def n2FeaCon3(img1, img2, img3):
    x_features = []
    for i in range(img1.shape[0]):
        image1 = conVector(img1[i, :])
        image2 = conVector(img2[i, :])
        image3 = conVector(img3[i, :])
        feature_vector = numpy.concatenate((image1, image2, image3), axis=0)
        x_features.append(feature_vector)
    return numpy.asarray(x_features)

def newFeaCon3(img1, img2, img3):
    img1=img1[0]
    img2=img2[0]
    img3=img3[0]
    x_features = []
    for i in range(img1.shape[0]):
        image1 = conVector(img1[i, :])
        image2 = conVector(img2[i, :])
        image3 = conVector(img3[i, :])
        feature_vector = numpy.concatenate((image1, image2, image3), axis=0)
        x_features.append(feature_vector)
    return numpy.asarray(x_features)


def FeaCon4(img1, img2, img3, img4):
    x_features = []

    for i in range(img1.shape[0]):
        image1 = conVector(img1[i, :])
        image2 = conVector(img2[i, :])
        image3 = conVector(img3[i, :])
        image4 = conVector(img4[i, :])
        feature_vector = numpy.concatenate((image1, image2, image3, image4), axis=0)
        x_features.append(feature_vector)
    return numpy.asarray(x_features)

def histLBP(image,radius,n_points):
    # 'uniform','default','ror','var'
    lbp = local_binary_pattern(image, n_points, radius, method='nri_uniform')
    n_bins = 59
    hist,ax=numpy.histogram(lbp,n_bins,[0,59])
    return hist

def all_lbp(image):
    # global and local
    feature = []
    for i in range(image.shape[0]):
        feature_vector = histLBP(image[i,:,:], radius=1.5, n_points=8)
        feature.append(feature_vector)
    return numpy.asarray(feature)
def all_lbp1(image):
    image=original2(image)
    # global and local
    feature = []
    for i in range(image.shape[0]):
        feature_vector = histLBP(image[i,:,:], radius=1.5, n_points=8)
        feature.append(feature_vector)
    return numpy.asarray(feature)

#
def HoGFeatures(image):
    try:
        img,realImage=hog(image,orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(3, 3), block_norm='L2-Hys', visualize=True,
                    transform_sqrt=False, feature_vector=True)
        return realImage
    except:
        return image


def hog_features_patches(image,patch_size,moving_size):
    img=numpy.asarray(image)
    width, height = img.shape
    w = int(width / moving_size)
    h = int(height / moving_size)
    patch = []
    for i in range(0, w):
        for j in range(0, h):
            patch.append([moving_size * i, moving_size * j])
    hog_features = numpy.zeros((len(patch)))
    realImage=HoGFeatures(img)
    for i in range(len(patch)):
        hog_features[i] = numpy.mean(
            realImage[patch[i][0]:(patch[i][0] + patch_size), patch[i][1]:(patch[i][1] + patch_size)])
    return hog_features

def global_hog_small(image):
    feature = []
    for i in range(image.shape[0]):
        feature_vector = hog_features_patches(image[i,:,:],20,10)
        feature.append(feature_vector)
    return numpy.asarray(feature)
def global_hog_small1(image):
    image=original2(image)
    feature = []
    for i in range(image.shape[0]):
        feature_vector = hog_features_patches(image[i,:,:],2,2)
        feature.append(feature_vector)
    return numpy.asarray(feature)

def CONVA(image):
    nf=Prewitt(image)
    f1 = conva(nf)
    return f1
def CONVA2(image):
    nf=Prewitt(image)
    f1 = conva(nf)
    return f1

def MAXP_F(image):
    maxPf = maxP(image, 2, 4)
    f1 = conva(maxPf)
    return f1



def Sobel_F(image):
    eqlf = eql(image)
    f1 = conva(eqlf)
    return f1

def eqlf(image):
    eqlf = eql(image)
    f1 = conva(eqlf)
    return numpy.asarray(f1)

def sobelf(image):
    sobelf = sobelxy(image)
    f1 = conva(sobelf)
    return numpy.asarray(f1)

def cannynf(image):
    cannyf1 = cannyf(image)
    f1 = conva(cannyf1)
    return numpy.asarray(f1)

def lobal_hog_small(image):
    feature = []
    for i in range(image.shape[0]):
        feature_vector = hog_features_patches(image[i,:,:], 2, 2)
        feature.append(feature_vector)
    return numpy.asarray(feature)

def add_hog_small(image):
    feature = []
    for i in range(image.shape[0]):
        feature_vector = hog_features_patches(image[i,:,:], 10, 10)
        feature.append(feature_vector)
    return numpy.asarray(feature)

def lobal_hog_small2(image):
    feature = []
    for i in range(image.shape[0]):
        feature_vector = hog_features_patches(image[i,:,:], 2, 2)
        feature.append(feature_vector)
    return numpy.asarray(feature)

def conva(image):
    feature1 = []
    for i in range(image.shape[0]):
        img_vector =conVector(image[i,:,:])
        feature1.append(img_vector)
    return numpy.asarray(feature1)

def all_sift(image):
    width, height = image[0, :, :].shape
    min_length = numpy.min((width, height))
    feature = []
    for i in range(image.shape[0]):
        img = numpy.asarray(image[i, 0:width, 0:height])
        extractor = sift_features.SingleSiftExtractor(min_length)
        feaArrSingle = extractor.process_image(img[0:min_length, 0:min_length])
        # dimension 128 for all images
        w, h = feaArrSingle.shape
        feature_vector = numpy.reshape(feaArrSingle, (h,))
        feature.append(feature_vector)
    return numpy.asarray(feature)
import cv2
import numpy as np

import cv2
import numpy as np





def sift_conva(image):
    a=0

    width, height = image[0, :, :].shape
    min_length = numpy.min((width, height))
    feature = []
    for i in range(image.shape[0]):
        img = numpy.asarray(image[i, 0:width, 0:height])
        extractor = sift_features.SingleSiftExtractor(min_length)
        feaArrSingle = extractor.process_image(img[0:min_length, 0:min_length])
        # dimension 128 for all images
        w, h = feaArrSingle.shape
        feature_vector = numpy.reshape(feaArrSingle, (h,))
        feature.append(feature_vector)
    f2=conva(image)
    new_f=numpy.asarray(feature)
    feature=FeaCon2(new_f,f2)
    return numpy.asarray(feature),image,a




def sift_conva1(image):
    a=0
    image1=image[1]
    image=image[0]
    width, height = image[0, :, :].shape
    min_length = numpy.min((width, height))
    feature = []
    for i in range(image.shape[0]):
        img = numpy.asarray(image[i, 0:width, 0:height])
        extractor = sift_features.SingleSiftExtractor(min_length)
        feaArrSingle = extractor.process_image(img[0:min_length, 0:min_length])
        # dimension 128 for all images
        w, h = feaArrSingle.shape
        feature_vector = numpy.reshape(feaArrSingle, (h,))
        feature.append(feature_vector)
    f2=conva(image1)
    new_f=numpy.asarray(feature)
    feature=FeaCon2(new_f,f2)

    return numpy.asarray(feature),image,a


# def hog_conva(image):
#     feature = []
#     for i in range(image.shape[0]):
#         feature_vector = hog_features_patches(image[i,:,:], 10, 10)
#         feature.append(feature_vector)
#     f2=conva(image)
#     new_f=numpy.asarray(feature)
#     feature=FeaCon2(new_f,f2)
#     return numpy.asarray(feature)

def hog_conva(image):
    a=0
    feature = []
    for i in range(image.shape[0]):
        feature_vector = hog_features_patches(image[i,:,:], 10, 10)
        feature.append(feature_vector)
    f2=conva(image)
    new_f=numpy.asarray(feature)
    feature=FeaCon2(new_f,f2)
    return numpy.asarray(feature),image,a

def hog_conva1(image):
    a=0
    image1 = image[1]
    image = image[0]
    feature = []
    for i in range(image.shape[0]):
        feature_vector = hog_features_patches(image[i,:,:], 10, 10)
        feature.append(feature_vector)
    f2=conva(image1)
    new_f=numpy.asarray(feature)
    feature=FeaCon2(new_f,f2)
    return numpy.asarray(feature),image,a


def hist_conva(left):
    img = []
    for i in range(left.shape[0]):
        img.append(all_histogram(left[i, :, :]))
    f2=conva(left)
    new_f=numpy.asarray(img)
    feature=FeaCon2(new_f,f2)
    return numpy.asarray(feature)

def lbp_conva(image):
    a=0
    # global and local
    feature = []
    for i in range(image.shape[0]):
        feature_vector = histLBP(image[i,:,:], radius=1.5, n_points=8)
        feature.append(feature_vector)
    f2=conva(image)
    new_f=numpy.asarray(feature)
    feature=FeaCon2(new_f,f2)
    return numpy.asarray(feature),image,a

def lbp_conva1(image):
    a=0
    # global and local
    image1=image[1]
    image=image[0]
    feature = []
    for i in range(image.shape[0]):
        feature_vector = histLBP(image[i,:,:], radius=1.5, n_points=8)
        feature.append(feature_vector)
    f2=conva(image1)
    new_f=numpy.asarray(feature)
    feature=FeaCon2(new_f,f2)
    return numpy.asarray(feature),image,a


def all_sift1(image):
    image=original2(image)
    width, height = image[0, :, :].shape
    min_length = numpy.min((width, height))
    feature = []
    for i in range(image.shape[0]):
        img = numpy.asarray(image[i, 0:width, 0:height])
        extractor = sift_features.SingleSiftExtractor(min_length)
        feaArrSingle = extractor.process_image(img[0:min_length, 0:min_length])
        # dimension 128 for all images
        w, h = feaArrSingle.shape
        feature_vector = numpy.reshape(feaArrSingle, (h,))
        feature.append(feature_vector)
    return numpy.asarray(feature)

def all_conva(image,K1,K2):
    image=maxP(image,K1,K2)
    new_ff=conva(image)
    return numpy.asarray(new_ff)

def twovec(image):
    a=all_sift(image)
    d=conVector(image)
    c=FeaCon2(a,d)

    # c = addCon2(a, b)
    return c

min_max_scaler = preprocessing.MinMaxScaler()
def featuret(image,m,i):
    i=round((i*0.1),2)
    i2=round((1-i),2)
    # img1是已有特征
    img1=image[0]
    # img2是像素点
    img2=image[1]
    if m==0:
        f1=all_sift(img2)
    elif m==1:
        f1=all_lbp(img2)
    else:
        f1=add_hog_small(img2)
    f1=min_max_scaler.fit_transform(f1)
    f1=f1*i
    newf=addCon2(f1,img1*i2)
    return newf,img2

def cannyf(left):
    img = []
    for i in range(left.shape[0]):
        img.append(canny(left[i, :, :]))
    return np.asarray(img)

def featuret2(image,m):
    # img1是已有特征
    img1=image[0]
    # img2是像素点
    img2=image[1]
    if m==0:
        f1=all_sift(img2)
    elif m==1:
        f1=all_lbp(img2)
    else:
        f1=lobal_hog_small(img2)
    f1 = min_max_scaler.fit_transform(f1)
    newf=FeaCon2(f1,img1)
    return newf,img2

def cannyf(left):
    img = []
    for i in range(left.shape[0]):
        img.append(canny(left[i, :, :]))
    return np.asarray(img)



def G_feature(image,m):
    if m==0:
        f1=all_sift(image)
    elif m==1:
        f1=all_lbp(image)
    elif m==2:
        eqlf=eql(image)
        f1=conva(eqlf)
    elif m==3:
        sobelf=sobelxy(image)
        f1=conva(sobelf)
    elif m==4:
        canny=cannyf(image)
        f1=conva(canny)
    else:
        f1=lobal_hog_small(image)
    f1=min_max_scaler.fit_transform(f1)
    return f1,image


# def L_feature(image,m):
#     if m==0:
#         f1=all_sift(image)
#     elif m==1:
#         f1=all_lbp(image)
#     else:
#          f1=lobal_hog_small(image)
#     f1=min_max_scaler.fit_transform(f1)
#     return f1,image

def new_FCA2(image, m, i):
    x=image[2]
    i = round((i * 0.1), 2)
    i2 = round((1 - i), 2)
    # img1是已有特征
    img1 = image[0]
    # img2是像素点
    img2 = image[1]
    # concaf是像素点是一般特征
    if m == 0:
        f1 = all_sift(img2)
    elif m == 1:
        f1 = all_lbp(img2)
    else:
        f1 = add_hog_small(img2)
    try:
        x=FeaCon2(f1,x)
    except:
        x=f1
    f1 = f1 * i
    newf = addCon2(f1, img1 * i2)
    return newf,img2,x

def new_FCA1(image, m, i):
    i = round((i * 0.1), 2)
    i2 = round((1 - i), 2)
    # img1是已有特征
    img1 = image[0]
    # img2是像素点
    img2 = image[1]
    # concaf是像素点是一般特征
    if m == 0:
        f1 = all_sift(img2)
    elif m == 1:
        f1 = all_lbp(img2)
    else:
        f1 = add_hog_small(img2)
    f1 = f1 * i
    newf = addCon2(f1, img1 * i2)
    return newf,img2

def all_histogram(image):
    # global and local
    n_bins = 32
    hist, ax = numpy.histogram(image, n_bins, [0, 1])
    # dimension 24 for all type images
    return hist

def hist(left):
    img = []
    for i in range(left.shape[0]):
        img.append(all_histogram(left[i, :, :]))
    return numpy.asarray(img)

def hist1(left):
    left = original2(left)
    img = []
    for i in range(left.shape[0]):
        img.append(all_histogram(left[i, :, :]))
    return numpy.asarray(img)


def L_feature(image,m):
    if m==0:
        f1=all_sift(image)
    elif m==1:
        f1=all_lbp(image)
    elif m==2:
        f1=lobal_hog_small(image)
    else:
        f1=hist(image)
    try:
       f1=min_max_scaler.fit_transform(f1)
    except:
        f1=f1
    return f1,image


# def L2_feature(image,m):
#     a=0
#     if m==0:
#         f1=hist_conva(image)
#     elif m==1:
#         f1=sift_conva(image)
#     else:
#
#     return f1,image,a

# def L_feature(image,m):
#     if m==0:
#         f1=all_sift(image)
#     elif m==1:
#         f1=all_lbp(image)
#     elif m==2:
#         eqlf=eql(image)
#         f1=conva(eqlf)
#     elif m==3:
#         sobelf=sobelxy(image)
#         f1=conva(sobelf)
#     elif m==4:
#         canny=cannyf(image)
#         f1=conva(canny)
#     else:
#         f1=lobal_hog_small(image)
#     f1=min_max_scaler.fit_transform(f1)
#     return f1,image






def conf(image):
    f1=conva(image)
    return f1,image
def twovec2(image):
    a=all_sift(image)
    d=all_lbp(image)
    c=FeaCon2(a,d)
    return c



# def twovec2(image):
#     a=all_sift(image)
#     b=lobal_hog_small(image)
#     c=FeaCon2(a,b)
#     # c = addCon2(a, b)
#     return c




def gau(left, si):
    img = []
    for i in range(left.shape[0]):
        img.append(ndimage.gaussian_filter(left[i, :, :], sigma=si))
    return np.asarray(img)
def eql(left):
    img = []
    for i in range(left.shape[0]):
        img.append(exposure.equalize_hist(left[i, :, :]))
    return np.asarray(img)
def abss(left):
    img = []
    for i in range(left.shape[0]):
        img.append(numpy.abs(left[i, :, :]))
    return np.asarray(img)

def gauD(left, si, or1, or2):
    img  = []
    for i in range(left.shape[0]):
        img.append(ndimage.gaussian_filter(left[i,:,:],sigma=si, order=[or1,or2]))
    return np.asarray(img)

def gab(left,the,fre):
    fmax=numpy.pi/2
    a=numpy.sqrt(2)
    freq=fmax/(a**fre)
    thea=numpy.pi*the/8
    img = []
    for i in range(left.shape[0]):
        filt_real,filt_imag=numpy.asarray(gabor(left[i,:,:],theta=thea,frequency=freq))
        img.append(filt_real)
    return numpy.asarray(img)

def gaussian_Laplace1(left):
    return ndimage.gaussian_laplace(left,sigma=1)

def gaussian_Laplace2(left):
    return ndimage.gaussian_laplace(left,sigma=2)

def laplace(left):
    return ndimage.laplace(left)

def sobelxy(left):
    img = []
    for i in range(left.shape[0]):
        img.append(ndimage.sobel(left[i, :, :]))
    return np.asarray(img)

def cannyf(left):
    img = []
    for i in range(left.shape[0]):
        img.append(canny(left[i, :, :]))
    return np.asarray(img)

def sobelx(left):
    img = []
    for i in range(left.shape[0]):
        img.append(ndimage.sobel(left[i,:,:], axis=0))
    return np.asarray(img)

def sobely(left):
    img = []
    for i in range(left.shape[0]):
        img.append(ndimage.sobel(left[i, :, :], axis=1))
    return np.asarray(img)

#max filter
def maxf(image):
    img = []
    size = 3
    for i in range(image.shape[0]):
        img.append(ndimage.maximum_filter(image[i,:,:],size))
    return np.asarray(img)

#median_filter
def medianf(image):
    img = []
    size = 3
    for i in range(image.shape[0]):
        img.append(ndimage.median_filter(image[i,:,:],size))
    return np.asarray(img)

#mean_filter
def meanf(image):
    img = []
    size = 3
    for i in range(image.shape[0]):
        img.append(ndimage.convolve(image[i,:,:], numpy.full((3, 3), 1 / (size * size))))
    return np.asarray(img)

#minimum_filter
def minf(image):
    img = []
    size = 3
    for i in range(image.shape[0]):
        img.append(ndimage.minimum_filter(image[i,:,:],size))
    return np.asarray(img)

def lbp(image):
    img = []
    for i in range(image.shape[0]):
        # 'uniform','default','ror','var'
        lbp = local_binary_pattern(image[i,:,:], 8, 1.5, method='nri_uniform')
        img.append(np.divide(lbp,59))
    return np.asarray(img)


def hog_feature(image):
    try:
        img = []
        for i in range(image.shape[0]):
            img1, realImage = hog(image[i, :, :], orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(3, 3), block_norm='L2-Hys', visualize=True,
                                transform_sqrt=False, feature_vector=True)
            img.append(realImage)
        data = np.asarray(img)
    except: data = image
    return data

def mis_match(img1,img2):
    n, w1,h1=img1.shape
    n, w2,h2=img2.shape
    w=min(w1,w2)
    h=min(h1,h2)
    return img1[:, 0:w,0:h],img2[:, 0:w,0:h]

def mean_fusion(image1, image2):
    fused_image = (image1 + image2) / 2
    return fused_image

def median_fusion(image1, image2):
    fused_image = np.median([image1, image2], axis=0)
    return fused_image

def max_pooling(image, target_size):
    pooled_image = image[:, :target_size[0], :target_size[1]]
    return pooled_image

def min_fusion(image1, image2):
    fused_image = np.minimum(image1, image2)
    return fused_image

def min_fusion1(image1,image2):
    min_size = (np.min([image1.shape[1], image2.shape[1]]), np.min([image1.shape[2], image2.shape[2]]))
    pooled_image1 = max_pooling(image1, min_size)
    pooled_image2 = max_pooling(image2, min_size)
    fused_image = min_fusion(pooled_image1, pooled_image2)
    return fused_image

def median_fusion1(image1,image2):
    max_size = (max(image1.shape[1], image2.shape[1]), max(image1.shape[2], image2.shape[2]))
    fused_images = []
    for img1, img2 in zip(image1, image2):
        # Resize each image separately
        resized_image1 = resize_image(img1, max_size)
        resized_image2 = resize_image(img2, max_size)
        # Fuse the pooled images and add the result to the list
        fused_image = median_fusion(resized_image1, resized_image2)
        fused_images.append(fused_image)
    return np.array(fused_images)

# def mean_fusion1(image1,image2):
#     min_size = (np.min([image1.shape[1], image2.shape[1]]), np.min([image1.shape[2], image2.shape[2]]))
#     pooled_image1 = max_pooling(image1, min_size)
#     pooled_image2 = max_pooling(image2, min_size)
#     fused_image = mean_fusion(pooled_image1, pooled_image2)
#     return fused_image

def average_pooling(image, target_size):
    pooled_image = image[:, :target_size[0], :target_size[1]]
    pooled_image = np.mean(pooled_image, axis=(1, 2), keepdims=True)
    return pooled_image

def mean_fusion1(image1, image2):
    max_size = (max(image1.shape[1], image2.shape[1]), max(image1.shape[2], image2.shape[2]))
    fused_images = []
    # Process each pair of corresponding images in the datasets
    for img1, img2 in zip(image1, image2):
        # Resize each image separately
        resized_image1 = resize_image(img1, max_size)
        resized_image2 = resize_image(img2, max_size)
        # Fuse the pooled images and add the result to the list
        fused_image = mean_fusion(resized_image1, resized_image2)
        fused_images.append(fused_image)
    return np.array(fused_images)

def mixconadd(img1, img2, w2):
    w1=1-w2
    img11,img22=mis_match(img1,img2)
    return numpy.add(img11*w1,img22*w2)

def mixconsub(img1, img2, w2):
    w1 = 1 - w2
    img11,img22=mis_match(img1,img2)
    return numpy.subtract(img11*w1,img22*w2)

def sqrt(left):
    with numpy.errstate(divide='ignore',invalid='ignore'):
        x = numpy.sqrt(left,)
        if isinstance(x, numpy.ndarray):
            x[numpy.isinf(x)] = 1
            x[numpy.isnan(x)] = 1
        elif numpy.isinf(x) or numpy.isnan(x):
            x = 1
    return x

def relu(left):
    return (abs(left)+left)/2

def maxP(left, kel1, kel2):
    img = []
    for i in range(left.shape[0]):
        current = skimage.measure.block_reduce(left[i,:,:], (kel1,kel2),numpy.max)
        img.append(current)
    return np.asarray(img)

import numpy as np
from skimage.measure import block_reduce
from skimage.transform import resize
def resize_image(image, target_size):
    return resize(image, target_size, mode='reflect', anti_aliasing=True)
def min_fusion2(image1, image2):
    fused_image = np.minimum(image1, image2)
    return fused_image
def avg_pooling2x2(image):
    # Compute the pooling operation on each channel separately
    return np.stack([block_reduce(channel, (2, 2), np.mean) for channel in image])




def avg_pooling(image):
    pool_size = (2, 2)
    pooled_image = block_reduce(image, pool_size, np.mean)
    return pooled_image


def min_fusion5(image1, image2):
    # Get the minimum size across all images in the datasets
    max_size = (max(image1.shape[1], image2.shape[1]), max(image1.shape[2], image2.shape[2]))
    fused_images = []
    # Process each pair of corresponding images in the datasets
    for img1, img2 in zip(image1, image2):
        # Resize each image separately
        resized_image1 = resize_image(img1, max_size)
        resized_image2 = resize_image(img2, max_size)
        # Fuse the pooled images and add the result to the list
        fused_image = min_fusion2(resized_image1, resized_image2)
        fused_images.append(fused_image)

    # Convert the list of fused images back to a numpy array
    return np.array(fused_images)


def interpolation_downsampling_dataset(input_data,kel1,kel2):
    num_samples = input_data.shape[0]
    new_height = input_data.shape[1] // kel1
    new_width = input_data.shape[2] // kel2

    downscaled_data = np.zeros((num_samples, new_height, new_width))

    for i in range(num_samples):
        current_sample = input_data[i]
        downscaled_sample = ndimage.zoom(current_sample, (1 / kel1, 1 / kel2), order=1)
        downscaled_data[i] = downscaled_sample[:new_height, :new_width]

    return downscaled_data


def downsampling(input_data,kel1,kel2):
    current = interpolation_downsampling_dataset(input_data,kel1,kel2)
    return input_data, np.asarray(current)


def average_pooling(input_data,kel1,kel2):
    # kel1 = 2
    # kel2 = 2
    img = []
    for i in range(input_data.shape[0]):
        current = skimage.measure.block_reduce(input_data[i, :, :], (kel1, kel2), np.mean)
        img.append(current)
    return input_data, np.asarray(img)


def maxPnew(left,kel1,kel2):

    img = []
    for i in range(left.shape[0]):
        current = skimage.measure.block_reduce(left[i,:,:], (kel1,kel2),numpy.max)
        img.append(current)
    return left,np.asarray(img)

def maxPnew2(left, kel1, kel2):
    img = []
    for i in range(left.shape[0]):
        current = skimage.measure.block_reduce(left[i,:,:], (kel1,kel2),numpy.max)
        img.append(current)
    return np.asarray(img)

def maxPACONCA(left, kel1, kel2):
    f=[]
    img=maxP(left,kel1,kel2)
    for i in range(left.shape[0]):
        a=conVector(img)
        f.append(a)
    return np.asarray(f)

def Prewitt(left):
    img = []
    for i in range(left.shape[0]):
        img.append(ndimage.prewitt(left[i, :, :]))
    return np.asarray(img)
def PrewittH(left):
    img = []
    for i in range(left.shape[0]):
        img.append(ndimage.prewitt(left[i, :, :], axis=0))
    return np.asarray(img)
def PrewittV(left):
    img = []
    for i in range(left.shape[0]):
        img.append(ndimage.prewitt(left[i, :, :],axis=1))
    return np.asarray(img)
# 罗伯茨算子
def Roberts(left):
    img = []
    for i in range(left.shape[0]):
        img.append(roberts(left[i, :, :]))
    return np.asarray(img)

def random_m():
    a=round(random.random(), 2)
    return a


def random_filters():
    filters = []
    a = numpy.random.randint(0, 10)
    b = numpy.random.randint(0, 10 - a)
    # c = (10 - a - b)
    filters.append(round((a*0.1),1))
    filters.append(round((b*0.1),1))
    # filters.append(round((c*0.1),1))
    return filters
def random_filters1(filter_size):
    filters = []
    a = numpy.random.randint(0, 10)
    b = 10-a
    c = (10 - a - b)
    filters.append(round((a*0.1),1))
    filters.append(round((b*0.1),1))
    # filters.append(c*0.1)
    return filters

def imresize(arr, size, interp='bilinear', mode=None):
    im = Image.fromarray(arr, mode=mode)
    ts = type(size)
    if numpy.issubdtype(ts, numpy.signedinteger):
        percent = size / 100.0
        size = tuple((numpy.array(im.size)*percent).astype(int))
    elif numpy.issubdtype(type(size), numpy.floating):
        size = tuple((numpy.array(im.size)*size).astype(int))
    else:
        size = (size[1], size[0])
    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
    imnew = im.resize(size, resample=func[interp])
    return numpy.array(imnew)

def regionSquare(left,x,y,size):
    width, height = left[0, :, :].shape
    img = []
    for i in range(left.shape[0]):
        x_end = min(width, x + size)
        y_end = min(height, y + size)
        a=left[i, :, :]
        slice=a[x:x_end, y:y_end]
        img.append((slice))
    return np.asarray(img)


def regionR(left,x,y,size,size2):
    left=np.array(left)
    width, height = left[0, :, :].shape
    img = []
    for i in range(left.shape[0]):
        x_end = min(width, x + size)
        y_end = min(height, y + size2)
        a=left[i, :, :]
        slice=a[x:x_end, y:y_end]
        img.append((slice))

    return np.asarray(img)


def original2(left):
    img = []
    for i in range(left.shape[0]):
        width, height = left[0, :, :].shape
        width = int(width / 4)
        height = int(height / 4)
        image = imresize(left[i, :, :], (width, height))
        img.append((image))
    return np.asarray(img)
def original(left):
    img = []
    for i in range(left.shape[0]):
        image = left[i, :, :]
        img.append((image))
    return np.asarray(img)

def LEFT_RIGHT(img,mode=None):
    # 左右反转
    im = Image.fromarray(img, mode=mode)
    img_lr = im.transpose(Image.FLIP_LEFT_RIGHT)
    return numpy.array(img_lr)
def updown(img,mode=None):
    # 左右反转
    im = Image.fromarray(img, mode=mode)
    img_lr = im.transpose(Image.FLIP_TOP_BOTTOM)
    return numpy.array(img_lr)

def ttransp(left):
    img = []
    for i in range(left.shape[0]):
        image = LEFT_RIGHT(left[i, :, :])
        img.append((image))
    return np.asarray(img)

def uptrans(left):

    img = []
    for i in range(left.shape[0]):
        image = updown(left[i, :, :])
        img.append((image))
    return np.asarray(img)


def random_filters():
    filters = []
    a = numpy.random.randint(0, 10)
    b = numpy.random.randint(0, 10 - a)
    # c = (10 - a - b)
    filters.append(round((a*0.1),1))
    filters.append(round((b*0.1),1))
    # filters.append(round((c*0.1),1))
    return filters