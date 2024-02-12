# python packages
import operator
import random
import time
import gp_restrict as gp_restrict
import algo_subtwotreegp as evalGP
import numpy as np
from deap import base, creator, tools, gp
import felgp_functions as felgp_fs
from strongGPDataType import Int1, Int2, Int3, Int4, Int5, Int6, Int100, Int7, Int8, Int9,Int102
from strongGPDataType import Float1, Float2, Float3
from strongGPDataType import Array1, Array2, Array3, Array4, Array5, Array6, Filter, Array0, Vector, Origin1, Vector2,tuple1,shallowf,Origin2,tuple2,deepf,Int101,Int15,Int16,Array8,Array7,Array,Vector1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import cv2
import math
np.set_printoptions(threshold=1000000)

# defined by author
import saveFile
import sys

import warnings
import sys

warnings.filterwarnings("ignore")




# parameters:
# 修改：时间上的选取第三个分类器为权值则不计算。

pop_size =250
generation =50
cxProb = 0.8
mutProb = 0.19
elitismProb = 0.01
totalRuns = 1
initialMinDepth = 2
initialMaxDepth = 6
maxDepth = 11

##GP
pset = gp.PrimitiveSetTyped('MAIN', [Array0, Array2], Array7, prefix='Image')

#
pset.addPrimitive(felgp_fs.linear_svmtwo, [Vector1,Array2], Array7, name='S_SVM')

pset.addPrimitive(felgp_fs.FeaCon2, [Vector1, Vector1],Vector1, name='Roots_D2')
pset.addPrimitive(felgp_fs.N_FeaCon3, [Vector1, Vector1,Vector1],Vector1, name='Roots_D3')

pset.addPrimitive(felgp_fs.newFeaCon, [tuple1,Int5], Vector1, name='L_FeaCon')
pset.addPrimitive(felgp_fs.new_FCA2, [tuple1,Int4,Int100], tuple1, name='Add_F')

#
pset.addPrimitive(felgp_fs.sift_conva1, [Vector], tuple1, name='SIFT_Conva')
pset.addPrimitive(felgp_fs.hog_conva1, [Vector], tuple1, name='HOG_Conva')
pset.addPrimitive(felgp_fs.lbp_conva1, [Vector], tuple1, name='LBP_Conva')


pset.addPrimitive(felgp_fs.maxPnew, [Origin1,Int3,Int3], Vector, name='MaxP')
pset.addPrimitive(felgp_fs.average_pooling, [Origin1,Int3,Int3], Vector, name='AverP')
pset.addPrimitive(felgp_fs.downsampling, [Origin1,Int3,Int3], Vector, name='DownsampP')
#
pset.addPrimitive(felgp_fs.min_fusion5, [Origin1, Origin1], Origin1, name='Min_fusion')
pset.addPrimitive(felgp_fs.median_fusion1, [Origin1, Origin1], Origin1, name='Median_fusion')
pset.addPrimitive(felgp_fs.mean_fusion1, [Origin1, Origin1], Origin1, name='Mean_fusion')
pset.addPrimitive(felgp_fs.eql, [Origin1], Origin1, name='Equalize')

pset.addPrimitive(felgp_fs.regionR, [Array0, Int7, Int8, Int9,Int9], Origin1, name='Region_R')
pset.addPrimitive(felgp_fs.regionSquare, [Array0, Int7, Int8, Int9], Origin1, name='Region_S')

pset.addPrimitive(felgp_fs.gau, [Array0, Int1], Array0, name='Gau1')
pset.addPrimitive(felgp_fs.gauD, [Array0, Int1, Int2, Int2], Array0, name='GauD1')
pset.addPrimitive(felgp_fs.gab, [Array0, Float1, Float2], Array0, name='Gabor1')
pset.addPrimitive(felgp_fs.laplace, [Array0], Array0, name='Lap')
pset.addPrimitive(felgp_fs.gaussian_Laplace1, [Array0], Array0, name='LoG1')
pset.addPrimitive(felgp_fs.abss, [Array0], Array0, name='Abs1')
pset.addPrimitive(felgp_fs.gaussian_Laplace2, [Array0], Array0, name='LoG2')
pset.addPrimitive(felgp_fs.sobelxy, [Array0], Array0, name='Sobel')
pset.addPrimitive(felgp_fs.medianf, [Array0], Array0, name='Med')
pset.addPrimitive(felgp_fs.maxf, [Array0], Array0, name='Max')
pset.addPrimitive(felgp_fs.minf, [Array0], Array0, name='Min')
pset.addPrimitive(felgp_fs.meanf, [Array0], Array0, name='Mean')
pset.addPrimitive(felgp_fs.sqrt, [Array0], Array0, name='Sqrt')
pset.addPrimitive(felgp_fs.relu, [Array0], Array0, name='Relu')
pset.addPrimitive(felgp_fs.Prewitt, [Array0], Array0, name='Prewitt')
pset.addPrimitive(felgp_fs.Roberts, [Array0], Array0, name='Roberts')



# Terminals
pset.renameArguments(ARG0='grey')
pset.addEphemeralConstant('Singma', lambda: random.randint(1, 4), Int1)
pset.addEphemeralConstant('Order', lambda: random.randint(0, 3), Int2)
pset.addEphemeralConstant('m', lambda: random.randint(0, 2), Int4)
pset.addEphemeralConstant('local_fp', lambda: random.randint(0, 1), Int5)
pset.addEphemeralConstant('Theta', lambda: random.randint(0, 8), Float1)
pset.addEphemeralConstant('Frequency', lambda: random.randint(0, 5), Float2)
pset.addEphemeralConstant('n', lambda: round(random.random(), 3), Float3)
pset.addEphemeralConstant('KernelSize', lambda: random.randrange(2,10,2), Int3)
pset.addEphemeralConstant('Ks', lambda: random.randrange(4, 10, 2), Int102)
pset.addEphemeralConstant('X', lambda: random.randint(0, bound1 - 20), Int7)
pset.addEphemeralConstant('Y', lambda: random.randint(0, bound2 - 20), Int8)
pset.addEphemeralConstant('i', lambda: felgp_fs.random_m(), Int100)
pset.addEphemeralConstant('C', lambda: random.randint(0, 5), Int101)


# 边长
pset.addEphemeralConstant('Size', lambda: random.randint(20, 50), Int9)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=initialMinDepth, max_=initialMaxDepth)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("mapp", map)


def evalTrain(toolbox, individual, hof, trainData, trainLabel):
    if len(hof) != 0 and individual in hof:
        ind = 0
        while ind < len(hof):
            if individual == hof[ind]:
                accuracy, = hof[ind].fitness.values
                ind = len(hof)
            else: ind+=1
    else:
         try:
              func = toolbox.compile(expr=individual)
              output = np.asarray(func(trainData, trainLabel))
              y_predict = np.argmax(output[0], axis=1)
              accuracy = 100*np.sum(y_predict == trainLabel) / len(trainLabel)
         except:
           accuracy=0
    return accuracy,


toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("selecttwo", tools.selTournament, tournsize=5)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
toolbox.register("mutate_eph", gp.mutEphemeral, mode='all')
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))


def GPMain(randomSeeds):
    random.seed(randomSeeds)
    pop = toolbox.population(pop_size)
    hof = tools.HallOfFame(10)
    log = tools.Logbook()
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size_tree = tools.Statistics(key=len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size_tree=stats_size_tree)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    log.header = ["gen", "evals"] + mstats.fields

    pop, log,ind,BSND = evalGP.eaSimple(pop, toolbox, cxProb, mutProb, elitismProb, generation, randomSeeds,
                               stats=mstats, halloffame=hof, verbose=True)

    return pop, log, hof,ind,BSND


def evalTest(toolbox, individual, trainData, trainLabel, test, testL):
    x_train = np.concatenate((trainData, test), axis=0)
    func = toolbox.compile(expr=individual)
    try:
        output = np.asarray(func(x_train, trainLabel))
        output1 = output[0]
        # print(output.shape)
        y_predict = np.argmax(output1, axis=1)
        accuracy = 100 * np.sum(y_predict == testL) / len(testL)
    except:
        accuracy = 0
    return accuracy

# def evalTest2(toolbox, individual, trainData, trainLabel, test, testL,pop):
#     x_train = np.concatenate((trainData, test), axis=0)
#     print(individual)
#     try:
#         output1=0
#         for i in pop:
#             func = toolbox.compile(expr=i)
#             output = np.asarray(func(x_train, trainLabel))
#             output1 = output[0]+output1
#         y_predict = np.argmax(output1, axis=1)
#         accuracy = 100 * np.sum(y_predict == testL) / len(testL)
#     except:
#         accuracy = 0
#     print("适应度最好的10个个体集成")
#     print(accuracy)
def evalTest2(toolbox, individual, trainData, trainLabel, test, testL, pop):
    x_train = np.concatenate((trainData, test), axis=0)
    operations = ['S_SVM', 'L_FeaCon', 'LBP_Conva', 'MaxP', 'Median_fusion', 'Region_S',
                  'Gau1', 'GauD1', 'Gabor1', 'Lap', 'LoG1', 'Abs1', 'LoG2', 'Sobel',
                  'Med', 'Max', 'Min', 'Mean', 'Sqrt', 'Relu', 'Prewitt', 'Roberts', 'Roots_D',
                  'Add_F', 'SIFT_Conva', 'HOG_Conva', 'AverP', 'DownsampP', 'Min_fusion',
                  'Mean_fusion', 'Equalize', 'Region_R']
    operation_counts = []
    # 遍历每个个体
    for i in pop:
        individual_str = str(i)
        total = 0
        # 对每个操作名进行计数
        for operation in operations:
            count = individual_str.count(operation)
            # print(f"{operation}: {count}")
            total += count
        operation_counts.append(total)
        # print("\n")  # 输出空行以区分不同个体的结果

    # 计算最优个体中每个操作的总数
    optimal_str = str(individual)

    optimal_total = sum([optimal_str.count(operation) for operation in operations])

    # 计算每个个体与最优个体的距离
    distances = [abs(count - optimal_total) for count in operation_counts]
    print(distances)
    # 获取距离最大的4个值的索引
    top4_indices = sorted(range(len(distances)), key=lambda i: distances[i])[-3:]
    # 选择距离最大的4个个体
    top4_individuals = [pop[i] for i in top4_indices]
    top4_individuals.append(individual)
    top4_individuals.append(individual)
    try:
        output1 = 0
        for i in top4_individuals:
            print(str(i))
            func = toolbox.compile(expr=i)
            output = np.asarray(func(x_train, trainLabel))
            output1 = output[0] + output1
        y_predict = np.argmax(output1, axis=1)
        accuracy = 100 * np.sum(y_predict == testL) / len(testL)
    except:
        accuracy = 0
    print("返回10个体，按照距离策略，进行3个最远距离+二最优个体进行5集成。")
    print(accuracy)
    return accuracy
def evalTest3(toolbox, individual, trainData, trainLabel, test, testL,pop):
    x_train = np.concatenate((trainData, test), axis=0)
    print(individual)
    try:
        output1=0
        print(len(pop))
        for i in pop:
            func = toolbox.compile(expr=i)
            output = np.asarray(func(x_train, trainLabel))
            output1 = output[0]+output1
        # print(output.shape)
        y_predict = np.argmax(output1, axis=1)
        accuracy = 100 * np.sum(y_predict == testL) / len(testL)
    except:
        accuracy = 0
    print("超过30代后的每个最优个体全部集成")
    print(accuracy)
    return accuracy

def evalTest4(toolbox, individual, trainData, trainLabel, test, testL,pop):
    print(len(pop))
    x_train = np.concatenate((trainData, test), axis=0)
    print(individual)
    try:
        output1=0
        for i in pop:
            func = toolbox.compile(expr=i)
            output = np.asarray(func(x_train, trainLabel))
            output1 = output[0]+output1
        y_predict = np.argmax(output1, axis=1)
        accuracy = 100 * np.sum(y_predict == testL) / len(testL)
    except:
        accuracy = 0
    print("停滞后的个体集成")
    print(accuracy)
    pop.append(individual)
    try:
        output1=0
        for i in pop:
            func = toolbox.compile(expr=i)
            output = np.asarray(func(x_train, trainLabel))
            output1 = output[0]+output1
        y_predict = np.argmax(output1, axis=1)
        accuracy2 = 100 * np.sum(y_predict == testL) / len(testL)
    except:
        accuracy2 = 0
    print("停滞后的个体多一票最优")
    print(accuracy2)
    return accuracy,accuracy2

def evalTest5(toolbox, individual, trainData, trainLabel, test, testL, pop):
    x_train = np.concatenate((trainData, test), axis=0)
    operations = ['S_SVM', 'L_FeaCon', 'LBP_Conva', 'MaxP', 'Median_fusion', 'Region_S',
                  'Gau1', 'GauD1', 'Gabor1', 'Lap', 'LoG1', 'Abs1', 'LoG2', 'Sobel',
                  'Med', 'Max', 'Min', 'Mean', 'Sqrt', 'Relu', 'Prewitt', 'Roberts',
                  'Add_F', 'SIFT_Conva', 'HOG_Conva', 'AverP', 'DownsampP', 'Min_fusion',
                  'Mean_fusion', 'Equalize', 'Region_R']
    operation_counts = []
    # 遍历每个个体
    for i in pop:
        individual_str = str(i)
        total = 0
        # 对每个操作名进行计数
        for operation in operations:
            count = individual_str.count(operation)
            # print(f"{operation}: {count}")
            total += count
        operation_counts.append(total)
        # print("\n")  # 输出空行以区分不同个体的结果

    # 计算最优个体中每个操作的总数
    optimal_str = str(individual)

    optimal_total = sum([optimal_str.count(operation) for operation in operations])

    # 计算每个个体与最优个体的距离
    distances = [abs(count - optimal_total) for count in operation_counts]
    print(distances)
    # 获取距离最大的4个值的索引
    top4_indices = sorted(range(len(distances)), key=lambda i: distances[i])[-4:]
    # 选择距离最大的4个个体
    top4_individuals = [pop[i] for i in top4_indices]
    top4_individuals.append(individual)
    try:
        output1 = 0
        for i in top4_individuals:
            print(str(i))
            func = toolbox.compile(expr=i)
            output = np.asarray(func(x_train, trainLabel))
            output1 = output[0] + output1
        y_predict = np.argmax(output1, axis=1)
        accuracy = 100 * np.sum(y_predict == testL) / len(testL)
    except:
        accuracy = 0
    print("返回10个体，按照距离策略，4最大距离+一个最优。")
    print(accuracy)
    return accuracy

def evalTest6(toolbox, individual, trainData, trainLabel, test, testL, pop):
    x_train = np.concatenate((trainData, test), axis=0)
    top4_individuals = pop[:3]
    top4_individuals.append(individual)
    top4_individuals.append(individual)
    try:
        output1 = 0
        for i in top4_individuals:
            print(str(i))
            func = toolbox.compile(expr=i)
            output = np.asarray(func(x_train, trainLabel))
            output1 = output[0] + output1
        y_predict = np.argmax(output1, axis=1)
        accuracy = 100 * np.sum(y_predict == testL) / len(testL)
    except:
        accuracy = 0
    print("返回10个体，直接取前三，加上两个最优，进行5集成。")
    print(accuracy)
    return accuracy

def evalTest7(toolbox, individual, trainData, trainLabel, test, testL, pop):
    x_train = np.concatenate((trainData, test), axis=0)
    operations = ['S_SVM', 'L_FeaCon', 'LBP_Conva', 'MaxP', 'Median_fusion', 'Region_S',
                  'Gau1', 'GauD1', 'Gabor1', 'Lap', 'LoG1', 'Abs1', 'LoG2', 'Sobel',
                  'Med', 'Max', 'Min', 'Mean', 'Sqrt', 'Relu', 'Prewitt', 'Roberts',
                  'Add_F', 'SIFT_Conva', 'HOG_Conva', 'AverP', 'DownsampP', 'Min_fusion',
                  'Mean_fusion', 'Equalize', 'Region_R']
    operation_counts = []
    # 遍历每个个体
    for i in pop:
        individual_str = str(i)
        total = 0
        # 对每个操作名进行计数
        for operation in operations:
            count = individual_str.count(operation)
            # print(f"{operation}: {count}")
            total += count
        operation_counts.append(total)
        # print("\n")  # 输出空行以区分不同个体的结果

    # 计算最优个体中每个操作的总数
    optimal_str = str(individual)

    optimal_total = sum([optimal_str.count(operation) for operation in operations])

    # 计算每个个体与最优个体的距离
    distances = [abs(count - optimal_total) for count in operation_counts]
    print(distances)
    # 获取距离最大的4个值的索引
    top4_indices = sorted(range(len(distances)), key=lambda i: distances[i])[-3:]
    # 选择距离最大的4个个体
    top4_individuals = [pop[i] for i in top4_indices]
    top4_individuals.append(individual)
    top4_individuals.append(individual)
    try:
        output1 = 0
        for i in top4_individuals:
            print(str(i))
            func = toolbox.compile(expr=i)
            output = np.asarray(func(x_train, trainLabel))
            output1 = output[0] + output1
        y_predict = np.argmax(output1, axis=1)
        accuracy = 100 * np.sum(y_predict == testL) / len(testL)
    except:
        accuracy = 0
    print("返回10个体，按照距离策略，3其他+2个最优。")
    print(accuracy)
    return accuracy
def method_name(randseed):
    print(randseed)
    train = pd.read_csv('D:/dataset/csv/'+dataSetName+'.csv')
    train_image = []
    for i in range(train.shape[0]):  # 2408
        # 加载图像，这里进入文件夹下，遍历csv文件中列名为'image'的列，即图片名，进而读取图片
        # ‘训练集文件夹/’ C:\Users\96598\Desktop\jaffe_lq\Blur
        img_cv2 = cv2.imread('D:/dataset/'+datasetname+'/' + train['image'][i])
        # 将目标大小设置为（224,224,3）
        image = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        # 180x130 才是130x180
        # 论文写的是 180x 130  这里写成130 180
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        train_image.append(image)
    # 将列表转为numpy数组
    X = np.array(train_image)  # train_image中有2408张图片,都是numpy格式
    X = X.astype('float32')
    # 区分目标
    Y = train['class']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=testsize, random_state=randseed, stratify=Y, shuffle=True)
    np.save(datasetname+"_train_data", X_train)
    np.save(datasetname+"_train_label", y_train)
    np.save(datasetname+"_test_data", X_test)
    np.save(datasetname+"_test_label", y_test)
    A = np.load(dataSetName + '_train_label.npy')
    print(A)


import json
with open("FEI2.json", "r") as f:
    data_params = json.load(f)
results = {}
for key, dataset in data_params.items():
    dataset_results = {}
    datasetname = dataset["datasetname"]
    width = int(dataset["width"])
    height = int(dataset["height"])
    testsize = int(dataset["testsize"])
    testnum = 30
    dataSetName=dataset["dataSetName"]
    sum1 = 0
    a = 0
    timesum = 0
    list1 = []
    list_Tree_distances_TwoBest = []
    list_Thirty_pop = []
    list_TZ1 = []
    list_Tz2 = []
    list_Four_distance_1best = []
    list_qianThree_twobest = []
    list_Tree_distances_TwoBest_shao = []
    testnum = int(testnum)
    for i in range(testnum):
        randomSeeds = random.randint(0,200)
        method_name(randomSeeds)
        x_train = np.load(dataSetName + '_train_data.npy') / 255.0
        y_train = np.load(dataSetName + '_train_label.npy')
        x_test = np.load(dataSetName + '_test_data.npy') / 255.0
        y_test = np.load(dataSetName + '_test_label.npy')
        print(y_train)
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        print(x_train.max())
        bound1, bound2 = x_train[1, :, :].shape
        num_train = x_train.shape[0]
        beginTime = time.process_time()
        toolbox.register("evaluate", evalTrain, toolbox, trainData=x_train, trainLabel=y_train)
        pop, log, hof, ind, bsind = GPMain(randomSeeds)
        endTime = time.process_time()
        trainTime = endTime - beginTime
        testResults = evalTest(toolbox, hof[0], x_train, y_train, x_test, y_test)

        testTime = time.process_time() - endTime
        print('testResults ', testResults)
        if testResults != 0:
            list1.append(testResults)
            sum1 = sum1 + testResults
            a = a + 1
        randomSeeds=randomSeeds+1

        timesum = trainTime + timesum
    print("训练时间为")
    print(timesum)
    print("平均值为：")
    print(sum1 / a)
    print("标准差为：")
    print(np.std(list1, ddof=1))
    print("最大值为：")
    print(np.max(list1))
    print(list1)




