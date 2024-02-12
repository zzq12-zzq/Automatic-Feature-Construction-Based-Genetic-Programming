import pickle

from deap import gp






def saveLog (fileName, log):
   f = open(fileName, 'wb')
   pickle.dump(log, f)
   f.close()
   return
# def saveLog (fileName, log):
#    f=open(fileName, 'wb')
#    pickle.dump(log, f)
#    f.close()
 #   return


# def plotTree(pathName,individual):
#    nodes, edges, labels = gp.graph(individual)
#    g = pgv.AGraph()
#    g.add_nodes_from(nodes)
#    g.add_edges_from(edges)
#    g.layout(prog="dot")

#    for i in nodes:
#        n = g.get_node(i)
#        n.attr["label"] = labels[i]
#    g.draw(pathName)
#    return


def bestInd(toolbox, population, number):
    bestInd = []
    best = toolbox.selectElitism(population, k=number)
    for i in best:
        bestInd.append(i)
    return bestInd
        


def saveResults(fileName, *data_dict_list):
    with open(fileName, 'w') as f:
        for data_dict in data_dict_list:
            for key, value in data_dict.items():
                f.write(f'{key}: {value}\n')
            f.write('\n')

def save_Best_Algorithm_Results(randomSeeds, dataSetName, hof, trainTime, testResults, log,
                                Tree_distances_TwoBest, Thirty_pop, TZ1, Tz2, Four_distance_1best,
                                qianThree_twobest, Tree_distances_TwoBest_shao):
    data_dict_list = [
        {'randomSeed': randomSeeds, 'dataSetName': dataSetName},
        {'trainTime': trainTime},
        {'trainResults': hof[0].fitness},
        {'testResults': testResults},
        {'bestInd in training': hof[0]},
        {'log': log},
        {'Tree_distances_TwoBest': Tree_distances_TwoBest},
        {'Thirty_pop': Thirty_pop},
        {'TZ1': TZ1},
        {'Tz2': Tz2},
        {'Four_distance_1best': Four_distance_1best},
        {'qianThree_twobest': qianThree_twobest},
        {'Tree_distances_TwoBest_shao': Tree_distances_TwoBest_shao}
    ]
    fileName = f'D:/EX_Result/Best_Algorithm/{randomSeeds}_{dataSetName}.txt'
    saveResults(fileName, *data_dict_list)