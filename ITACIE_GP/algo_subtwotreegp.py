import operator
import random
from deap import tools
from collections import defaultdict
# def delete(toolbox,pop,size):
#     temp = [toolbox.clone(ind) for ind in pop]
#     res = []
#     ttres=[]
#     for i in temp:
#         if i not in res:
#             res.append(i)
#         else:
#             ttres.append(i)
#     tempsize=len(temp)-size-len(res)
#     ttres=ttres[0:tempsize]
#     res=ttres+res
#     return res
def delete(toolbox, pop, size):
    unique_elements = list(set(toolbox.clone(ind) for ind in pop))
    remaining_size = size - len(unique_elements)

    if remaining_size > 0:
        duplicate_elements = [ind for ind in pop if pop.count(ind) > 1][:remaining_size]
        unique_elements.extend(duplicate_elements)

    return unique_elements[:size]

def pop_compare(ind1, ind2):
    # List all available primitive types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)
    for idx, node in enumerate(ind1[1:],1):
        types1[node.ret].append(idx)
    for idx, node in enumerate(ind2[1:],1):
        types2[node.ret].append(idx)
    return types1==types2

def remove_duplicates(population):
    return list(set(population))

def varAnd(population, toolbox, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param elitpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.

    The variation goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.  A
    first loop over :math:`P_\mathrm{o}` is executed to mate pairs of
    consecutive individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting :math:`P_\mathrm{o}`
    is returned.

    This variation is named *And* beceause of its propention to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematicaly, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """
    offspring = [toolbox.clone(ind) for ind in population]
    new_cxpb=cxpb/(cxpb+mutpb)

    #num_cx=int(new_cxpb*len(offspring))
    #num_mu=len(offspring)-num_cx
    #print(new_cxpb, new_mutpb)
    # Apply crossover and mutation on the offspring
    i = 1
    while i < len(offspring):
        if random.random() < new_cxpb:
            if (offspring[i - 1] == offspring[i]) or pop_compare(offspring[i - 1], offspring[i]):
                offspring[i - 1], = toolbox.mutate(offspring[i - 1])
                offspring[i], = toolbox.mutate(offspring[i])
            else:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            i = i + 2
        else:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
            i = i + 1
    return offspring
from collections import Counter
# def delete_specific(toolbox, pop, del_size):
#     pop_counter = Counter(toolbox.clone(ind) for ind in pop)
#     non_duplicated_elements = [item for item in pop_counter if pop_counter[item] == 1]
#     duplicated_elements = [item for item in pop_counter if pop_counter[item] > 1]
#
#     if del_size > 0:
#         if del_size <= len(duplicated_elements):
#             duplicated_elements = duplicated_elements[:-del_size]
#         else:
#             print("Warning: The number of duplicated items to be deleted exceeds the total number of duplicates.")
#             duplicated_elements = []
#
#     return non_duplicated_elements + duplicated_elements

def delete_specific(toolbox, pop, num_to_delete):
    pop_dict = {i: pop.count(i) for i in pop}
    pop_dict = {k: v for k, v in sorted(pop_dict.items(), key=lambda item: item[1], reverse=True)}

    del_elements = []
    for ind, count in pop_dict.items():
        if count > 1 and len(del_elements) < num_to_delete:
            del_elements.append(ind)
        if len(del_elements) >= num_to_delete:
            break

    new_pop = [ind for ind in pop if ind not in del_elements]

    return new_pop
def Other(ac, length):
    # 按照字典的值进行排序，并将其转换为列表形式
    sorted_items = sorted(ac.items(), key=lambda x: x[1])
    # 选择一定比例的个体
    select_pop = int(length * 0.06) if length * 0.06 >= 1 else 1
    # 取排序后的列表中的前 select_pop 个值的键
    sorted_keys = [k for k, v in sorted_items[:select_pop]]

    return sorted_keys

def One(ac, length):
    # 按照字典的值进行排序，并将其转换为列表形式
    sorted_items = sorted(ac.items(), key=lambda x: x[1])
    # 选择一定比例的个体
    select_pop = int(length * 0.1) if length * 0.1 >= 1 else 1
    # 取排序后的列表中的前 select_pop 个值的键
    sorted_keys = [k for k, v in sorted_items[:select_pop]]

    return sorted_keys
    # for gen in range(1, ngen + 1):
    #     def create_new_individual():
    #         new_individual = toolbox.population(1)[0]
    #         new_individual.fitness.values = toolbox.evaluate(individual=new_individual, hof=cop_po)
    #         return new_individual
    #     # Select the next generation individuals by elitism
    #     elitismNum = int(elitpb * len(population))
    #     population_for_eli = [toolbox.clone(ind) for ind in population]
    #     offspringE = toolbox.selectElitism(population_for_eli, k=elitismNum)
    #     qbest.append(halloffame[0].fitness.values[-1])
    #     if int(qbest[gen]) <= int(qbest[gen - 1]):
    #         a = a + 1
    #     else:
    #         a = 0
    #     # Select the next generation individuals for crossover and mutation
    #     offspring = toolbox.select(population, len(population) - elitismNum)
    #     temp = [toolbox.clone(ind) for ind in population]
    #     temp = [item for item in temp if item not in offspring]
    #     Eliminate_Pop = temp
    #     tree_value = [ind.fitness.values for ind in Eliminate_Pop]
    #     Elim_Dict = dict(zip(Eliminate_Pop, tree_value))
    #     if gen==1:
    #        Elim_Pop = One(Elim_Dict, length)
    #     else:
    #        Elim_Pop = Other(Elim_Dict, length)
    #     for i in range(len(Elim_Pop)):
    #         Pop_Elim.append(Elim_Pop[i])
    #     Pop_ElimSize = len(Pop_Elim)
    #
    #     # tt_secpop = seconbestind(ab, length)
    #     # for i in range(len(Elim2_Pop)):
    #     #     pop_sectt.append(tt_secpop[i])
    #     # pop_secttsize = len(pop_sectt)
    #     if a == 5:
    #         if flag == 0:
    #             # 记录最大值
    #             strategy1_bestvalue = halloffame[0].fitness.values[-1]
    #             # 在加入存档的个体，满足原先的代价(初始个体多少个，最后的offspring就有多少个)。
    #             new_offspring = delete(toolbox, offspring, Pop_ElimSize)
    #             offspring = new_offspring + Pop_Elim
    #             Pop_Elim = []
    #             flag = flag + 1
    #             print("策略1生效")
    #         elif flag == 1:
    #             # 如果这段时间确实有发育的痕迹，那么继续用策略1.
    #             if strategy1_bestvalue < halloffame[0].fitness.values[-1]:
    #                 strategy1_bestvalue = halloffame[0].fitness.values[-1]
    #                 flag = 0
    #                 print("策略1持续生效")
    #             # else:
    #             #     new_offspring = delete(toolbox, offspring, pop_secttsize)
    #             #     offspring = new_offspring + pop_sectt
    #             #     pop_sectt = []
    #             #     flag = flag + 1
    #             #     print("策略2生效")
    #     # Vary the pool of individuals
    #     offspring = varAnd(offspring, toolbox, cxpb, mutpb)
    #     # Evaluate the individuals with an invalid fitness
    #     invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    #     # print(len(invalid_ind))
    #     for i in invalid_ind:
    #         i.fitness.values = toolbox.evaluate(individual=i, hof=cop_po)
    #     offspring[0:0] = offspringE
    #     offspring = [ind for ind in offspring if ind.fitness.values[0] != 0]
    #     if len(offspring) < len(population):
    #         num_to_add = len(population)- len(offspring)
    #         for _ in range(num_to_add):
    #             new_individual = create_new_individual()
    #             offspring.append(new_individual)
    #     # Update the hall of fame with the generated
    #     if halloffame is not None:
    #         halloffame.update(offspring)
    #     cop_po = offspring.copy()
    #     hof_store.update(offspring)
    #     for i in hof_store:
    #         cop_po.append(i)
    #     # 更新，但更新子树缓存的哈希表。
    #
    #
    #     population[:] = offspring
    #     # Append the current generation statistics to the logbook
    #     record = stats.compile(population) if stats else {}
    #     # print(record)
    #     logbook.record(gen=gen, nevals=len(offspring), **record)
    #     # print(record)
    #     if verbose:
    #         print(logbook.stream)


def eaSimple(population, toolbox, cxpb, mutpb, elitpb, ngen, randomseed, stats=None,
             halloffame=None, verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param etilpb: The probability of elitism
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evalutions for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            elitismNum
            offspringE=selectElitism(population,elitismNum)
            population = select(population, len(population)-elitismNum)
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            offspring=offspring+offspringE
            evaluate(offspring)
            population = offspring.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` and :meth::`toolbox.selectElitism`,
     aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    # Evaluate the individuals with an invalid fitness
    # invalid_ind = [ind for ind in population if not ind.fitness.valid]
    # print(len(invalid_ind))
    # 缓存个体时，返回了个体的适应度值。但是没有存储子树。必须在算法开始阶段存储哈希表。
    # 在这里要加参数，并且多返回值。返回值为精度和子树哈希表。这样做，可能空间要大点，因为第一轮就存满了。
    sub_hof={}
    results_dict2={}
    key = operator.attrgetter("height")
    for i in population:
        i.fitness.values = toolbox.evaluate(individual=i, hof=[])
    if halloffame is not None:
        halloffame.update(population)
    hof_store = tools.HallOfFame(5 * len(population))
    hof_store.update(population)
    cop_po = population
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)
    qbest = ['0']
    a = 0
    #  先对字典排序,返回个体。
    strategy1_bestvalue = 0
    flag = 0
    length = len(population)
    Pop_Elim = []
    best_individuals = []
    improved_individuals = []
    bsgt=[]
    c=0
    last_best_fitness = None
    results_dict = {}
    for gen in range(1, ngen + 1):
        def create_new_individual():
            new_individual = toolbox.population(1)[0]
            new_individual.fitness.values = toolbox.evaluate(individual=new_individual, hof=cop_po)
            return new_individual

        elitismNum = int(elitpb * len(population))
        population_for_eli = [toolbox.clone(ind) for ind in population]
        offspringE = toolbox.selectElitism(population_for_eli, k=elitismNum)
        qbest.append(halloffame[0].fitness.values[-1])

        # if int(qbest[gen]) <= int(qbest[gen - 1]):
        #     a = a + 1
        if last_best_fitness is not None and last_best_fitness == qbest[gen]:
            a += 1
        else:
            a = 0
        if flag == 1:
            offspring = toolbox.selecttwo(population, len(population) - elitismNum)
        else:
            offspring = toolbox.select(population, len(population) - elitismNum)
        temp = [toolbox.clone(ind) for ind in population]
        temp = [item for item in temp if item not in offspring]
        Eliminate_Pop = temp
        tree_value = [ind.fitness.values for ind in Eliminate_Pop]
        Elim_Dict = dict(zip(Eliminate_Pop, tree_value))

        if gen == 1:
            Elim_Pop = One(Elim_Dict, length)
        else:
            Elim_Pop = Other(Elim_Dict, length)

        for i in range(len(Elim_Pop)):
            Pop_Elim.append(Elim_Pop[i])
        print(gen, len(Pop_Elim))
        if a == 5:
            if flag == 0:
                duplicate_count = len([ind for ind in offspring if offspring.count(ind) > 1])
                add_size = int(0.25 * duplicate_count)
                print(add_size)
                add_elements = Pop_Elim[:add_size]
                strategy1_bestvalue = halloffame[0].fitness.values[-1]
                offspring = delete_specific(toolbox, offspring, add_size)
                offspring = offspring + add_elements
                for ind in offspring:
                    ind.initial_fitness = ind.fitness.values[-1]
                Pop_Elim = Pop_Elim[add_size:]
                flag = flag + 1
                if gen>29:
                   bsgt.append(halloffame[0])
                print("策略1生效")
                c=1
                a=0
            elif flag == 1:
                if strategy1_bestvalue < halloffame[0].fitness.values[-1]:
                    strategy1_bestvalue = halloffame[0].fitness.values[-1]
                    flag = 0
                    bsgt.append(halloffame[0])
                    print("策略1下一轮继续生效")
                    a=0
                else:
                    # 如果种群适应度没有进一步提升，从improved_individuals选择一些个体加入种群
                    if len(improved_individuals) > 0:
                        duplicate_count = len([ind for ind in offspring if offspring.count(ind) > 1])
                        num_to_replace = min(int(0.20 * duplicate_count), len(improved_individuals))

                        # 用更新成功的个体替换部分重复的个体
                        duplicate_individuals = [ind for ind in offspring if offspring.count(ind) > 1]
                        bsgt.append(halloffame[0])
                        if duplicate_individuals:
                            for i in range(num_to_replace):
                                replace_individual = random.choice(duplicate_individuals)
                                duplicate_individuals.remove(replace_individual)  # 先从 duplicate_individuals 移除
                                offspring.remove(replace_individual)  # 然后从 offspring 移除
                                offspring.append(improved_individuals[i])
                            print("策略2生效")
                            print(f"替换了 {num_to_replace} 个个体")
                        else:
                            print("策略2未生效，没有重复个体")
                        # 如果更新成功的个体数量不足以替换所有重复的个体
                    c=1
                    a=0
                    flag=0
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for i in invalid_ind:
            i.fitness.values = toolbox.evaluate(individual=i, hof=cop_po)
        if c==1:
            for ind in offspring:
                if hasattr(ind, 'initial_fitness') and ind.fitness.values[-1] > ind.initial_fitness:
                    print(f"新个体 {ind} 提升了适应度值。")
                    improved_individuals.append(ind)
            c=0
        offspring[0:0] = offspringE
        offspring = [ind for ind in offspring if ind.fitness.values[0] != 0]

        if len(offspring) < len(population):
            num_to_add = len(population) - len(offspring)
            for _ in range(num_to_add):
                new_individual = create_new_individual()
                offspring.append(new_individual)

        if halloffame is not None:
            halloffame.update(offspring)

        best_ind = tools.selBest(offspring, 1)[0]
        # if best_ind.fitness.values[0] >= 100:
        #     print("Evolution stopped: fitness reached 100.")
        #     break

        cop_po = offspring.copy()
        hof_store.update(offspring)
        for i in hof_store:
            cop_po.append(i)

        population[:] = offspring

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(offspring), **record)

        if verbose:
            print(logbook.stream)
        if gen >= 25:
            top2_ind = tools.selBest(population, 1)
            best_individuals.extend(top2_ind)
            fitness_values = top2_ind[0].fitness.values
            results_dict[gen] = {'top2_ind': top2_ind, 'fitness_values': fitness_values}


    def remove_duplicates(population):
        unique_population = []
        for ind in population:
            if ind not in unique_population:
                unique_population.append(ind)
        return unique_population
    population=remove_duplicates(population)
    # print(len(population))
    # top10_individuals = tools.selBest(population, 20)
    # Sort the population in descending order based on fitness values

    sorted_population = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)
    # Get the top 20 individuals
    top20_individuals = sorted_population[:10]

    for i in top20_individuals:
        fitness_values = i.fitness.values
        results_dict2[i] = {'top2_ind': i, 'fitness_values': fitness_values}
    print(results_dict2)

    return top20_individuals,logbook,best_individuals,bsgt

