import larch
import random
from larch_plugins.xafs import feffdat
from larch import Interpreter
import operator 
import numpy as np
from operator import itemgetter
from larch_plugins.io import read_ascii
from larch_plugins.xafs import autobk
import datetime
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

mylarch = Interpreter()

#range for rdm num generator
rangeA = (np.linspace(5,150,146) * 0.01).tolist()
rangeB = (np.linspace(-500,500,1001) * 0.01).tolist()
rangeC = (np.linspace(-20,20,41) * 0.01).tolist()
rangeD = (np.linspace(1,15,15) * 0.001).tolist()
rangeA.append(0)
#rangeB.append(0)
#rangeC.append(0)
#rangeD.append(0)

front = 'Cu Data/path Data/feff'
end = '.dat'

def fitness(indi,exp):
    loss = 0
    yTotal = [0]*(401)
    for i in range(1,10):
        if i < 10:
            filename = front+'000'+str(i)+end
        elif i< 100:
            filename = front+'00'+str(i)+end
        else:
            filename = front+'0'+str(i)+end
        path=feffdat.feffpath(filename, s02=str(indi[i-1][0]), e0=str(indi[i-1][1]), sigma2=str(indi[i-1][2]), deltar=str(indi[i-1][3]), _larch=mylarch)
        feffdat._path2chi(path, _larch=mylarch)
        y = path.chi
        for k in range(len(y)):
            yTotal[k] += y[k]
    global g
    interval = (np.linspace(0,400,401)).tolist()
    intervalInt = [int(i) for i in interval]
    rdmInterval = random.sample(intervalInt,100)
    for j in rdmInterval:
        loss = loss + (yTotal[j]*g.k[j]**2 - exp[j]*g.k[j]**2)**2
    return loss

def generateACombo():
    a = random.choice(rangeA)
    global b
    c = random.choice(rangeC)
    d = random.choice(rangeD)
    return [a,b,c,d]
    
def generateIndi():
    indi = []
    for i in range(10):
        indi.append(generateACombo())
    return indi

def generateFirstGen(genSize):
    gen = []
    i = 0
    while i < genSize:
        gen.append(generateIndi())
        i+=1
    return gen

def computePerfPop(pop,exp):
    populationPerf = {}
    for individual in pop:
        individualTuple = tuple(tuple(x) for x in individual)
        populationPerf[individualTuple] = fitness(individual, exp)
    return sorted(populationPerf.items(), key = operator.itemgetter(1), reverse=False)

def selectFromPopulation(populationSorted, best_sample, lucky_few):
    nextBreeders = []
#    fitSum = 0
#    current = 0
#    for indi in populationSorted:
#        fitSum += indi[1]
#    pick = random.uniform(0, fitSum)
#    for indi in populationSorted:
#        current += indi[1]
#        if current < pick:
#            newIndi = []
#            for combo in indi[0]:
#                newIndi.append(list(combo))
#            nextBreeders.append(newIndi)
    for i in range(best_sample):
        nextBreeders.append(populationSorted[i])
    for i in range(lucky_few):
        j = random.randint(best_sample,len(populationSorted)-1)
        nextBreeders.append(populationSorted[j])
    random.shuffle(nextBreeders)
    return nextBreeders

def createChild(individual1, individual2):
    child = []
    global diffCounter
    global chance_of_mutation
    if diffCounter > 10:
        print("******************Different Breeding******************")
        chance_of_mutation = 40
        diffCounter = 0
        for i in range(len(individual1)):
            if (int(100 * random.random()) < 50):
                child.append(individual1[i][0:2] + individual2[i][2:4])
            else:
                child.append(individual2[i][0:2] + individual1[i][2:4])
    else:
        chance_of_mutation = 20
        for i in range(len(individual1)):
            j = random.randint(0,1)
            if j == 0:
                child.append(individual1[i])
            elif j == 1:
                child.append(individual2[i])
    return child

def createChildren(breeders, number_of_child):
    nextPopulation = []
    for i in range(int(len(breeders)/2)):
        for j in range(number_of_child):
#            b1 = random.randint(0,len(breeders)-1)
#            b2 = random.randint(0,len(breeders)-1)
#            nextPopulation.append(createChild(breeders[b1],breeders[b2]))
            nextPopulation.append(createChild(breeders[i], breeders[len(breeders) -1 -i]))
    return nextPopulation

def mutateIndi(indi):
    indi = generateIndi()
    return indi

def mutatePopulation(population, chance_of_mutation, chance_of_mutation_e0):
    mutateTime = 0
    for i in range(len(population)):
        if random.random() * 100 < chance_of_mutation:
            mutateTime+=1
            population[i] = mutateIndi(population[i])
    if random.random() * 100 < chance_of_mutation_e0:
        e0 = random.choice(rangeB)
        print("Mutate e0 to:", e0)
        for i in range(len(population)):
            for j in population[i]:
                j[1] = e0
    print("Mutate Times:", mutateTime)
    return population

def nextGeneration (firstGeneration, exp, best_sample, lucky_few, number_of_child, chance_of_mutation, chance_of_mutation_e0):
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    global genNum
    global historyBest
    global bestDiff
    global diffCounter
    genNum+=1
    print(st)
    print("Gen:", genNum)
    populationTupleSorted = computePerfPop(firstGeneration, exp)
    populationSorted = []
    print("Best Fit:", populationTupleSorted[0][1])
    for indi in populationTupleSorted:
        newIndi = []
        for combo in indi[0]:
            #print("--",combo)
            newIndi.append(list(combo))
        populationSorted.append(newIndi)
    bestDiff = abs(populationTupleSorted[0][1]-historyBest)
    historyBest = populationTupleSorted[0][1]
    
    
    if bestDiff < 0.1:
        diffCounter += 1
    else:
        diffCounter = 0
    print("2nd Fit:",populationTupleSorted[1][1])
    print("3rd Fit:",populationTupleSorted[2][1])
    print("4th Fit:",populationTupleSorted[3][1])
    print("Last Fit:",populationTupleSorted[len(populationTupleSorted)-1][1])
    print("Different from last best fit:",bestDiff)
    print("Best fit combination:\n",populationTupleSorted[0][0])
#    if genNum%1 == 0:
#        print("Best fit combination:\n",populationTupleSorted[0][0])
#        indi = populationTu-pleSorted[0][0]
#        yTotal = [0]*(401)
##        lenY = 0
#        for i in range(1,10):
#            if i < 10:
#                filename = front+'000'+str(i)+end
#            elif i< 100:
#                filename = front+'00'+str(i)+end
#            else:
#                filename = front+'0'+str(i)+end
#            path=feffdat.feffpath(filename, s02=str(indi[i-1][0]), e0=str(indi[i-1][1]), sigma2=str(indi[i-1][2]), deltar=str(indi[i-1][3]), _larch=mylarch)
#            feffdat._path2chi(path, _larch=mylarch)
#            y = path.chi
##            lenY = len(y)
#            for k in range(len(y)):
#                yTotal[k] += y[k]
        
#        global g
##        for m in range(lenY):
##            yTotal[m] = yTotal[m]*g.k**2
#        global global_yTotal
#        plt.plot(g.k, g.chi*g.k**2)
#        plt.plot(g.k[0:401], yTotal*g.k[0:401]**2)
##        plt.ylim(top=6, bottom=-6)
#        plt.show()
        
        
        
#        file.write("Gen Num: %d" % genNum)
#        file.write("Fitness:"+str(populationTupleSorted[0][1]))
#        file.write("Combination:"+str(populationTupleSorted[0][0]))
    
    nextBreeders = selectFromPopulation(populationSorted, best_sample, lucky_few)
    nextPopulation = createChildren(nextBreeders, number_of_child)
    
    lenDiff = len(firstGeneration)-len(nextPopulation)

    for i in range(lenDiff):
        j = random.randint(0,len(firstGeneration)-1)
        nextPopulation.append(firstGeneration[j])
    print(len(nextPopulation))
    
    nextGeneration = mutatePopulation(nextPopulation, chance_of_mutation, chance_of_mutation_e0)
    return nextGeneration

def multipleGeneration(number5_of_generation, exp, size_population, best_sample, lucky_few, number_of_child, chance_of_mutation, chance_of_mutation_e0):
    historic = []
    historic.append(generateFirstGen(size_population))
    for i in range (number_of_generation):
        historic.append(nextGeneration(historic[i], exp, best_sample, lucky_few, number_of_child, chance_of_mutation, chance_of_mutation_e0))
    return historic
 
#printing tool - NOT DONE!!!!!!!!!!!!!
def printSimpleResult(historic, exp, number_of_generation): #bestSolution in historic. Caution not the last
	result = getListBestIndividualFromHistorique(historic, exp)[number_of_generation-1]
	print ("solution: \"" + result[0] + "\" de fitness: " + str(result[1]))
    
#analysis tools
def getBestIndividualFromPopulation (population, exp):
	return computePerfPop(population, exp)[0]

def getListBestIndividualFromHistorique (historic, exp):
	bestIndividuals = []
	for population in historic:
		bestIndividuals.append(getBestIndividualFromPopulation(population, exp))
	return bestIndividuals

#main program
    

file = open("Result.txt","w")
g = read_ascii('Cu Data/cu_10k.xmu', _larch = mylarch)
autobk(g, rbkg=1.45, _larch = mylarch)
exp = g.chi
#kidNum = 0
genNum = 0
size_population = 1000
best_sample = 200
lucky_few = 200
number_of_child = 4
number_of_generation = 1000
chance_of_mutation = 20
chance_of_mutation_e0 = 0
historyBest = 0
bestDiff = 9999
diffCounter = 0
#e0
#b = random.choice(rangeB)
b = 1.86
if ((best_sample + lucky_few) / 2 * number_of_child >= size_population):
	print ("population size not stable")
else:
	historic = multipleGeneration(number_of_generation, exp, size_population, best_sample, lucky_few, number_of_child, chance_of_mutation, chance_of_mutation_e0)
	
	printSimpleResult(historic, exp, number_of_generation)































