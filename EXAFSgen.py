import larch
import random
from larch_plugins.xafs import feffdat
from larch import Interpreter
import operator 
import numpy as np
from operator import itemgetter
from larch_plugins.io import read_ascii
from larch_plugins.xafs import autobk

mylarch = Interpreter()

#range for rdm num generator
rangeA = (np.linspace(5,150,146) * 0.01).tolist()
rangeB = (np.linspace(-500,500,1001) * 0.01).tolist()
rangeC = (np.linspace(-20,20,41) * 0.01).tolist()
rangeD = (np.linspace(0,35,36) * 0.001).tolist()
rangeA.append(0)
#rangeB.append(0)
#rangeC.append(0)
#rangeD.append(0)

front = '/Users/42413/Documents/GitHub/EXAFS/Cu Data/path Data/feff'
end = '.dat'

def fitness(indi,exp):
    loss = 0
    z =[]
    for i in range(1,103):
        if i < 10:
            filename = front+'000'+str(i)+end
            #print(filename)
        elif i< 100:
            filename = front+'00'+str(i)+end
            #print(filename)
        else:
            filename = front+'0'+str(i)+end
            #print(filename)
        path=feffdat.feffpath(filename, s02=str(indi[i][0]), e0=str(indi[i][1]), sigma2=str(indi[i][2]), deltar=str(indi[i][3]), _larch=mylarch)
        feffdat._path2chi(path, _larch=mylarch)
        y = path.chi
        yTotal = [0]*(len(y)+1)
        for k in range(len(y)):
            yTotal[k] += y[k]
    for j in range(len(yTotal)):
        loss = loss + abs(yTotal[j] - exp[j])
    return loss

def generateACombo():
    a = random.choice(rangeA)
    b = random.choice(rangeB)
    c = random.choice(rangeC)
    d = random.choice(rangeD)
    return [a,b,c,d]
    
def generateIndi():
    indi = []
    for i in range(103):
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
        #print("++",individual)
        individualTuple = tuple(tuple(x) for x in individual)
        #print(individualTuple)
        populationPerf[individualTuple] = fitness(individual, exp)
    return sorted(populationPerf.items(), key = operator.itemgetter(1), reverse=True)

def selectFromPopulation(populationSorted, best_sample, lucky_few):
    nextGeneration = []
    print("Best Fit:",fitness(populationSorted[0], exp))
    for i in range(best_sample):
        #print(len(populationSorted))
        nextGeneration.append(populationSorted[i])
        #print("best: ",populationSorted[i][0])
        #print()
    for i in range(lucky_few):
        j = random.randint(0,len(populationSorted))
        nextGeneration.append(populationSorted[j])
        #print("lucky: ",populationSorted[j][0])
        #print()
    random.shuffle(nextGeneration)
    return nextGeneration

def createChild(individual1, individual2):
    child = []
    for i in range(len(individual1)):
        if (int(100 * random.random()) < 50):
            child.append(individual1[i][0:2] + individual2[i][2:4])
        else:
            child.append(individual2[i][0:2] + individual1[i][2:4])
    #print("CHILD:",child)
    return child

def createChildren(breeders, number_of_child):
    nextPopulation = []
    #print(len(breeders))
    for i in range(int(len(breeders)/2)):
        for j in range(number_of_child):
            #print("a child")
            nextPopulation.append(createChild(breeders[i], breeders[len(breeders) -1 -i]))
    return nextPopulation

def mutateIndi(indi):
    print("Mutate!")
    indi = generateIndi()
    return indi

def mutatePopulation(population, chance_of_mutation):
	for i in range(len(population)):
		if random.random() * 100 < chance_of_mutation:
			population[i] = mutateIndi(population[i])
	return population

def nextGeneration (firstGeneration, exp, best_sample, lucky_few, number_of_child, chance_of_mutation):
    global genNum
    genNum+=1
    print("Gen:", genNum)
    populationTupleSorted = computePerfPop(firstGeneration, exp)
    newIndi = []
    populationSorted = []
    #print((populationTupleSorted))
    for indi in populationTupleSorted:
        for combo in indi[0]:
            #print("--",combo)
            newIndi.append(list(combo))
        populationSorted.append(newIndi)
    #print(populationSorted)
    nextBreeders = selectFromPopulation(populationSorted, best_sample, lucky_few)
    nextPopulation = createChildren(nextBreeders, number_of_child)
    nextGeneration = mutatePopulation(nextPopulation, chance_of_mutation)
    return nextGeneration

def multipleGeneration(number_of_generation, exp, size_population, best_sample, lucky_few, number_of_child, chance_of_mutation):
	historic = []
	historic.append(generateFirstGen(size_population))
	for i in range (number_of_generation):
		historic.append(nextGeneration(historic[i], exp, best_sample, lucky_few, number_of_child, chance_of_mutation))
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

g = read_ascii('/Users/42413/Documents/GitHub/EXAFS/Cu Data/cu_10k.xmu', _larch = mylarch)
autobk(g, rbkg=1.45, _larch = mylarch)
exp = g.chi
genNum = 0
size_population = 800
best_sample = 20
lucky_few = 20
number_of_child = 40
number_of_generation = 50
chance_of_mutation = 5

if ((best_sample + lucky_few) / 2 * number_of_child != size_population):
	print ("population size not stable")
else:
	historic = multipleGeneration(number_of_generation, exp, size_population, best_sample, lucky_few, number_of_child, chance_of_mutation)
	
	printSimpleResult(historic, exp, number_of_generation)































