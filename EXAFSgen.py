import larch
import random
from larch_plugins.xafs import feffdat
from larch import Interpreter
import operator 
import numpy as np
from operator import itemgetter

mylarch = Interpreter()

#range for rdm num generator
rangeA = (np.linspace(5,150,146) * 0.01).tolist()
rangeB = (np.linspace(-500,500,1001) * 0.01).tolist()
rangeC = (np.linspace(-20,20,41) * 0.01).tolist()
rangeD = (np.linspace(0,35,36) * 0.001).tolist()
rangeA.append(0)
rangeB.append(0)
rangeC.append(0)
rangeD.append(0)

def fitness(test,exp):
    loss = 0
    z =[]
    #for i in range(len(test)):
        #path=feffdat.feffpath('/Users/csp572/Desktop/Cu/Cu/Cu_3.61/feff0001.dat',test[i][0], test[i][1], test[i][2], test[i][3], _larch=mylarch)
        #y = path.chi
        
        #for j in range(len(y[i])):
            #loss = loss + abs(y[i][j] - exp[i][j])
    return #loss

def generateACombo():
    a = random.choice(rangeA)
    b = random.choice(rangeB)
    c = random.choice(rangeC)
    d = random.choice(rangeD)
    return [a,b,c,d]
    
def generateIndi():
    indi = []
    for i in range(100):
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
    return sorted(populationPerf.items(), key = operator.itemgetter(1), reverse=True)

def selectFromPopulation(populationSorted, best_sample, lucky_few):
	nextGeneration = []
	for i in range(best_sample):
		nextGeneration.append(populationSorted[i][0])
	for i in range(lucky_few):
		nextGeneration.append(random.choice(populationSorted)[0])
	random.shuffle(nextGeneration)
	return nextGeneration

def createChild(individual1, individual2):
	child = []
	for i in range(len(individual1)):
		if (int(100 * random.random()) < 50):
			child = individual1[i][0:2] + individual2[i][2:4]
		else:
			child = individual2[i][0:2] + individual1[i][2:4]
	return child

def createChildren(breeders, number_of_child):
	nextPopulation = []
	for i in range(len(breeders)/2):
		for j in range(number_of_child):
			nextPopulation.append(createChild(breeders[i], breeders[len(breeders) -1 -i]))
	return nextPopulation

def mutateIndi(indi):
    indi = generateIndi()
    return indi

def mutatePopulation(population, chance_of_mutation):
	for i in range(len(population)):
		if random.random() * 100 < chance_of_mutation:
			population[i] = mutateIndi(population[i])
	return population

def nextGeneration (firstGeneration, exp, best_sample, lucky_few, number_of_child, chance_of_mutation):
	 populationSorted = computePerfPop(firstGeneration, exp)
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
 
#printing tool
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
exp = []
size_population = 100
best_sample = 20
lucky_few = 20
number_of_child = 5
number_of_generation = 50
chance_of_mutation = 5

if ((best_sample + lucky_few) / 2 * number_of_child != size_population):
	print ("population size not stable")
else:
	historic = multipleGeneration(number_of_generation, exp, size_population, best_sample, lucky_few, number_of_child, chance_of_mutation)
	
	printSimpleResult(historic, exp, number_of_generation)































