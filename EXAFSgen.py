import larch
import random
import numpy as np

def fitness(test, exp):
    
def generateACombo ():
    a = random.randrange(5, 150) * 0.01
    b = random.randrange(-500, 500) * 0.01
    c = random.randrange(-20, 20) * 0.01
    d = random.randrange(0, 35) * 0.001
    return [a,b,c,d]
    
def generateAPop():
    pop = []
    for i in 100:
        pop.append(generateAcombo())
    return pop

def generateFistGen(genSize):
    gen = []
    i = 0
    while i < genSize:
        gen.append(generateAPop())
        i+=1
    return gen

def computePerfPopulation(population, password):
	populationPerf = {}
	for individual in population:
		populationPerf[individual] = fitness(password, individual)
	return sorted(populationPerf.items(), key = operator.itemgetter(1), reverse=True)