import larch
import random
from larch_plugins.xafs import feffdat
from larch import Interpreter
import operator 

mylarch = Interpreter()

def fitness(test,exp):
    loss = 0
    for i in range(len(test)):
        path=feffdat.feffpath('/Users/csp572/Desktop/Cu/Cu/Cu_3.61/feff0001.dat',test[i][0], test[i][1], test[i][2], test[i][3], _larch=mylarch)
        y = path.chi
        for j in range(len(y[i])):
            loss = loss + abs(y[i][j] - exp[i][j])
    return loss

def generateACombo():
    a = random.randrange(5, 150) * 0.01
    b = random.randrange(-500, 500) * 0.01
    c = random.randrange(-20, 20) * 0.01
    d = random.randrange(0, 35) * 0.001
    return [a,b,c,d]
    
def generateIndi():
    pop = []
    for i in 100:
        pop.append(generateACombo())
    return pop

def generateFistGen(genSize):
    gen = []
    i = 0
    while i < genSize:
        gen.append(generateIndi())
        i+=1
    return gen

def computePerfPop(pop, exp):
	populationPerf = {}
	for individual in pop:
		populationPerf[individual] = fitness(individual, exp)
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
	child = ""
	for i in range(len(individual1)):
		if (int(100 * random.random()) < 50):
			child += individual1[i]
		else:
			child += individual2[i]
	return child



