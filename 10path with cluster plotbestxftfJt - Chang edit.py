import larch
import random
from larch_plugins.xafs import feffdat
from larch_plugins.xafs import xftf
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
rangeA = (np.linspace(5,95,91) * 0.01).tolist()
rangeB = (np.linspace(-100,100,201) * 0.01).tolist()
largerRangeB = (np.linspace(-600,600,1201) * 0.01).tolist()
rangeC = (np.linspace(1,15,15) * 0.001).tolist()
rangeD = (np.linspace(-20,20,41) * 0.01).tolist()

rangeA.append(0)
#rangeB.append(0)
#rangeC.append(0)
#rangeD.append(0)

front = '/Users/XuChang/Downloads/EXAFS-master/Cu Data/path Data/feff'
end = '.dat'

intervalK = (np.linspace(80,340,261)).tolist()
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
        for k in intervalK:
            yTotal[int(k)] += y[int(k)]
    global g
    for j in intervalK:
        #loss = loss + (yTotal[int(j)]*g.k[int(j)]**2 - exp[int(j)]*g.k[int(j)]**2)**2
        loss = loss + abs(yTotal[int(j)]*g.k[int(j)]**2 - exp[int(j)]*g.k[int(j)]**2)*(abs(exp[int(j)]*g.k[int(j)]**2))**(0.5)
    return loss

#changg
def fourierFitness(indi,exp):
    loss = 0
#    yTotal = [0]*(401)
    chir_magTotal = [0]*(326)
    for i in range(1,10):
        if i < 10:
            filename = front+'000'+str(i)+end
        elif i< 100:
            filename = front+'00'+str(i)+end
        else:
            filename = front+'0'+str(i)+end
        path=feffdat.feffpath(filename, s02=str(indi[i-1][0]), e0=str(indi[i-1][1]), sigma2=str(indi[i-1][2]), deltar=str(indi[i-1][3]), _larch=mylarch)
        feffdat._path2chi(path, _larch=mylarch)
        SumGroup = path
        SumGroup.chi = path.chi
        SumGroup.k = path.k
        xftf(SumGroup.k, SumGroup.chi, kmin=3, kmax=17, dk=4, window='hanning', kweight=2, group=SumGroup, _larch=mylarch)
         
#        y = path.chi
#        for k in intervalK:
#            yTotal[int(k)] += y[int(k)]
#            
        for kk in range(0, len(SumGroup.r)):
            chir_magTotal[int(kk)] += SumGroup.chir_mag[int(kk)]
#    global g
    for j in range(0, len(SumGroup.r)):
        loss = loss + (chir_magTotal[int(j)] - exp[int(j)])**2
        #loss = loss + abs(yTotal[int(j)]*g.k[int(j)]**2 - exp[int(j)]*g.k[int(j)]**2)*(abs(exp[int(j)]*g.k[int(j)]**2))**(0.5)

#    print("[lose is", loss, "]")
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
    global MetaDictionary
    
#    #changg
#    global sortedFourier
    
    for individual in pop:    
        #counter for dictionary
        n = 1
        individualTuple = tuple(tuple(x) for x in individual)
        fit = fitness(individual, exp)
        
#        #changg
#        fourierFit = fourierFitness(individual, g.chir_mag)
        
        #Meta Dic
        for combo in individual:
            if n not in MetaDictionary:
                MetaDictionary[n] = {"A":{},"C":{},"D":{}}
            
            if (combo[0] not in MetaDictionary[n]["A"]):
                MetaDictionary[n]["A"][combo[0]] = fit
            elif fit < MetaDictionary[n]["A"][combo[0]]:
                MetaDictionary[n]["A"][combo[0]] = fit
                
            if (combo[2] not in MetaDictionary[n]["C"]):
                MetaDictionary[n]["C"][combo[2]] = fit
            elif fit < MetaDictionary[n]["C"][combo[2]]:
                MetaDictionary[n]["C"][combo[2]] = fit
                
            if (combo[3] not in MetaDictionary[n]["D"]):
                MetaDictionary[n]["D"][combo[3]] = fit
            elif fit < MetaDictionary[n]["D"][combo[3]]:
                MetaDictionary[n]["D"][combo[3]] = fit
            n += 1
        #Sorted Dic
        
        populationPerf[individualTuple] = fit
    return sorted(populationPerf.items(), key = operator.itemgetter(1), reverse=False)

#changg
def computeFourierPerfPop(pop,exp):
    populationPerf = {}
#    global MetaDictionary
    
    #changg
    global sortedFourier
    
    for individual in pop:    
        #counter for dictionary
#        n = 1
        individualTuple = tuple(tuple(x) for x in individual)
        
        #changg
#        fit = fitness(individual, exp)
        fourierFit = fourierFitness(individual, g.chir_mag)
        
        #Meta Dic
#        for combo in individual:
#            if n not in MetaDictionary:
#                MetaDictionary[n] = {"A":{},"C":{},"D":{}}
#            
#            if (combo[0] not in MetaDictionary[n]["A"]):
#                MetaDictionary[n]["A"][combo[0]] = fit
#            elif fit < MetaDictionary[n]["A"][combo[0]]:
#                MetaDictionary[n]["A"][combo[0]] = fit
#                
#            if (combo[2] not in MetaDictionary[n]["C"]):
#                MetaDictionary[n]["C"][combo[2]] = fit
#            elif fit < MetaDictionary[n]["C"][combo[2]]:
#                MetaDictionary[n]["C"][combo[2]] = fit
#                
#            if (combo[3] not in MetaDictionary[n]["D"]):
#                MetaDictionary[n]["D"][combo[3]] = fit
#            elif fit < MetaDictionary[n]["D"][combo[3]]:
#                MetaDictionary[n]["D"][combo[3]] = fit
#            n += 1
        #Sorted Dic
        
        populationPerf[individualTuple] = fourierFit
    return sorted(populationPerf.items(), key = operator.itemgetter(1), reverse=False)


#def computePerfPop(pop,exp):
#    populationPerf = {}
#    for individual in pop:
#        individualTuple = tuple(tuple(x) for x in individual)
#        populationPerf[individualTuple] = fitness(individual, exp)
#    return sorted(populationPerf.items(), key = operator.itemgetter(1), reverse=False)

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
    global original_chance_of_mutation

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
        chance_of_mutation = original_chance_of_mutation
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
        global b
        e0 = random.choice(rangeB)
        b = e0
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
    
    #changg
    populationFourierTupleSorted = computeFourierPerfPop(firstGeneration, exp)
    
    populationSorted = []
    print("Best Fit:", populationTupleSorted[0][1])
    
   
#    print("[")
#    for ji in populationFourierTupleSorted:
#        print(ji[1]) 
#    print("]")
    
    
    for indi in populationTupleSorted:
        newIndi = []
        for combo in indi[0]:
            #print("--",combo)
            newIndi.append(list(combo))
        populationSorted.append(newIndi)
    bestDiff = abs(populationTupleSorted[0][1]-historyBest)
    historyBest = populationTupleSorted[0][1]
    global bestBest
    global best
    global bestFitIndi
    #if this indi is better than the history best
    if historyBest < bestBest:
        bestBest = historyBest
        bestFitIndi = populationTupleSorted[0][0]
        global bestYTotal
        bestYTotal = [0]*(401)
        
        global bestChir_magTotal
        bestChir_magTotal = [0]*(326)
        
        
#        lenY = 0
        for i in range(1,10):
            if i < 10:
                filename = front+'000'+str(i)+end
            elif i< 100:
                filename = front+'00'+str(i)+end
            else:
                filename = front+'0'+str(i)+end
            path=feffdat.feffpath(filename, s02=str(bestFitIndi[i-1][0]), e0=str(bestFitIndi[i-1][1]), sigma2=str(bestFitIndi[i-1][2]), deltar=str(bestFitIndi[i-1][3]), _larch=mylarch)
            feffdat._path2chi(path, _larch=mylarch)
            y = path.chi
#            lenY = len(y)
            for k in intervalK:
                bestYTotal[int(k)] += y[int(k)]
                
        '''Jeff'''
        best.chi = bestYTotal
        best.k = path.k
        xftf(best.k, best.chi, kmin=3, kmax=17, dk=4, window='hanning', kweight=2, group=best, _larch=mylarch)
        #whats the interval?? 326.
        #for k in range(0, 326, 1):
        #bestChir_magTotal[k] += path.chir_mag[k]
            
        '''Jeff end'''
        
    if bestDiff < 0.1:
        diffCounter += 1
    else:
        diffCounter = 0
    print("2nd Fit:",populationTupleSorted[1][1])
    print("3rd Fit:",populationTupleSorted[2][1])
    print("4th Fit:",populationTupleSorted[3][1])    
    print("Last Fit:",populationTupleSorted[len(populationTupleSorted)-1][1])

    #changg
    print("~~~~~~\nfourier Best Fit:", populationFourierTupleSorted[0][1])
    print("fourier 2nd Fit:", populationFourierTupleSorted[1][1])
    print("fourier 3rd Fit:", populationFourierTupleSorted[2][1])
    print("fourier 4th Fit:", populationFourierTupleSorted[3][1])
    print("fourier Last Fit:", populationFourierTupleSorted[len(populationFourierTupleSorted)-1][1])
    print("~~~~~~")

    print("Different from last best fit:",bestDiff)
    print("Best fit combination:\n",populationTupleSorted[0][0])
    print("History Best:",bestBest)
    print("History Best Indi:\n",bestFitIndi)
    if genNum%1 == 0:
#        print("Best fit combination:\n",populationTupleSorted[0][0])
        indi = populationTupleSorted[0][0]
        yTotal = [0]*(401)
        
        chir_magTotal = [0]*(326)
        
        
#        lenY = 0
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
#            lenY = len(y)
            for k in intervalK:
                yTotal[int(k)] += y[int(k)]
                
        '''chang'''
        SumGroup = path
        SumGroup.chi = yTotal
        SumGroup.k = path.k
        xftf(SumGroup.k, SumGroup.chi, kmin=3, kmax=17, dk=4, window='hanning', kweight=2, group=SumGroup, _larch=mylarch)
            
        '''chang end'''
        
        global g
#        for m in range(lenY):
#            yTotal[m] = yTotal[m]*g.k**2
        global global_yTotal
        plt.plot(g.k, g.chi*g.k**2)
        plt.plot(g.k[80:341], yTotal[80:341]*g.k[80:341]**2)
#        plt.ylim(top=6, bottom=-6)
        plt.show()
        plt.plot(g.k, g.chi*g.k**2)
        plt.plot(g.k[80:341], bestYTotal[80:341]*g.k[80:341]**2)
#        plt.ylim(top=6, bottom=-6)
        plt.show()
        
        '''chang: plot fourier'''
        plt.plot(g.r, g.chir_mag)
        plt.plot(SumGroup.r, SumGroup.chir_mag)
        plt.show()
        
        plt.plot(g.r, g.chir_mag)
        plt.plot(best.r, best.chir_mag)
        plt.show()
        
        
        '''chang end'''
        
        
        
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

def multipleGeneration(number_of_generation, exp, size_population, best_sample, lucky_few, number_of_child, chance_of_mutation, chance_of_mutation_e0):
    historic = []
    historic.append(generateFirstGen(size_population))
    #fix
    for i in range(int(number_of_generation/2)):
        historic.append(nextGeneration(historic[i], exp, best_sample, lucky_few, number_of_child, chance_of_mutation, chance_of_mutation_e0))
    global bestFitIndi
    newE0 = findE0(bestFitIndi,exp)
    global b
    b = newE0
    chance_of_mutation_e0 = 0
    #change all E0 to fixed
    for indi in historic[-1]:
        for combo in indi:
            combo[1] = newE0
    for i in range (int(number_of_generation/2), number_of_generation):
        historic.append(nextGeneration(historic[i], exp, best_sample, lucky_few, number_of_child, chance_of_mutation, chance_of_mutation_e0))
    #print plots for ACD
    global MetaDictionary
    for key in MetaDictionary:
        for attr in MetaDictionary[key]:
            print("path: ",str(key),", attribute: ",str(attr),"printed below:")
            listOfX = [x for x in MetaDictionary[key][attr]]
            listOfY = [y for y in MetaDictionary[key][attr].values()]
            plt.scatter(listOfX,listOfY)
            plt.show()
    return historic
 
#printing tool - NOT DONE!!!!!!!!!!!!!
def printSimpleResult(historic, exp, number_of_generation): #bestSolution in historic. Caution not the last
	result = getListBestIndividualFromHistorique(historic, exp)[number_of_generation-1]
	#print ("solution: \"" + result[0] + "\" de fitness: " + str(result[1]))
    
#analysis tools
def getBestIndividualFromPopulation (population, exp):
	return computePerfPop(population, exp)[0]

def getListBestIndividualFromHistorique (historic, exp):
	bestIndividuals = []
	for population in historic:
		bestIndividuals.append(getBestIndividualFromPopulation(population, exp))
	return bestIndividuals

#fitE0
def findE0(bestFit,exp):
    print("Finished First Half of Generation, Optimizing E0...")
    lowestX = 99999
    lowestY = 99999
    listOfx = []
    listOfy = []
    bestFitList = [list(x) for x in bestFit]
    for i in largerRangeB:
        for j in bestFitList:
            j[1] = i
        indi = tuple(tuple(x) for x in bestFitList)
        fit = fitness(indi,exp)
        if fit < lowestY:
            lowestY = fit
            lowestX = i
        listOfx.append(i)
        listOfy.append(fit)
    #print(listOfy)
    plt.plot(listOfx,listOfy)
    plt.show()
    print("Continue With E0 =",lowestX)
    return lowestX
#main program
    

file = open("Result.txt","w")
g = read_ascii('/Users/XuChang/Downloads/EXAFS-master/Cu Data/cu_10k.xmu', _larch = mylarch)
best = read_ascii('/Users/XuChang/Downloads/EXAFS-master/Cu Data/cu_10k.xmu', _larch = mylarch)
autobk(g, rbkg=1.45, _larch = mylarch)
autobk(best, rbkg=1.45, _larch=mylarch)

'''chang'''
xftf(g.k, g.chi, kmin=3, kmax=17, dk=4, window='hanning',
     kweight=2, group=g, _larch=mylarch)
xftf(best.k, best.chi, kmin=3, kmax=17, dk=4, window='hanning',
     kweight=2, group=best, _larch=mylarch) 
#show(g, _larch=mylarch)
'''chang end'''


exp = g.chi
#kidNum = 0
genNum = 0
size_population = 1000
best_sample = 400
lucky_few = 200
number_of_child = 3
#number of generations 
number_of_generation = 1000
chance_of_mutation = 20
original_chance_of_mutation = chance_of_mutation
chance_of_mutation_e0 = 20
historyBest = 0
bestDiff = 9999
diffCounter = 0
bestBest = 999999999
bestFitIndi = (())
bestChir_magTotal = [0]*(326)
bestYTotal = [0]*(401)
MetaDictionary = {}

#changg
sortedFourier = []

#range fixed at 10, change later
for i in range(10):
    bestFitIndi+=((0,0,0,0),)
#e0
b = random.choice(rangeB)
if ((best_sample + lucky_few) / 2 * number_of_child > size_population):
	print ("population size not stable")
else:
	historic = multipleGeneration(number_of_generation, exp, size_population, best_sample, lucky_few, number_of_child, chance_of_mutation, chance_of_mutation_e0)
	#"How much longer you gonna be on this?"
	printSimpleResult(historic, exp, number_of_generation)
