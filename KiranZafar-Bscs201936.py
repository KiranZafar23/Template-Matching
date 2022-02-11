import matplotlib.image as img
import matplotlib.pyplot as plt
import matplotlib.patches as pacthes
import numpy as nump
import random

barriImage = nump.array(img.imread(r"F:\AI\groupGray.jpg"))
chotiImage = nump.array(img.imread(r"F:\AI\boothiGray.jpg"))


#initialising random population of hundred points from largeImage
def initializePopulation(barriImageRows,barriImageCols,populationSize):
    population = []
    for i in range(populationSize):
        population.append((random.randint(0,barriImageRows), random.randint(0, barriImageCols)))
    return population


#Evaluatting fitness values to determine how much small image matches with large image
def evaluateFitness(population,barriImage,chotiImage):
    chotiImageRows = chotiImage.shape[0]
    chotiImageCols = chotiImage.shape[1]
    co_rel = []
    for val in population:
        cropImage = barriImage[val[1]:val[1]+chotiImageRows, val[0]:val[0]+chotiImageCols]
        co_rel_value = corelationCoefficient(chotiImage, cropImage)
        co_rel.append(co_rel_value)
    return co_rel


#finding co_relation between small image and crop image from large image
def corelationCoefficient(chotiImage, cropImage):   
    result = (nump.mean((chotiImage - chotiImage.mean()) * (cropImage - cropImage.mean())))/(chotiImage.std() * cropImage.std())
    return result


#Selecting pairs in accordance with co-relation or matching parcentages
def selection(population,fitnessValues):
    selectionList = sorted(zip(fitnessValues, population), reverse=True)
    populationList = []
    fitnessList = []
    for i in range(len(population)):
        fitnessList.append(selectionList[i][0])
        populationList.append(selectionList[i][1])
    return populationList, fitnessList


#adding zeros at start to complete 10 bits as input size is 1024 = 2^10
def tenBit(num):
    x=num[::-1]
    while len(x)<10:
        x+='0'
        num = x[::-1]
    return num


#exchanging some binary digits to create new generation 
def crossover(rankedPopulation):
    newGeneration = []
    newGeneration.append(list(rankedPopulation[0]))
    for j in range(1, len(rankedPopulation)-1, 2):
        p1x = rankedPopulation[j][0]
        p1y = rankedPopulation[j][1]
        p2x = rankedPopulation[j+1][0]
        p2y = rankedPopulation[j+1][0]
        bin_p1x = tenBit(bin(p1x).replace("0b",""))     #coverting from decimal to binary
        bin_p1y = tenBit(bin(p1y).replace("0b",""))
        bin_p2x = tenBit(bin(p2x).replace("0b",""))
        bin_p2y = tenBit(bin(p2y).replace("0b",""))
        bin_c1x = bin_p1x[:7] + bin_p2y[7:]            #exchaning bits to create child from parents
        bin_c1y = bin_p1x[7:] + bin_p2y[:7]             
        bin_c2x = bin_p2x[:7] + bin_p1y[7:]
        bin_c2y = bin_p2x[7:] + bin_p1y[:7]
        dec_c1x = int(bin_c1x,2)                        #recoverting from binary to decimal           
        dec_c1y = int(bin_c1y,2)
        dec_c2x = int(bin_c2x,2)
        dec_c2y = int(bin_c2y,2)
        newGeneration.append([dec_c1x,dec_c1y])
        newGeneration.append([dec_c2x,dec_c2y])
    newGeneration.append(list(rankedPopulation[-1]))
    return newGeneration


#vit flip mutation
def exchangeBits(num_x, randNum):  
    if num_x[randNum]=='1':
        num_x[randNum] = '0'
    elif num_x[randNum]=='0':
        num_x[randNum] = '1'
    num_x = ''.join(num_x)    
    return num_x


#here, each sample undergoes a change of one binary bit
def mutation(nextGen):
    for x in range(len(nextGen)):
        while(nextGen[x][0]>995):
            num_x = tenBit(bin(nextGen[x][0]).replace("0b",""))
            num_x = list(num_x)
            randNum = random.randint(0,9)
            nextGen[x][0]=int(exchangeBits(num_x, randNum),2)


        while(nextGen[x][1]>477):
            num_y = tenBit(bin(nextGen[x][0]).replace("0b",""))
            num_y = list(num_y)
            randNum = random.randint(0,9)
            nextGen[x][1]=int(exchangeBits(num_y, randNum),2)
    return nextGen


#it draws patch on matching image
def patch(population,s_x,s_y,barriImage):
    fig, ax = plt.subplots()
    plt.gray()
    ax.imshow(barriImage)
    if len(population) != 0:
        rect = pacthes.Rectangle(population[0], s_y, s_x, linewidth=1, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
    plt.show()


populationSize = 100
chotiImageRows = chotiImage.shape[0]
chotiImageCols = chotiImage.shape[1]
barriImageRows = barriImage.shape[0]-chotiImageRows
barriImageCols = barriImage.shape[1]-chotiImageCols
population = initializePopulation(barriImageCols,barriImageRows,populationSize)
Maxfitness = []
Meanfitness = []
Matched = []
counter = 0
while counter<500:
    fitnessValues = evaluateFitness(population,barriImage,chotiImage)
    rankedPopulation, fitness = selection(population,fitnessValues)
    Meanfitness.append(nump.mean(fitness))                              #storing max and meanfitness to plot graph           
    Maxfitness.append(fitness[0])
    nextGeneration = crossover(rankedPopulation)
    population = mutation(nextGeneration)
    counter+=1
Matched.append(population[0])
if Matched != []:
    patch(Matched,chotiImageRows,chotiImageCols,barriImage)
else:
    patch([],chotiImageRows,chotiImageCols,barriImage)

plt.figure(1)
plt.plot(Maxfitness,'b')
plt.plot(Meanfitness,'g')
plt.show()
