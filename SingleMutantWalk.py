import shelve
import numpy as np
import operator
import random
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class SingleMutantWalk:
    def __init__(self, shelveName):
        self.shelveName = shelveName # stores the name of the shelve
        self.landscape = shelve.open(shelveName) # all variants with corresponding fitness values
        # self.screenedVariants, only the screened variants, not the fitted ones, available only on the fittedVariants object
        self.aminoAcids = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"] # list of amino acid 1 letter codes
        self.enumerateAminoAcids(random=True) # maps each amino acid to an integer [0, 19] in a dictionary entitied self.aminoAcidDict
        self.getDataMatrixFromLandscape() # formulates the landscape into a data matrix and solution vector suitable for machine learning algorithms
        self.maxFitness = max(self.landscape.values()) # the value of the maximum fitness peak
        self.maxPeakResults = {"not max": 0, "max": 0} # "not max" represents how many times the single mutant walk did not converge onto the maximum fitness peak
        self.improvementResults = {} # will hold values of variants and the fitness peaks values they converge on in the given landscape


    def runSingleWalk(self, startingVariant):
        # print("optimizing variant {0}, beginning at fitness {1}".format(startingVariant, self.landscape[startingVariant]))
        bestVariant0pos, varient0fitness = self.getBestVariant(startingVariant, 0)
        bestVariant1pos, varient1fitness = self.getBestVariant(startingVariant, 1)
        bestVariant2pos, varient2fitness = self.getBestVariant(startingVariant, 2)
        bestVariant3pos, varient3fitness = self.getBestVariant(startingVariant, 3) # explore all positions and find the best variants at each
        bestVariants1stRound = [bestVariant0pos, bestVariant1pos, bestVariant2pos, bestVariant3pos]
        bestVariants1stRoundFitness = [varient0fitness, varient1fitness, varient2fitness, varient3fitness]
        bestVariant1stRound = bestVariants1stRound[bestVariants1stRoundFitness.index(max(bestVariants1stRoundFitness))]
        # print("{0} was optimized to {1} after the first round, new fitness value is {2}".format(startingVariant, bestVariant1stRound, self.landscape[bestVariant1stRound]))
        unexplored = [index for index, value in enumerate(bestVariants1stRoundFitness) if value is not max(bestVariants1stRoundFitness)] # all remaining unexplored positions
        bestVariant0pos2ndRound, varient0fitness2ndRound = self.getBestVariant(bestVariant1stRound, unexplored[0])
        bestVariant1pos2ndRound, varient1fitness2ndRound = self.getBestVariant(bestVariant1stRound, unexplored[1])
        bestVariant2pos2ndRound, varient2fitness2ndRound = self.getBestVariant(bestVariant1stRound, unexplored[2])
        bestVariants2ndRound = [bestVariant0pos2ndRound, bestVariant1pos2ndRound, bestVariant2pos2ndRound]
        bestVariants2ndRoundFitness = [varient0fitness2ndRound, varient1fitness2ndRound, varient2fitness2ndRound]
        bestVariant2ndRound = bestVariants2ndRound[bestVariants2ndRoundFitness.index(max(bestVariants2ndRoundFitness))]
        # print("{0} was optimized to {1} after the second round, new fitness value is {2}".format(bestVariant1stRound, bestVariant2ndRound, self.landscape[bestVariant2ndRound]))
        unexplored = [index for index, value in enumerate(bestVariants2ndRoundFitness) if value is not max(bestVariants2ndRoundFitness)] # all remaining unexplored positions
        bestVariant0pos3rdRound, varient0fitness3rdRound = self.getBestVariant(bestVariant2ndRound, unexplored[0])
        bestVariant1pos3rdRound, varient1fitness3rdRound = self.getBestVariant(bestVariant2ndRound, unexplored[1])
        bestVariants3rdRound = [bestVariant0pos3rdRound, bestVariant1pos3rdRound]
        bestVariants3rdRoundFitness = [varient0fitness3rdRound, varient1fitness3rdRound]
        bestVariant3rdRound = bestVariants3rdRound[bestVariants3rdRoundFitness.index(max(bestVariants3rdRoundFitness))]
        # print("{0} was optimized to {1} after the third round, new fitness value is {2}".format(bestVariant2ndRound, bestVariant3rdRound, self.landscape[bestVariant3rdRound]))
        unexplored = [index for index, value in enumerate(bestVariants3rdRoundFitness) if value is not max(bestVariants3rdRoundFitness)] # all remaining unexplored positions
        bestVariant4thRound, bestVariant4thRoundFitness = self.getBestVariant(bestVariant3rdRound, unexplored[0])
        # print("{0} was optimized to {1} after the fourth round, final fitness value is {2}".format(bestVariant3rdRound, bestVariant4thRound, self.landscape[bestVariant4thRound]))
        return bestVariant4thRound, bestVariant4thRoundFitness

    def runRecombination(self, variantList=None):
        # randomly samples the landscape 489 variants then makes a recombinatorial library from the top 3 variants and returns the best variant out of that library (theoretical maximum size 3^4 = 81 variants)
        sampledLibrary = {} # keep track of the variants and their corresponding fitnesses that we have sampled
        if not variantList:
            variantList = list(self.landscape.keys()) # list out all possible variants for random sampling
        for randomVariant in np.random.choice(variantList, 489, replace=False):
            sampledLibrary[randomVariant] = self.landscape[randomVariant] # keeps track of that variant
        # now, filter out all the entries in sampledLibrary dict except the top 3 variants
        firstHighest, secondHighest, thirdHighest = self.findFirstSecondAndThirdHighest(list(sampledLibrary.values())) # get the top 3 variant values
        filteredDict = filterDict(sampledLibrary, lambda keyValue: keyValue[1] == firstHighest or keyValue[1] == secondHighest or keyValue[1] == thirdHighest)
        bestVariant, secondBestVariant, thirdBestVariant = list(filteredDict.keys())
        recombinationVariants = self.findRecombinations(bestVariant, secondBestVariant, thirdBestVariant) # find the recombinatorial library from the top 3 variants
        return self.getBestVariantFromList(recombinationVariants)

    def sampleRecombination(self, timesToRun):
        print("Running recombination strategy {0} times".format(timesToRun))
        # runs the recombination directed evolution strategy the specified number of times
        peakFittnesses = []
        variantList = list(self.landscape.keys())
        counter = 0
        for _ in range(timesToRun):
            peakFitness = self.runRecombination(variantList=variantList)[1]
            peakFittnesses.append(peakFitness)
            counter += 1
            if not counter % 100:
                print("Completed {0} out of {1} recombination simulations".format(counter, timesToRun))
        return sum(peakFittnesses)/len(peakFittnesses)
        
    def sampleSingleWalk(self, timesToRun):
        print("Running single walk strategy {0} times".format(timesToRun))
        peakFittnesses = []
        variantList = list(self.landscape.keys())
        counter = 0 # keep count of how many simulations we have completed
        for variant in np.random.choice(variantList, timesToRun, replace=False):
            peakFitness = self.runSingleWalk(variant)[1]
            peakFittnesses.append(peakFitness)
            counter += 1
            if not counter % 100:
                print("Completed {0} out of {1} single walk simulations".format(counter, timesToRun))
        return sum(peakFittnesses)/len(peakFittnesses)

    def getBestVariant(self, startingVariant, position):
        beginning = startingVariant[:position]
        end = startingVariant[(position+1):]
        possibleVariants = [beginning + mutation + end for mutation in self.aminoAcids]
        possibleVariantsDict = {}
        for variant in possibleVariants:
            try:
                possibleVariantsDict[variant] = self.landscape[variant]
            except KeyError: # in the case the given variant is not one of the 149361 out of 160000 variants that were screened
                possibleVariantsDict[variant] = -1 # return a negative fitness value to assure that this nonexistent varient is not chosen for the next step in the single mutation walk
        bestVariant = max(possibleVariantsDict.items(), key=operator.itemgetter(1))[0]
        bestVariantFitness = possibleVariantsDict[bestVariant]
        return bestVariant, bestVariantFitness

    def runSingleWalkEntireLandscape(self):
        # runs a single mutation walk for every variant in the landscape
        counter = 0
        for variant in self.landscape.keys():
            self.improvementResults[variant] = {}
            bestVariantName, bestVariantFitness = self.runSingleWalk(variant)
            self.improvementResults[variant]["starting fitness"] = self.landscape[variant]
            self.improvementResults[variant]["final fitness"] = bestVariantFitness
            self.improvementResults[variant]["final variant"] = bestVariantName
            if bestVariantFitness == self.maxFitness:
                self.maxPeakResults["max"] += 1
            else:
                self.maxPeakResults["not max"] += 1
            counter += 1
            if not counter % 1000:
                print("completed {0} out of {1} variants".format(counter, len(self.landscape)))

    def getNumberOfPeaks(self):
        # returns how many unique peaks the single mutant walk on the entire landscape converged to
        # if we've already calculated this
        try:
            return len(self.uniquePeaks)
        except AttributeError:
            pass
        self.uniquePeaks = {}
        for variant in self.improvementResults.keys():
            finalVariant = self.improvementResults[variant]["final variant"]
            try:
                self.uniquePeaks[finalVariant]
            except KeyError:
                self.uniquePeaks[finalVariant] = self.improvementResults[variant]["final fitness"]
        return len(self.uniquePeaks)

    def getAverageFinalFitness(self):
        finalValues = [self.improvementResults[variant]["final fitness"] for variant in self.improvementResults.keys()]
        return sum(finalValues)/len(finalValues)

    def findFirstSecondAndThirdHighest(self, arr):
        # returns the first, second, and third highest values in a list
        arr.sort() # sort the array in ascending order
        firstHighest = arr[-1] # max element
        secondHighest = arr[-2] # second highest element
        thirdHighest = arr[-3] # third highest element
        return firstHighest, secondHighest, thirdHighest

    def findRecombinations(self, firstVariant, secondVariant, thirdVariant):
        recombinationList = []
        pos1 = [firstVariant[0], secondVariant[0], thirdVariant[0]]
        pos2 = [firstVariant[1], secondVariant[1], thirdVariant[1]]
        pos3 = [firstVariant[2], secondVariant[2], thirdVariant[2]]
        pos4 = [firstVariant[3], secondVariant[3], thirdVariant[3]]
        for aa1 in pos1:
            for aa2 in pos2:
                for aa3 in pos3:
                    for aa4 in pos4:
                        recombinedVariant = aa1 + aa2 + aa3 + aa4
                        if recombinedVariant not in recombinationList:
                            recombinationList.append(recombinedVariant)
        # print(len(recombinationList))
        return recombinationList

    def getBestVariantFromList(self, listOfVariants):
        maxVariant = "" # arbitrary non existent name of the max fitness variant
        maxFitness = -1 # arbitrary negative max fitness starting value
        for variant in listOfVariants:
            try:
                variantFitness = self.landscape[variant]
            except KeyError:
                continue
            if variantFitness > maxFitness:
                maxFitness = variantFitness
                maxVariant = variant
        return maxVariant, maxFitness

    # def getRandomVariantAndFitness(self, variantList=None):
    #     if not variantList:
    #         variantList = list(self.landscape.keys())
    #     # returns a random variant from the landscape
    #     randomIndex = np.random.randint(0, len(variantList)) # get a random index to choose a random variant from out landscape
    #     randomVariant = variantList[randomIndex]
    #     variantFitness = self.landscape[randomVariant]
    #     return randomVariant, variantFitness

    def enumerateAminoAcids(self, randomize=False):
        self.aminoAcidDict = {}
        counter = 0
        aminoAcids = self.aminoAcids[:] # make a copy of the list of amino acids
        if randomize:
            np.random.shuffle(aminoAcids)
        for aa in self.aminoAcids:
            self.aminoAcidDict[aa] = counter
            counter += 1

    def getDataMatrixFromLandscape(self):
        # converts the landscape into a data matrix and solution vector
        self.dataMatrix = []
        self.solutionVector = []
        for variant, fitness in self.landscape.items():
            variantEncoding = [] # list to hold integer values representing the amino acids at each position in this variant
            for aa in variant: # for each amino acid in this variant
                variantEncoding.append(self.aminoAcidDict[aa]) # add the integer value associated with that amino acid
            self.dataMatrix.append(variantEncoding) # add this variant to the data matrix
            self.solutionVector.append(fitness)

    def sampleLinearRegression(self, trainingSetSize, testingSetSize, timesToRun):
        # runs the sklearn LinearRegression algorithm on TRAININGSETSIZE training points to predict TRESTINGSETSIZE top variants. Returns the average best variant over TIMESTORUN simulations
        peakFittnesses = []
        variantList = list(self.landscape.keys())
        counter = 0
        for _ in range(timesToRun):
            randomIndeces = np.random.choice(len(self.landscape), trainingSetSize, replace=False)
            X = []
            y = []
            for index in randomIndeces[:trainingSetSize]: # have the model learn on training set of size specified by TRAININGSETSIZE
                X.append(self.dataMatrix[index])
                y.append(self.solutionVector[index])
            reg = LinearRegression().fit(X, y)
            unexploredIndeces = [index for index in np.arange(len(self.landscape)) if index not in randomIndeces] # all indeces that weren't part of the training data
            # next line takes a long time to run
            predictions = { variantList[index] : reg.predict([self.dataMatrix[index]]) for index in unexploredIndeces} # predict the fitness values for the variants at all of the above indeces
            predictions = { variant : predictedFitness for variant, predictedFitness in sorted(predictions.items(), key=lambda item: item[1])} # sort the list of variant fitness prediction in ascending order of fitness
            topPredictions = list(predictions.keys())[-testingSetSize:] # take the top TESTINGSIZE number of variants
            maxVariant, maxFitness = self.getBestVariantFromList(topPredictions)
            peakFittnesses.append(maxFitness)
            counter += 1
            # if not counter % 10:
            #     print("Completed {0} out of {1} linear regression simulations".format(counter, timesToRun))
        # change below afterwards
        # return sum(peakFittnesses)/len(peakFittnesses)
        return peakFittnesses

    def testLinearRegressionVariance(self):
        # investigates how many times you need to run the linear regression algorithm to be sure of the results
        xData = np.arange(600, 1000, 100) # number of times to run the algorithm
        yData = []
        runsNeeded = 5
        counter = 0
        for x in xData:
            newYData = []
            for _ in range(1): # run each timesToRun 10 times
                newYData += self.sampleLinearRegression(470, 100, x)
                counter += 1
                print("Completed {0} out of {1} linear regression variance simulations".format(counter, runsNeeded))
            yData.append(np.mean(newYData))
        plt.plot(xData, yData)
        plt.show()

    def testEnumerationOrder(self, timesToRun):
        # tests to see whether the number associated with each amino acid matters in terms of final fitness convergence
        aminoAcidDict = self.enumerateAminoAcids(randomize=True)
        xData = list(range(timesToRun))
        yData = []
        counter = 0
        for _ in range(timesToRun):
            yData.append(np.mean(self.sampleLinearRegression(470, 100, 600)))
            counter += 1
            print("finished {0} out of {1} random amino acid enumeration orders".format(counter, timesToRun))
        plt.plot(xData, yData)
        plt.show()

    def randomSample(self, timesToRun):
        # randomly samples the landscape 570 times and returns the average peak fitness
        peakFittnesses = []
        counter = 0
        for _ in range(timesToRun):
            randomVariants = np.random.choice(self.landscape, 570, replace=False)
            fitnesses = [self.landscape[variant] for variant in randomVariants]
            peakFittnesses.append(max(fitnesses))
            counter += 1
            if not counter % 100:
                print("Completed {0} out of {1} random sample simulations".format(counter, timesToRun))
        return sum(peakFittnesses)/len(peakFittnesses)

    def close(self):
        # closes the landscape shelve
        self.landscape.close()

    def open(self):
        # opens the landscape shelve
        self.landscape = shelve.open(self.shelveName)

    def sample(self):
        # runs 600 randomly sampled single mutant walk simulations and 600 recombination simluations
        # singleWalkAverageFitness = self.sampleSingleWalk(600)
        # recombinationAverageFitness = self.sampleRecombination(600)
        randomSampleAverageFitness = self.randomSample(600)
        # linearRegressionAverageFitness = self.sampleLinearRegression(470, 100, 600)
        # print("Average single walk final fitness: {0}".format(singleWalkAverageFitness))
        # print("Average recombination final fitness: {0}".format(recombinationAverageFitness))
        # print("Average linear regression final fitness: {0}".format(linearRegressionAverageFitness))
        print("Average random sample final fitness: {0}".format(randomSampleAverageFitness))

    def optimizeModelRatio(self):
        # finds the best amount of training variants and fitting variants that sum to 570 for training the linear regression algorithm
        return

def filterDict(dictObj, cb):
    newDict = {}
    for (key, value) in dictObj.items():
        if cb((key, value)):
            newDict[key] = value
    return newDict