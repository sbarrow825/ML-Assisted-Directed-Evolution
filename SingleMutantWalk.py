import shelve
import numpy as np
import operator

class SingleMutantWalk:

    def __init__(self, shelveName):
        self.shelveName = shelveName # stores the name of the shelve
        self.landscape = shelve.open(shelveName) # all variants with corresponding fitness values
        self.aminoAcids = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"] # list of amino acid 1 letter codes
        self.maxFitness = max(self.landscape.values()) # the value of the maximum fitness peak
        self.maxPeakResults = {"not max": 0, "max": 0} # "not max" represents how many times the single mutant walk did not converge onto the maximum fitness peak
        self.improvementResults = {} # will hold values of variants and the fitness peaks values they converge on in the given landscape

    def runSingleWalk(self, startingVariant):
        print("optimizing variant {0}".format(startingVariant))
        bestVariant0pos, varient0fitness = self.getBestVariant(startingVariant, 0)
        bestVariant1pos, varient1fitness = self.getBestVariant(startingVariant, 1)
        bestVariant2pos, varient2fitness = self.getBestVariant(startingVariant, 2)
        bestVariant3pos, varient3fitness = self.getBestVariant(startingVariant, 3) # explore all positions and find the best variants at each
        bestVariants1stRound = [bestVariant0pos, bestVariant1pos, bestVariant2pos, bestVariant3pos]
        bestVariants1stRoundFitness = [varient0fitness, varient1fitness, varient2fitness, varient3fitness]
        bestVariant1stRound = bestVariants1stRound[bestVariants1stRoundFitness.index(max(bestVariants1stRoundFitness))]
        print("{0} was optimized to {1} after the first round".format(startingVariant, bestVariant1stRound))
        unexplored = [index for index, value in enumerate(bestVariants1stRoundFitness) if value is not max(bestVariants1stRoundFitness)] # all remaining unexplored positions
        bestVariant0pos2ndRound, varient0fitness2ndRound = self.getBestVariant(bestVariant1stRound, unexplored[0])
        bestVariant1pos2ndRound, varient1fitness2ndRound = self.getBestVariant(bestVariant1stRound, unexplored[1])
        bestVariant2pos2ndRound, varient2fitness2ndRound = self.getBestVariant(bestVariant1stRound, unexplored[2])
        bestVariants2ndRound = [bestVariant0pos2ndRound, bestVariant1pos2ndRound, bestVariant2pos2ndRound]
        bestVariants2ndRoundFitness = [varient0fitness2ndRound, varient1fitness2ndRound, varient2fitness2ndRound]
        bestVariant2ndRound = bestVariants2ndRound[bestVariants2ndRoundFitness.index(max(bestVariants2ndRoundFitness))]
        print("{0} was optimized to {1} after the second round".format(bestVariant1stRound, bestVariant2ndRound))
        unexplored = [index for index, value in enumerate(bestVariants2ndRoundFitness) if value is not max(bestVariants2ndRoundFitness)] # all remaining unexplored positions
        bestVariant0pos3rdRound, varient0fitness3rdRound = self.getBestVariant(bestVariant2ndRound, unexplored[0])
        bestVariant1pos3rdRound, varient1fitness3rdRound = self.getBestVariant(bestVariant2ndRound, unexplored[1])
        bestVariants3rdRound = [bestVariant0pos3rdRound, bestVariant1pos3rdRound]
        bestVariants3rdRoundFitness = [varient0fitness3rdRound, varient1fitness3rdRound]
        bestVariant3rdRound = bestVariants3rdRound[bestVariants3rdRoundFitness.index(max(bestVariants3rdRoundFitness))]
        print("{0} was optimized to {1} after the third round".format(bestVariant2ndRound, bestVariant3rdRound))
        unexplored = [index for index, value in enumerate(bestVariants3rdRoundFitness) if value is not max(bestVariants3rdRoundFitness)] # all remaining unexplored positions
        print("{0} was optimized to {1} after the fourth round".format(bestVariant3rdRound, self.getBestVariant(bestVariant3rdRound, unexplored[0])[0]))
        return self.getBestVariant(bestVariant3rdRound, unexplored[0])

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

    def getNumberOfPeaks(self):
        # returns how many unique peaks the single mutant walk on the entire landscape converged to
        self.uniquePeaks = {}
        for variant in self.improvementResults.keys():
            finalVariant = self.improvementResults[variant]["final variant"]
            try:
                self.uniquePeaks[finalVariant]
            except KeyError:
                self.uniquePeaks[finalVariant] = self.improvementResults[variant]["final fitness"]
        return len(self.uniquePeaks)

    def close(self):
        # closes the landscape shelve
        self.landscape.close()

    def open(self):
        # opens the landscape shelve
        self.landscape = shelve.open(self.shelveName)

def openPreviousAnalysis():
    test = shelve.open("analyzedRoughDraft")
    test = test["first analysis"]
    return test

test = openPreviousAnalysis()