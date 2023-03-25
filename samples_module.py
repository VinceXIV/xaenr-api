from cmath import nan
from skimage.morphology import thin
import pandas as pd
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import random


class Samples:
    def __init__(self):
        self.mnist_train = pd.read_csv("H:/Image/mnist_train.csv")

    def getAllSamples(self, number):
        if number < 0 or number > 9:
            raise ValueError(
                "Expected a value between 0 - 9. You provided " + str(number)
            )
        else:
            switcher = {
                0: self.mnist_train.loc[self.mnist_train["5"] == 0],
                1: self.mnist_train.loc[self.mnist_train["5"] == 1],
                2: self.mnist_train.loc[self.mnist_train["5"] == 2],
                3: self.mnist_train.loc[self.mnist_train["5"] == 3],
                4: self.mnist_train.loc[self.mnist_train["5"] == 4],
                5: self.mnist_train.loc[self.mnist_train["5"] == 5],
                6: self.mnist_train.loc[self.mnist_train["5"] == 6],
                7: self.mnist_train.loc[self.mnist_train["5"] == 7],
                8: self.mnist_train.loc[self.mnist_train["5"] == 8],
                9: self.mnist_train.loc[self.mnist_train["5"] == 9]
            }
            
            return switcher.get(number)
        

    def getSample(self, number, index=0, convertToAngle=True, limitElementCount=None, limitPercentile = None, rangeStart = 0, rangeEnd = 100, rangeWholeNumber = True, thinify=False):
    
        # print("ConvetToAngle argument in the getSample method has been set to "+str(convertToAngle)+", while limitPercentile has been set to "+str(limitPercentile))

        if(limitElementCount != None and limitPercentile != None):
            print("You have set both limitElementCount and limitPercentile to values that are not of NoneType. limitElementCount will be used in this operation")

        if number < 0 or number > 9:
            raise ValueError(
                "Expected a value between 0 - 9. You provided " + str(number)
            )
        else:
            switcher = {
                0:
                    np.array(
                        self.mnist_train.loc[self.mnist_train["5"] == 0].iloc[index, 1:]
                    ).reshape((28, 28)),
                1:
                    np.array(
                        self.mnist_train.loc[self.mnist_train["5"] == 1].iloc[index, 1:]
                    ).reshape((28, 28)),
                2:
                    np.array(
                        self.mnist_train.loc[self.mnist_train["5"] == 2].iloc[index, 1:]
                    ).reshape((28, 28)),
                3:
                    np.array(
                        self.mnist_train.loc[self.mnist_train["5"] == 3].iloc[index, 1:]
                    ).reshape((28, 28)),
                4:
                    np.array(
                        self.mnist_train.loc[self.mnist_train["5"] == 4].iloc[index, 1:]
                    ).reshape((28, 28)),
                5:
                    np.array(
                        self.mnist_train.loc[self.mnist_train["5"] == 5].iloc[index, 1:]
                    ).reshape((28, 28)),
                6:
                    np.array(
                        self.mnist_train.loc[self.mnist_train["5"] == 6].iloc[index, 1:]
                    ).reshape((28, 28)),
                7:
                    np.array(
                        self.mnist_train.loc[self.mnist_train["5"] == 7].iloc[index, 1:]
                    ).reshape((28, 28)),
                8:
                    np.array(
                        self.mnist_train.loc[self.mnist_train["5"] == 8].iloc[index, 1:]
                    ).reshape((28, 28)),
                9:
                    np.array(
                        self.mnist_train.loc[self.mnist_train["5"] == 9].iloc[index, 1:]
                    ).reshape((28, 28))
            }

            if(limitElementCount != None):
                sample = switcher.get(number)
                elementValues = self.getElementValues(sample=sample)
                elementValues.sort()

                noOfElementsToReplace = len(elementValues) - limitElementCount if len(elementValues) > limitElementCount else 0
                replaceValueThreshold = elementValues[noOfElementsToReplace] 

                # self.getElementsByValue gets all elements below replaceValueThreshold by default
                elementSet = self.getElementsByValue(sample=sample, value=replaceValueThreshold, limitElementCount=limitElementCount)
                if(convertToAngle):
                    return self.replaceElements(sample=self.convertToAngle(sample, rangeStart, rangeEnd, rangeWholeNumber), elementSet=elementSet, thinify=thinify)
                else:
                    return self.replaceElements(sample=switcher.get(number), elementSet=elementSet)
            elif(limitPercentile == None):
                if(convertToAngle):
                    return self.convertToAngle(switcher.get(number), rangeStart, rangeEnd, rangeWholeNumber, thinify=thinify)
                else:
                    return switcher.get(number)
            elif(limitPercentile != None):
                sample = switcher.get(number).astype('float64')
                sampleValues = []
                for row in sample:
                    sampleValues.extend(row)

                sampleValues = [i for i in sampleValues if not math.isnan(i)]
                value = np.percentile(a=sampleValues, q=limitPercentile)            
                sample = self.replaceValues(sample=sample, value=value)
                if(convertToAngle):
                    return self.convertToAngle(sample, rangeStart, rangeEnd, rangeWholeNumber, thinify)
                else:
                    return sample

    def summarizeSample(self, sample):
        elementValues = self.getElementValues(sample=sample)

        f, xarr = plt.subplots(nrows=1, ncols=2)
        f.set_size_inches(w=7, h=3)

        xarr[0].set_title("Histogram")
        xarr[1].set_title("Boxplot")
        xarr[0].hist(elementValues)
        xarr[1].boxplot(elementValues)

    def getSampleElementsCount(self, sample):
        return len(self.getSampleElements(sample=sample))

    def getSampleElements(self, sample):
        sampleElements = []
        for i in range(len(sample)):
            for j in range(len(sample[0])):
                if(not math.isnan(sample[i][j])):
                    sampleElements.append((i, j))
        
        return sampleElements

    def replaceValues(self, sample, value = 0, replaceWith = math.nan, less=True, equal = True, greater = False):
        np_2d_array = copy.deepcopy(sample)

        for row in range(len(np_2d_array)):
            for col in range(len(np_2d_array[0])):
                if(
                    (equal and np_2d_array[row][col] == value) or
                    (less and np_2d_array[row][col] < value) or
                    (greater and np_2d_array[row][col] > value )
                ):
                    np_2d_array[row][col] = replaceWith

        return np_2d_array

    def getElementValues(self, sample):
        result = []
        for row in sample:
            for value in row:
                if(not math.isnan(value)):
                    result.append(value)

        result.sort()
        return result

    def replaceElements(self, sample, elementSet, newValue=math.nan, invertSelection=False):
        newSample = copy.deepcopy(sample.astype('float64'))

        if(not invertSelection):
            for element in elementSet:
                try:
                    newSample[element[0]][element[1]] = newValue
                # When we convertToAngle, the dimensions reduce by 1. if it was (10, 10), it changes to (9, 9), thus accessing
                # the error is raised. Here we are ignoring the error and continuing
                except IndexError:
                    continue
        else:
            for row in range(len(sample)):
                for col in range(len(sample[0])):
                    if(not elementSet.__contains__((row, col))):
                        try:
                            newSample[element[0]][element[1]] = newValue
                        # When we convertToAngle, the dimensions reduce by 1. if it was (10, 10), it changes to (9, 9), thus accessing
                        # the error is raised. Here we are ignoring the error and continuing
                        except IndexError:
                            continue

        return newSample

    def getElementsByValue(self, sample, value=0, less=True, equal = False, greater = False, limitElementCount=None):

        elements = set()

        moreElements = []
        for row in range(len(sample)):
            for col in range(len(sample[0])):
                elementValue = sample[row][col]
                if(
                    (elementValue < value and less == True) or
                    (elementValue == value and equal == True) or
                    (elementValue > value and greater == True)
                ):
                    elements.add((row, col))

                if(limitElementCount != None and elementValue == value):
                    moreElements.append((row, col))

        expectedCount = (len(sample)*len(sample[0])) - limitElementCount
        additionalElements = expectedCount - len(elements)

        if(additionalElements > 0):
            additional = random.sample(moreElements, additionalElements)
            elements = elements.union(set(additional))
        
        return elements

    def convertToAngle(self, sample_img, range_start, range_end, range_whole_number, thinify=False):
        nrows = int(len(sample_img) - 1)
        ncols = int(len(sample_img) - 1)

        result = np.zeros(shape=(nrows, ncols))

        if(not thinify):
            for row in range(nrows):
                for col in range(ncols):
                    item = self.__createItem(sample_img, row, col)
                    angle = self.__getAngle(item, range_start, range_end, range_whole_number)

                    result[row, col] = angle
        else:
            thinified = thin(sample_img).astype("float64")
            for row in range(nrows):
                for col in range(ncols):
                    item = self.__createItem(thinified, row, col)
                    angle = self.__getAngle(item, range_start, range_end, range_whole_number)

                    result[row, col] = angle

            for row in range(nrows):
                for col in range(ncols):
                    if(thinified[row][col] == 0):
                        result[row, col] = math.nan

        return result

    def __createItem(self, sample_img, row, col):
        a = sample_img[row, col]
        b = sample_img[row, col + 1]
        c = sample_img[row + 1, col]
        d = sample_img[row + 1, col + 1]

        result = np.array([a, b, c, d])
        return result.reshape(2, 2)

    def __getAngle(self, item, range_start, range_end, range_whole_number):
        # a = item[0, 0] if (not math.isnan(item[0, 0])) else 0
        # b = item[0, 1] if (not math.isnan(item[0, 1])) else 0
        # c = item[1, 0] if (not math.isnan(item[1, 0])) else 0
        # d = item[1, 1] if (not math.isnan(item[1, 1])) else 0

        a = item[0, 0]
        b = item[0, 1]
        c = item[1, 0]
        d = item[1, 1]

        machine_epsilon = 10 ** (-16)

        Hor = abs(a - b) + abs(c - d)
        Vert = abs(a - c) + abs(b - d)

        # Don't process images with no edgest
        if Hor == 0 and Vert == 0:
            return nan

        Vert = Vert + machine_epsilon

        angle_in_rads = np.arctan(Hor / Vert)
        angle_normalized = angle_in_rads/(np.pi/2)
        angle_range = range_end - range_start

        if(range_whole_number):
            return round((angle_normalized * angle_range) + range_start, 0)
        else:
            return round((angle_normalized * angle_range) + range_start, 2)

