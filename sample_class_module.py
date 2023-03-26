from cmath import nan
from skimage.morphology import thin
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import random


class Sample:
    def __init__(self, sample, description = "No description available"):
        self.sample = sample
        self.description = description
       
    def summarize(self):
        elementValues = self.getElementValues()

        f, xarr = plt.subplots(nrows=1, ncols=2)
        f.set_size_inches(w=7, h=3)

        xarr[0].set_title("Histogram")
        xarr[1].set_title("Boxplot")
        xarr[0].hist(elementValues)
        xarr[1].boxplot(elementValues)

    def getElementsCount(self):
        return len(self.getSampleElements())

    def getSampleElements(self):
        sampleElements = []
        for i in range(len(self.sample)):
            for j in range(len(self.sample[0])):
                if(not math.isnan(self.sample[i][j])):
                    sampleElements.append((i, j))
        
        return sampleElements

    def replaceValues(self, value = 0, replaceWith = math.nan, less=True, equal = True, greater = False):
        np_2d_array = copy.deepcopy(self.sample)

        for row in range(len(np_2d_array)):
            for col in range(len(np_2d_array[0])):
                if(
                    (equal and np_2d_array[row][col] == value) or
                    (less and np_2d_array[row][col] < value) or
                    (greater and np_2d_array[row][col] > value )
                ):
                    np_2d_array[row][col] = replaceWith

        return np_2d_array

    def getElementValues(self):
        result = []
        for row in self.sample:
            for value in row:
                if(not math.isnan(value)):
                    result.append(value)

        result.sort()
        return result

    def replaceElements(self, elementSet, newValue=math.nan, invertSelection=False):
        newSample = copy.deepcopy(self.sample.astype('float64'))

        if(not invertSelection):
            for element in elementSet:
                try:
                    newSample[element[0]][element[1]] = newValue
                # When we convertToAngle, the dimensions reduce by 1. if it was (10, 10), it changes to (9, 9), thus accessing
                # the error is raised. Here we are ignoring the error and continuing
                except IndexError:
                    continue
        else:
            for row in range(len(self.sample)):
                for col in range(len(self.sample[0])):
                    if(not elementSet.__contains__((row, col))):
                        try:
                            newSample[element[0]][element[1]] = newValue
                        # When we convertToAngle, the dimensions reduce by 1. if it was (10, 10), it changes to (9, 9), thus accessing
                        # the error is raised. Here we are ignoring the error and continuing
                        except IndexError:
                            continue

        return newSample

    def getElementsByValue(self, value=0, less=True, equal = False, greater = False, limitElementCount=None):

        elements = set()

        moreElements = []
        for row in range(len(self.sample)):
            for col in range(len(self.sample[0])):
                elementValue = self.sample[row][col]
                if(
                    (elementValue < value and less == True) or
                    (elementValue == value and equal == True) or
                    (elementValue > value and greater == True)
                ):
                    elements.add((row, col))

                if(limitElementCount != None and elementValue == value):
                    moreElements.append((row, col))

        expectedCount = (len(self.sample)*len(self.sample[0])) - limitElementCount
        additionalElements = expectedCount - len(elements)

        if(additionalElements > 0):
            additional = random.sample(moreElements, additionalElements)
            elements = elements.union(set(additional))
        
        return elements

    def convertToAngle(self, range_start=0, range_end=1, range_whole_number=False, thinify=False):
        nrows = int(self.sample.shape[0] - 1)
        ncols = int(self.sample.shape[1] - 1)

        result = np.zeros(shape=(nrows, ncols))

        if(not thinify):
            for row in range(nrows):
                for col in range(ncols):
                    item = self.__createItem(row, col, self.sample)
                    angle = self.__getAngle(item, range_start, range_end, range_whole_number)

                    result[row, col] = angle
        else:
            thinified = thin(self.sample).astype("float64")
            for row in range(nrows):
                for col in range(ncols):
                    item = self.__createItem(row, col, thinified)
                    angle = self.__getAngle(item, range_start, range_end, range_whole_number)

                    result[row, col] = angle

            for row in range(nrows):
                for col in range(ncols):
                    if(thinified[row][col] == 0):
                        result[row, col] = math.nan

        return result

    def __createItem(self, row, col, sample):
        a = sample[row, col]
        b = sample[row, col + 1]
        c = sample[row + 1, col]
        d = sample[row + 1, col + 1]

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

