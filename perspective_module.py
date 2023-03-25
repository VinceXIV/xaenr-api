import math
import cmath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from compare_module import Compare

class Perspective:
    def __init__(self, refElement, sampleImage, usePlainDataframes=False, includeDistance=False, includeDistanceBy="add", limitDistance=None, limitPixels=None, useRanks=False):
        self.frameOfReference = refElement
        self.sampleSpace = sampleImage
        self.includeDistance = includeDistance
        self.usePlainDataframes = usePlainDataframes

        if(not self.__getSampleElements(sampleImage).__contains__(refElement)):
            raise ValueError("The element picked; ("+str(refElement[0])+", "+str(refElement[1])+"), is outside the sample space")

        if(includeDistance):        
            (
                angleDifference,
                self.angleSet,
                distanceDifference,
                self.distanceSet,
                directionDifference,
                self.directionSet
            ) = self.__getPerspectivizedSample(refElement, sampleImage, includeDistance, limitDistance, limitPixels)

            # Step 1
            self.angleDataFrame = pd.DataFrame.from_dict(angleDifference)
            self.distanceDataFrame = pd.DataFrame.from_dict(distanceDifference)
            self.directionDataFrame = pd.DataFrame.from_dict(directionDifference)

            # Step 2
            self.angleDataFrame = self.angleDataFrame.dropna(axis=0, how='all')
            self.distanceDataFrame = self.distanceDataFrame.dropna(axis=0, how='all')
            self.directionDataFrame = self.directionDataFrame.dropna(axis=0, how='all')

            self.angleDataFrame = self.angleDataFrame.dropna(axis=1, how='all')
            self.distanceDataFrame = self.distanceDataFrame.dropna(axis=1, how='all')
            self.directionDataFrame = self.directionDataFrame.dropna(axis=1, how='all')

            # Step 3
            self.angleDataFrame = self.angleDataFrame.transpose().sort_index(axis=1)
            self.distanceDataFrame = self.distanceDataFrame.transpose().sort_index(axis=1)
            self.directionDataFrame = self.directionDataFrame.transpose().sort_index(axis=1)

            # Step 4
            if(useRanks and includeDistanceBy=="add"):
                self.angleDistanceAdd = self.getAngleRanks().add(self.getDistanceRanks())
                self.angleDistanceSub = self.getAngleRanks().sub(self.getDistanceRanks()).abs()
                self.angleDirectionAdd = self.getAngleRanks().add(self.getDirectionRanks())
                self.angleDirectionSub = self.getAngleRanks().sub(self.getDirectionRanks()).abs()
                self.distanceDirectionAdd = self.getDistanceRanks().add(self.getDirectionRanks())
                self.distanceDirectionSub = self.getDistanceRanks().sub(self.getDirectionRanks()).abs()
            elif(not useRanks and includeDistanceBy == "add"):
                self.angleDistanceAdd = self.angleDataFrame.add(self.distanceDataFrame)
                self.angleDistanceSub = self.angleDataFrame.sub(self.distanceDataFrame).abs()
                self.angleDirectionAdd = self.angleDataFrame.add(self.directionDataFrame)
                self.angleDirectionSub = self.angleDataFrame.sub(self.directionDataFrame).abs()
                self.distanceDirectionAdd = self.distanceDataFrame.add(self.directionDataFrame)
                self.distanceDirectionSub = self.distanceDataFrame.sub(self.directionDataFrame).abs()
            elif(not useRanks and includeDistanceBy == "multiply"):
                self.angleDirectionAdd = self.angleDataFrame.add(self.directionDataFrame).multiply(self.distanceDataFrame)
                self.angleDirectionSub  = self.angleDataFrame.sub(self.directionDataFrame).multiply(self.distanceDataFrame).abs()                 
        else:
            (
                angleDifference,
                self.angleSet,
                directionDifference,
                self.directionSet
            ) = self.__getPerspectivizedSample(refElement, sampleImage, includeDistance, limitDistance, limitPixels)

            # Step 1
            self.angleDataFrame = pd.DataFrame.from_dict(angleDifference)
            self.directionDataFrame = pd.DataFrame.from_dict(directionDifference)

            # Step 2
            self.angleDataFrame = self.angleDataFrame.dropna(axis=0, how='all')
            self.directionDataFrame = self.directionDataFrame.dropna(axis=0, how='all')

            self.angleDataFrame = self.angleDataFrame.dropna(axis=1, how='all')
            self.directionDataFrame = self.directionDataFrame.dropna(axis=1, how='all')

            # Step 3
            self.angleDataFrame = self.angleDataFrame.transpose().sort_index(axis=1)
            self.directionDataFrame = self.directionDataFrame.transpose().sort_index(axis=1)

            # Step 4
            self.angleDirectionAdd = self.getAngleRanks().add(self.getDirectionRanks())
            self.angleDirectionSub = self.getAngleRanks().sub(self.getDirectionRanks()).abs()


    def getComparisonDataframe(self, other, usePlainDataframes=None):
        usePlainDataframes = self.usePlainDataframes if usePlainDataframes == None else usePlainDataframes

        if(not usePlainDataframes):
            if(self.includeDistance):
                angleDistanceAddDf = self.__getComparisonDataframe(self.angleDistanceAdd, other.angleDistanceAdd)
                angleDistanceSubDf = self.__getComparisonDataframe(self.angleDistanceSub, other.angleDistanceSub)
                angleDirectionAddDf = self.__getComparisonDataframe(self.angleDirectionAdd, other.angleDirectionAdd)
                angleDirectionSubDf = self.__getComparisonDataframe(self.angleDirectionSub, other.angleDirectionSub)
                distanceDirectionAddDf = self.__getComparisonDataframe(self.distanceDirectionAdd, other.distanceDirectionAdd)
                distanceDirectionSubDf = self.__getComparisonDataframe(self.distanceDirectionSub, other.distanceDirectionSub)

                result = angleDistanceAddDf.add(angleDistanceSubDf)
                result = result.add(angleDirectionAddDf).add(angleDirectionSubDf)
                result = result.add(distanceDirectionAddDf).add(distanceDirectionSubDf)

                return result

            else:
                angleDirectionAddDf = self.__getComparisonDataframe(self.angleDirectionAdd, other.angleDirectionAdd)
                angleDirectionSubDf = self.__getComparisonDataframe(self.angleDirectionSub, other.angleDirectionSub)

                return angleDirectionAddDf.add(angleDirectionSubDf)
        else:
            angle = self.__getComparisonDataframe(self.angleDataFrame, other.angleDataFrame)
            distance = self.__getComparisonDataframe(self.distanceDataFrame, other.distanceDataFrame)
            direction = self.__getComparisonDataframe(self.directionDataFrame, other.directionDataFrame)

            return angle.add(distance).add(direction)

    def compare(self, other, usePlainDataframes=None):
        usePlainDataframes = self.usePlainDataframes if usePlainDataframes == None else usePlainDataframes

        comparisionDataframe = self.getComparisonDataframe(other, usePlainDataframes)
        return Compare(comparisionDataframe)

    def getDistanceRanks(self):
        return self.__convertToRanks(self.distanceSet, self.distanceDataFrame)

    def getAngleRanks(self):
        return self.__convertToRanks(self.angleSet, self.angleDataFrame)
    
    def getDirectionRanks(self):
        return self.__convertToRanks(self.directionSet, self.directionDataFrame)

    def getAngleDistanceRankDifference(self):
        return self.getDistanceRanks().sub(self.getAngleRanks()).abs()

    def getAngleDistanceRankSum(self):
        return self.getDistanceRanks().add(self.getAngleRanks())

    def getAngleDirectionRankDifference(self):
        return self.getAngleRanks().sub(self.getDirectionRanks()).abs()

    def getAngleDirectionRankSum(self):
        return self.getAngleRanks().add(self.getDirectionRanks())

    def getDistanceDirectionRankDifference(self):
        return self.getDistanceRanks().sub(self.getDirectionRanks()).abs()

    def getDistanceDirectionRankSum(self):
        return self.getDistanceRanks().add(self.getDirectionRanks())

    def getDistanceRankList(self):
        return self.__convertToList(self.getDistanceRanks())

    def getAngleRankList(self):
        return self.__convertToList(self.getAngleRanks())

    def getDirectionRankList(self):
        return self.__convertToList(self.getDirectionRanks())

    def summarize(self, attribute="angle"):
        attributes =  [
                "angle", "distance", "direction", "angleDirectionAdd",
                "angleDirectionSub", "distanceDirectionAdd", "distanceDirectionSub"
            ]

        # print("Other attributes you could pass when calling this method include: ", attributes)

        itemList = self.__getItemList(attribute)

        f, xarr = plt.subplots(nrows=1, ncols=2)
        f.set_size_inches(w=7, h=3)
        xarr[0].set_title("Histogram")
        xarr[0].hist(itemList, bins=20)
        xarr[1].set_title("Box Plot")
        xarr[1].boxplot(itemList)

    def showPerspective(self):
        elementAngle = self.sampleSpace[self.frameOfReference[0], self.frameOfReference[1]]

        if(self.includeDistance):
            f, xarr = plt.subplots(nrows=1, ncols=3)
            f.set_size_inches(w=11, h=3)
            xarr[0].set_title("Angle: ("+str(elementAngle)+")")
            xarr[1].set_title("Distance")
            xarr[2].set_title("Direction")
            xarr[0].imshow(self.angleDataFrame)
            xarr[1].imshow(self.distanceDataFrame)
            xarr[2].imshow(self.directionDataFrame)
        else:
            f, xarr = plt.subplots(nrows=1, ncols=2)
            f.set_size_inches(w=10, h=5)
            xarr[0].set_title("Angle: ("+str(elementAngle)+")")
            xarr[1].set_title("Direction")
            xarr[0].imshow(self.angleDataFrame)
            xarr[1].imshow(self.directionDataFrame)

    def showElement(self, element=None):
        element = element if element != None else self.frameOfReference

        f, xarr = plt.subplots(nrows=1, ncols=2)
        f.set_size_inches(w=10, h=10)

        foRefCopy = copy.deepcopy(self.sampleSpace)
        foRefCopy[element[0]][element[1]]= -50

        xarr[0].imshow(self.sampleSpace)
        xarr[1].imshow(foRefCopy)       

    def showPerspectiveCount(self):
        return sum(self.angleDataFrame.count())

    def __convertToList(self, itemDataFrame):
        itemList =  []

        for row in itemDataFrame.axes[0]:
            for col in itemDataFrame.axes[1]:
                try:
                    if(math.isnan(itemDataFrame[col][row])):
                        continue
                    itemList.append(itemDataFrame[col][row])
                except:
                    continue
        
        return itemList

    def __convertToRanks(self, rankSet, sample):
        for i in range(len(rankSet)):
            value = rankSet[i]
            rank = i
            sample = sample.replace(value, rank)
        
        return sample

    def __getPerspectivizedSample(self, refElement, sampleImage, includeDistance, limitDistance, limitPixels):
        refElementRow = refElement[0]
        refElementCol = refElement[1]
        elementAngle = sampleImage[refElementRow, refElementCol]
        
        
        angleDifference = {}
        distanceDifference = {}
        directionDifference = {}

        angleSet = set()
        distanceSet = set()
        directionSet = set()

        if(includeDistance):
            for row in range(len(sampleImage)):
                angleDifference[row] = {}
                distanceDifference[row] = {}
                directionDifference[row] = {}
                for col in range(len(sampleImage[row])):
                    dist = self.__getDistance(refElementRow, refElementCol, row, col, sampleImage, limitDistance)
                    if(not math.isnan(sampleImage[row, col]) and limitDistance != None and limitPixels == None):
                        if(math.isnan(dist)):
                            continue
                        else:
                            dist = round(dist, 0)
                            angl = round(abs(elementAngle - sampleImage[row, col]), 0) #angle is relative to angle of ref element
                            dir = round(abs(elementAngle - self.__getDirection(refElement, (row, col))), 0) #direction is relative to direction of ref element

                            angleSet.add(angl)
                            distanceSet.add(dist)
                            directionSet.add(dir)

                            distanceDifference[row][col] = dist
                            angleDifference[row][col] = angl 
                            directionDifference[row][col] = dir  
                    elif(not math.isnan(sampleImage[row, col])):
                        limitPixels = np.Inf if limitPixels == None else limitPixels
                        if(abs(refElementRow - row) <= limitPixels and abs(refElementCol - col) <= limitPixels):
                            dist = round(dist, 0)
                            angl = round(abs(elementAngle - sampleImage[row, col]), 0)
                            dir = round(abs(elementAngle - self.__getDirection(refElement, (row, col))), 0)

                            angleSet.add(angl)
                            distanceSet.add(dist)
                            directionSet.add(dir)

                            distanceDifference[row][col] = dist
                            angleDifference[row][col] = angl 
                            directionDifference[row][col] = dir                             

            angleSet = sorted(angleSet)
            distanceSet = sorted(distanceSet)
            directionSet = sorted(directionSet)

            return (angleDifference, angleSet, distanceDifference, distanceSet, directionDifference, directionSet)
        else:
            for row in range(len(sampleImage)):
                angleDifference[row] = {}
                directionDifference[row] = {}
                for col in range(len(sampleImage[row])):
                    if(not math.isnan(sampleImage[row, col]) and limitDistance != None and limitPixels == None):
                        dist = self.__getDistance(refElementRow, refElementCol, row, col, sampleImage, limitDistance)

                        if(math.isnan(dist)):
                            continue
                        else:
                            angl = round(abs(elementAngle - sampleImage[row, col]), 0) #angle is relative to angle of ref element
                            dir = round(abs(elementAngle - self.__getDirection(refElement, (row, col))), 0) #direction is relative to direction of ref element

                            angleSet.add(angl)
                            directionSet.add(dir)

                            angleDifference[row][col] = angl 
                            directionDifference[row][col] = dir
                    elif(not math.isnan(sampleImage[row, col])):
                        limitPixels = np.Inf if limitPixels == None else limitPixels
                        if(abs(refElementRow - row) <= limitPixels and abs(refElementCol - col) <= limitPixels):
                                angl = round(abs(elementAngle - sampleImage[row, col]), 0)
                                dir = round(abs(elementAngle - self.__getDirection(refElement, (row, col))), 0) 

                                angleSet.add(angl)
                                directionSet.add(dir)

                                angleDifference[row][col] = angl 
                                directionDifference[row][col] = dir                          

            angleSet = sorted(angleSet)
            directionSet = sorted(directionSet)

            return (angleDifference, angleSet, directionDifference, directionSet)


    def __getDistance(self, refElementRow, refElementCol, row, col, sampleImage, limitDistance, range_start = 0, range_end = 100):
        largest_expected_distance = np.sqrt(len(sampleImage)**2 + len(sampleImage[0])**2)
        abs_distance = abs(np.sqrt((refElementRow - row)**2 + (refElementCol - col)**2))
        mult_factor = (range_end - range_start)/largest_expected_distance
        standardized_distance = abs_distance*mult_factor + range_start # Standardized as in it ranges between 0 and 100

        if(limitDistance == None):
            return standardized_distance
        elif(standardized_distance > limitDistance):
            return cmath.nan
        else:
            return standardized_distance   

    def __getDirection(self, element1, element2, range_start = 0, range_end = 100):
        hor = abs(element1[0] - element2[0])
        vert = abs(element1[1] - element2[1])
        
        machine_epsilon = 10**(-16)
        vert = vert + machine_epsilon
                
        angle_in_rads = np.arctan(hor / vert)
        angle_normalized = angle_in_rads/(np.pi/2)
        angle_range = range_end - range_start

        return round((angle_normalized * angle_range) + range_start, 2)

    def __convertToDictionary(self, dataFrame):
        colVals = dataFrame.columns.values
        rowVals = dataFrame.index

        result = {}
        for col in colVals:
            for row in rowVals:
                if(not math.isnan(dataFrame[col][row])):
                    result[(col, row)] = dataFrame[col][row]

        return result

    def __getComparisonDataframe(self, item_1, item_2):
        results = {}
        if(str(type(item_1)).__contains__("dict") and str(type(item_2)).__contains__("dict")):
            pass
        elif(str(type(item_1)).__contains__("DataFrame") and str(type(item_2)).__contains__("DataFrame")):
            item_1 = self.__convertToDictionary(item_1)
            item_2 = self.__convertToDictionary(item_2)

        for selfElement in item_1:
            results[selfElement] = {}
            for otherElement in item_2:
                results[selfElement][otherElement] = abs(item_1[selfElement]  - item_2[otherElement]) 

        return pd.DataFrame(results)  

    def __convertToDictionary(self, dataFrame):
        colVals = dataFrame.columns.values
        rowVals = dataFrame.index

        result = {}
        for col in colVals:
            for row in rowVals:
                if(not math.isnan(dataFrame[col][row])):
                    result[(col, row)] = dataFrame[col][row]

        return result

    def __getItemList(self, attribute):
        if(attribute == "angle"):
            return self.__convertToList(self.angleDataFrame)
        elif(attribute == "distance"):
            return self.__convertToList(self.distanceDataFrame)
        elif(attribute == "direction"):
            return self.__convertToList(self.directionDataFrame)
        elif(attribute == "angleDirectionAdd"):
            return self.__convertToList(self.angleDirectionAdd)
        elif(attribute == "angleDirectionSub"):
            return self.__convertToList(self.angleDirectionSub)
        elif(attribute == "angleDistanceAdd"):
            return self.__convertToList(self.angleDistanceAdd)
        elif(attribute == "angleDistanceSub"):
            return self.__convertToList(self.angleDistanceSub)
        elif(attribute == "distanceDirectionAdd"):
            return self.__convertToList(self.distanceDirectionAdd)
        elif(attribute == "distanceDirectionSub"):
            return self.__convertToList(self.distanceDirectionSub)

    def __getSampleElements(self, sample):
        sampleElements = []
        for i in range(len(sample)):
            for j in range(len(sample[0])):
                if(not math.isnan(sample[i][j])):
                    sampleElements.append((i, j))
        
        return sampleElements