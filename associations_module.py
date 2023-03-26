import pandas as pd
import math
import matplotlib.pyplot as plt
from perspective_module import Perspective
from compare_module import Compare


class Associations():
    def __init__(self, sampleImage, usePlainDataframes=True, includeDistance=False,
                    includeDistanceBy="multiply", limitDistance=None, limitPixels=None, useRanks=False):

        self.sampleImage = sampleImage
        self.includeDistance = includeDistance
        self.includeDistanceBy = includeDistanceBy
        self.limitDistance = limitDistance
        self.limitPixels = limitPixels
        self.useRanks = useRanks
        self.usePlainDataframes = usePlainDataframes

        if(usePlainDataframes and includeDistance):
            (
                self.angleDataframe,
                self.distanceDataframe,
                self.directionDataframe,
                self.perspectives
            )  = self.__getAssociations(
                sampleImage,
                includeDistance=includeDistance,
                includeDistanceBy=includeDistanceBy,
                limitDistance=limitDistance,
                limitPixels=limitPixels,
                useRanks=useRanks
            )
        elif(includeDistance and includeDistanceBy == "add"):
            (
                self.angleDistanceAdd,
                self.angleDistanceSub,
                self.angleDirectionAdd,
                self.angleDirectionSub,
                self.distanceDirectionAdd,
                self.distanceDirectionSub,
                self.perspectives
            ) = self.__getAssociations(
                sampleImage,
                includeDistance=includeDistance,
                includeDistanceBy=includeDistanceBy,
                limitDistance=limitDistance,
                limitPixels=limitPixels,
                useRanks=useRanks
            )
        else:
            (
                self.angleDirectionAdd,
                self.angleDirectionSub,
                self.perspectives
            ) = self.__getAssociations(
                sampleImage,
                includeDistance=includeDistance,
                includeDistanceBy=includeDistanceBy,
                limitDistance=limitDistance,
                limitPixels=limitPixels,
                useRanks=useRanks
            )

    def showPerspective(self, element):
        if(self.__getSampleElements.__contains(element)):
            self.perspectives[element].showPerspective(element)
        else:
            raise ValueError("The element picked; ("+str(element[0])+", "+str(element[1])+"), is outside the sample space")

    def getPerspective(self, element):
        if(self.__getSampleElements.__contains(element)):
            return self.perspectives[element]
        else:
            raise ValueError("The element picked; ("+str(element[0])+", "+str(element[1])+"), is outside the sample space")

    def showSelfImage(self):
        plt.imshow(self.sampleImage)     

    def compare(self, other, method="association", reference="self", useDistance="Average Cost"):
        # print("'method' argument in the compare method has been set to: "+str(method)+". method can be 'association' or 'perspective'")

        if(reference == "small"):
            if (len(self.angleDataframe) < len(other.angleDataframe)):
                reference = "self"
            elif (len(self.angleDataframe) > len(other.angleDataframe)):
                reference = "other"
        elif(reference == "big"):
            if (len(self.angleDataframe) > len(other.angleDataframe)):
                reference = "self"
            elif (len(self.angleDataframe) < len(other.angleDataframe)):
                reference = "other" 

        
        if(method == "association"):
            if(self.usePlainDataframes and self.includeDistance):
                a = self.__stretchDictionary(self.angleDataframe)
                b = self.__stretchDictionary(other.angleDataframe)
                angleDf = self.__getComparisonDataframe(a, b, reference=="self")

                a = self.__stretchDictionary(self.distanceDataframe)
                b = self.__stretchDictionary(other.distanceDataframe)
                distanceDf = self.__getComparisonDataframe(a, b, reference=="self")

                a = self.__stretchDictionary(self.directionDataframe)
                b = self.__stretchDictionary(other.directionDataframe)
                directionDf = self.__getComparisonDataframe(a, b, reference=="self")

                comparisonDataframe = angleDf.add(distanceDf).add(directionDf)

                return Compare(comparisonDataframe=comparisonDataframe)
            elif(self.includeDistance and self.includeDistanceBy == "add"):
                # Step 1

                a = self.__stretchDictionary(self.angleDistanceSub)
                b = self.__stretchDictionary(other.angleDistanceSub)
                angleDistanceSub = self.__getComparisonDataframe(a, b, reference=="self")

                a = self.__stretchDictionary(self.angleDistanceAdd)
                b = self.__stretchDictionary(other.angleDistanceAdd)
                angledistanceAdd = self.__getComparisonDataframe(a, b, reference=="self")

                a = self.__stretchDictionary(self.angleDirectionSub)
                b = self.__stretchDictionary(other.angleDirectionSub)
                angleDirectionSub = self.__getComparisonDataframe(a, b, reference=="self")

                a = self.__stretchDictionary(self.angleDirectionAdd)
                b = self.__stretchDictionary(other.angleDirectionAdd)
                angleDirectionAdd = self.__getComparisonDataframe(a, b, reference=="self")

                a = self.__stretchDictionary(self.distanceDirectionSub)
                b = self.__stretchDictionary(other.distanceDirectionSub)
                distanceDirectionSub = self.__getComparisonDataframe(a, b, reference=="self")

                a = self.__stretchDictionary(self.distanceDirectionAdd)
                b = self.__stretchDictionary(other.distanceDirectionAdd)
                distanceDirectionAdd = self.__getComparisonDataframe(a, b, reference=="self")

                # Step 2
                comparisonDataframe = angleDistanceSub.add(angledistanceAdd)
                comparisonDataframe = comparisonDataframe.add(angleDirectionSub).add(angleDirectionAdd)
                comparisonDataframe = comparisonDataframe.add(distanceDirectionSub).add(distanceDirectionAdd)

                return Compare(comparisonDataframe=comparisonDataframe)

                # return Compare(comparisonDataframe=comparisonDataframe)
            else:
                # Step 1
                a = self.__stretchDictionary(self.angleDirectionSub)
                b = self.__stretchDictionary(other.angleDirectionSub)
                angleDirectionSub = self.__getComparisonDataframe(a, b, reference=="self")

                a = self.__stretchDictionary(self.angleDirectionAdd)
                b = self.__stretchDictionary(other.angleDirectionAdd)
                angleDirectionAdd = self.__getComparisonDataframe(a, b, reference=="self")

                # Step 2
                comparisonDataframe = angleDirectionSub.add(angleDirectionAdd)
                return Compare(comparisonDataframe=comparisonDataframe)

        elif(method == "perspective"):
            comparisonDataframe = {}
            for selfPerspectiveElement in self.perspectives:
                comparisonDataframe[selfPerspectiveElement] = {}
                selfPerspective = self.perspectives[selfPerspectiveElement]
                for otherPerspectiveElement in other.perspectives:
                    otherPerspective = other.perspectives[otherPerspectiveElement]
                    compare = selfPerspective.compare(otherPerspective)
                    dist = 0
                    if(useDistance == "Average Cost"):
                        dist = compare.getAverageCost()
                    elif(useDistance == "Total Cost"):
                        dist = compare.getTotalCost()

                    comparisonDataframe[selfPerspectiveElement][otherPerspectiveElement] = dist
            
            return Compare(pd.DataFrame(comparisonDataframe))
                    

    # def examineFavoritePerspective(elementOrPerspective):
    #     if(str(type(elementOrPerspective)).__contains__("Perspective")):
    #         pass

    def compareAngleDistanceDifference(self, other):
        a = self.__stretchDictionary(self.angleDistanceSub)
        b = self.__stretchDictionary(other.angleDistanceSub)
        return Compare(self.__getComparisonDataframe(a, b))

    def compareAngleDistanceSum(self, other):
        a = self.__stretchDictionary(self.angleDistanceAdd)
        b = self.__stretchDictionary(other.angleDistanceAdd)
        return Compare(self.__getComparisonDataframe(a, b))

    def compareAngleDirectionDifference(self, other):
        a = self.__stretchDictionary(self.angleDirectionSub)
        b = self.__stretchDictionary(other.angleDirectionSub)
        return Compare(self.__getComparisonDataframe(a, b))  

    def compareAngleDirectionSum(self, other):
        a = self.__stretchDictionary(self.angleDirectionAdd)
        b = self.__stretchDictionary(other.angleDirectionAdd)
        return Compare(self.__getComparisonDataframe(a, b))                    

    def compareDistanceDirectionDifference(self, other):
        a = self.__stretchDictionary(self.distanceDirectionSub)
        b = self.__stretchDictionary(other.distanceDirectionSub)
        return Compare(self.__getComparisonDataframe(a, b))  

    def compareDistanceDirectionSum(self, other):
        a = self.__stretchDictionary(self.distanceDirectionAdd)
        b = self.__stretchDictionary(other.distanceDirectionAdd)
        return Compare(self.__getComparisonDataframe(a, b))  

    def __convertToDictionary(self, dataFrame):
        colVals = dataFrame.columns.values
        rowVals = dataFrame.index

        result = {}
        for col in colVals:
            for row in rowVals:
                if(not math.isnan(dataFrame[col][row])):
                    result[(col, row)] = dataFrame[col][row]

        return result

    def __getAssociations(self, sample, includeDistance, includeDistanceBy, limitDistance, limitPixels, useRanks):
        sampleElements = self.__getSampleElements()

        angleDistanceAdd = {}
        angleDistanceSub = {}
        angleDirectionAdd = {}
        angleDirectionSub = {}
        distanceDirectionAdd = {}
        distanceDirectionSub = {}

        angle = {}
        distance = {}
        direction = {}

        perspectives = {}

        for element in sampleElements:
            elementPerspective = Perspective(
                refElement=element,
                sampleImage=sample,
                usePlainDataframes=self.usePlainDataframes,
                includeDistance=includeDistance,
                includeDistanceBy=includeDistanceBy,
                limitDistance=limitDistance,
                limitPixels=limitPixels,
                useRanks=useRanks
            )

            perspectives[element] = elementPerspective

            if(self.usePlainDataframes and self.includeDistance):
                angle[element] = self.__convertToDictionary(elementPerspective.angleDataFrame)
                distance[element] = self.__convertToDictionary(elementPerspective.distanceDataFrame)
                direction[element] = self.__convertToDictionary(elementPerspective.directionDataFrame)
            elif(includeDistance and includeDistanceBy=="add"):
                angleDistanceAdd[element] = self.__convertToDictionary(elementPerspective.angleDistanceAdd)
                angleDistanceSub[element] = self.__convertToDictionary(elementPerspective.angleDistanceSub)
                angleDirectionAdd[element] = self.__convertToDictionary(elementPerspective.angleDirectionAdd)
                angleDirectionSub[element] = self.__convertToDictionary(elementPerspective.angleDirectionSub)
                distanceDirectionAdd[element] = self.__convertToDictionary(elementPerspective.distanceDirectionAdd)
                distanceDirectionSub[element] = self.__convertToDictionary(elementPerspective.distanceDirectionSub)
            else:
                angleDirectionAdd[element] = self.__convertToDictionary(elementPerspective.angleDirectionAdd)
                angleDirectionSub[element] = self.__convertToDictionary(elementPerspective.angleDirectionSub)

        if(self.usePlainDataframes and self.includeDistance):
            return (
                angle,
                distance,
                direction,
                perspectives
            )
        elif(includeDistance and includeDistanceBy=="add"):
            return (
                angleDistanceAdd,
                angleDistanceSub,
                angleDirectionAdd,
                angleDirectionSub,
                distanceDirectionAdd,
                distanceDirectionSub,
                perspectives
            )
        else:
            return (
                angleDirectionAdd,
                angleDirectionSub,
                perspectives
            )            

    def __stretchDictionary(self, dictionary):
        result = {}
        for keyLevel_1 in dictionary:
            for keyLevel_2 in dictionary[keyLevel_1]:
                # Don't use NaN. Also, don't use non-NaN values that have been used already
                if(not result.get((keyLevel_2, keyLevel_1)) and not math.isnan(dictionary[keyLevel_1][keyLevel_2])):
                    result[(keyLevel_1, keyLevel_2)] = dictionary[keyLevel_1][keyLevel_2]

        return result

    def __getComparisonDataframe(self, selfDictionary, otherDictionary, useSelfAsRef=True):
        #This method assumes the first item passed as a parameter is self

        if(useSelfAsRef):
            results = {}
            for selfElement in selfDictionary:
                results[selfElement] = {}
                for otherElement in otherDictionary:
                    results[selfElement][otherElement] = abs(selfDictionary[selfElement]  - otherDictionary[otherElement])

            return pd.DataFrame(results)
        else:
            results = {}
            for otherElement in otherDictionary:
                results[otherElement] = {}
                for selfElement in selfDictionary:
                    results[otherElement][selfElement] = abs(selfDictionary[selfElement]  - otherDictionary[otherElement])

            return pd.DataFrame(results)            

    def __getSampleElements(self):

        sampleElements = []
        for i in range(len(self.sampleImage)):
            for j in range(len(self.sampleImage[0])):
                if(not math.isnan(self.sampleImage[i][j])):
                    sampleElements.append((i, j))
        
        return sampleElements

