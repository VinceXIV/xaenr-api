import numpy as np
import pandas as pd
import math
import copy
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

class Compare:
    def __init__(self, comparisonDataframe):
        self.ncols = len(comparisonDataframe.columns)
        self.nrows = len(comparisonDataframe)
        self.comparisonDataframe = self.__replaceNanWithWorst(comparisonDataframe)
        self.cost = self.__makeSquared(self.__replaceNanWithWorst(comparisonDataframe))
        self.row_match, self.col_match = self.__getMatchDict()        

    def showMatch(self, reference="row", method="Total Distance"):
        # print("Currently using reference='"+reference+"', method='"+method+"'. reference can be 'big', 'row', or 'col'. On the other hand, method can be 'Total Distance', 'Average Distance', or 'Median Distance'")

        reference = "row" if (reference=="row" or (reference == "big" and self.nrows >= self.ncols) or (reference == "small" and self.nrows <= self.ncols)) else "column"

        canvas = self.__createCanvas()
        
        # We are expecting dictionary with column and row index in the form (int, int)
        if(str(type(self.cost.columns[0][0])).__contains__("int")):
            canvas = self.__paintCanvas(canvas=canvas, reference=reference, method=method)

        plt.imshow(canvas)

    def summarizeComparisonDataframe(self):
        itemList = self.__convertToList(self.comparisonDataframe)

        f, xarr = plt.subplots(nrows=1, ncols=2)
        f.set_size_inches(w=7, h=3)
        xarr[0].set_title("Histogram")
        xarr[0].hist(itemList, bins=8)
        xarr[1].set_title("Box Plot")
        xarr[1].boxplot(itemList)

    def filterAssociations(self, cost_is_less_than=5):
        result =  {}

        for row in self.comparisonDataframe.axes[0]:
            for col in self.comparisonDataframe.axes[1]:
                cost = self.comparisonDataframe[col][row]
                if(cost < cost_is_less_than):
                    result[(row, col)] = cost

        return result

    def getSimilarity(self, reference="big"):
        totalCost = self.getTotalCost()
        averageCost = self.getAverageCost()
        totalDistanceToNeighbors = self.getDistanceToNeighbors(method="total")
        averageDistanceToNeighbors = self.getDistanceToNeighbors(method="average")
        medianDistanceToNeighbors = self.getDistanceToNeighbors(method="median")

        if(
            reference == "row" or
            (reference == "big" and self.nrows >= self.ncols) or
            (reference == "small" and self.nrows <= self.ncols)
        ):
            return {
                "Total Cost": totalCost,
                "Average Cost": averageCost,
                "Total Distance": totalDistanceToNeighbors[0],
                "Average Distance": averageDistanceToNeighbors[0],
                "Median Distance": medianDistanceToNeighbors[0]
            }
        elif(
            reference == "column" or
            (reference == "big" and self.ncols >= self.nrows) or
            (reference == "small" and self.ncols <= self.ncols)
        ):
          return {
                "Total Cost": totalCost,
                "Average Cost": averageCost,
                "Total Distance": totalDistanceToNeighbors[1],
                "Average Distance": averageDistanceToNeighbors[1],
                "Median Distance": medianDistanceToNeighbors[1]
            }
        elif(reference == "all"):
          return {
                "Total Cost": totalCost,
                "Average Cost": averageCost,
                "Total Distance": totalDistanceToNeighbors,
                "Average Distance": averageDistanceToNeighbors,
                "Median Distance": medianDistanceToNeighbors
            }
        else:
            raise Exception("Expected similarity either be; 'big', 'small', 'column', or 'row'")                   

    def getDistanceToNeighbors(self, method="median", reference="all"):
        row_distances = []
        col_distances = []
        for elementMatch in self.row_match:
            row_distances.extend(self.__getNeighborPerformance(elementMatch, self.row_match))

        for elementMatch in self.col_match:
            col_distances.extend(self.__getNeighborPerformance(elementMatch, self.col_match))

        if(reference == "all"):
            if(method == "total"):
                return (np.sum(row_distances), np.sum(col_distances))
            elif(method == "average"):
                return (np.average(row_distances), np.average(col_distances))
            elif(method == "median"):
                return (np.median(row_distances), np.median(col_distances))
        elif(
            reference == "row" or
            (reference == "big" and self.nrows >= self.ncols) or
            (reference == "small" and self.nrows <= self.ncols)
        ):
            if(method == "total"):
                return np.sum(row_distances)
            elif(method == "average"):
                return np.average(row_distances)
            elif(method == "median"):
                return np.median(row_distances)            
        elif(
            reference == "column" or
            (reference == "big" and self.ncols >= self.nrows) or
            (reference == "small" and self.ncols <= self.ncols)
        ):
            if(method == "total"):
                return np.sum(col_distances)
            elif(method == "average"):
                return np.average(col_distances)
            elif(method == "median"):
                return np.median(col_distances)
        else:
            raise Exception("Invalid method or reference\nThis method expected 'all', 'row', or 'col' for reference argument and 'total', 'average' or 'median' for method argument")

    def getTotalCost(self):
        totalCost = 0
        worstVal = self.cost.max(axis=0).max()

        for i in range(len(self.row_match)):
            matchVal = self.cost[list(self.col_match)[i]][list(self.row_match)[i]]
            if(matchVal != worstVal): # Since the worst value was created just to make the dataframe have the same number of columns as rows
                totalCost += matchVal
        
        return totalCost

    def getAverageCost(self):
        totalCost = 0
        noOfItems = len(self.row_match)
        worstVal = self.cost.max(axis=0).max()
        #The 3 there is arbitrary. Anything greter than 1 would work
        for i in range(len(self.row_match)):
            matchVal = self.cost[list(self.col_match)[i]][list(self.row_match)[i]]
            if(matchVal != worstVal):
                totalCost += matchVal
            else:
                noOfItems -= 1
        
        return round(totalCost/noOfItems, 2)

    def getProportionOfZeroValue(self):
        noOfZeroValues = 0
        noOfAllValidElements = 0
        worstVal = self.cost.max(axis=0).max()
        for i in range(len(self.row_match)):
            matchVal = self.cost[list(self.col_match)[i]][list(self.row_match)[i]]
            if(matchVal == 0):
                noOfZeroValues += 1
            
            if(matchVal != worstVal):
                noOfAllValidElements += 1

        return noOfZeroValues/noOfAllValidElements

    def getMedianCost(self):
        matchVals = []
        worstVal = self.cost.max(axis=0).max()
        for i in range(len(self.row_match)):
            matchVal = self.cost[list(self.col_match)[i]][list(self.row_match)[i]]
            if(matchVal != worstVal):
                matchVals.append(matchVal)
        
        return round(np.median(matchVals), 2)

    def getRelationshipDistance(self, method="median", reference="column"):
        distances = []

        if(reference == "column" or
            (reference == "big" and self.ncols >= self.nrows) or
            (reference == "small" and self.ncols <= self.ncols)):
            distances = self.__getDistances(self.col_match)
        elif(reference == "row" or
            (reference == "big" and self.nrows >= self.ncols) or
            (reference == "small" and self.nrows <= self.ncols)):
            distances = self.__getDistances(self.row_match)
        elif(reference != "row" and reference != "column" and
            reference != "big" and reference != "small"):
            raise ValueError("Expected reference to either be 'row', 'column', 'big', of 'small'!")

        if(method == "total"):
            return np.sum(distances)
        elif(method == "average"):
            return np.average(distances)
        elif(method == "median"):
            return np.median(distances)
        else:
            raise ValueError("Expected method to either be 'total', 'average', or 'median'!")


    def __getDistances(self, dictionary):
        distances = []
        for element in dictionary:
            element_match = dictionary[element]
            for element2 in dictionary:
                element_match2 = dictionary[element2]

                dist1 = np.linalg.norm(np.array(element) - np.array(element2))
                dist2 = np.linalg.norm(np.array(element_match) -np.array(element_match2))
                distances.append(abs(dist1 - dist2))

        return distances
    
    def __createCanvas(self, value=math.nan):
        size = self.__getCanvasSize()
        return [[value]*size for i in range(size)]

    def __paintCanvas(self, canvas, reference, method):
        if(method == "Total Distance"):
            if(reference=="row"):
                for elementMatch in self.row_match:
                    canvas[elementMatch[1]][elementMatch[0]] = np.sum(self.__getNeighborPerformance(element=elementMatch, matchDict=self.row_match))
            elif(reference=="column"):
                for elementMatch in self.col_match:
                    canvas[elementMatch[1]][elementMatch[0]] = np.sum(self.__getNeighborPerformance(element=elementMatch, matchDict=self.col_match))                    
        elif(method == "Average Distance"):
            if(reference=="row"):
                for elementMatch in self.row_match:
                    canvas[elementMatch[1]][elementMatch[0]] = np.average(self.__getNeighborPerformance(element=elementMatch, matchDict=self.row_match))
            elif(reference=="column"):
                for elementMatch in self.col_match:
                    canvas[elementMatch[1]][elementMatch[0]] = np.average(self.__getNeighborPerformance(element=elementMatch, matchDict=self.col_match))
        elif(method == "Median Distance"):
            if(reference=="row"):
                for elementMatch in self.row_match:
                    canvas[elementMatch[1]][elementMatch[0]] = np.median(self.__getNeighborPerformance(element=elementMatch, matchDict=self.row_match))
            elif(reference=="column"):
                for elementMatch in self.col_match:
                    canvas[elementMatch[1]][elementMatch[0]] = np.median(self.__getNeighborPerformance(element=elementMatch, matchDict=self.col_match))
        else:
            raise ValueError("Expected reference to either be; 'row' or 'col', and method to either be 'Total Cost', 'Average Cost', 'Total Distance', 'Average Distance', or 'Median Distance'")

        return canvas

    def __getMatchDict(self):
        sampleColName = self.comparisonDataframe.columns[0]
        colType = str(type(sampleColName))

        if(
            colType.__contains__("tuple") and
            str(type(sampleColName[0])).__contains__("int")
        ):
            rowMatchElements = {}
            colMatchElements = {}

            row_ind, col_ind = linear_sum_assignment(self.cost)
            for i in range(len(row_ind)):
                if(self.cost.index[row_ind[i]][0] >= 0 and self.cost.columns[col_ind[i]][0] >= 0):
                    rowMatchElements[self.cost.index[row_ind[i]]] = self.cost.columns[col_ind[i]]
                    colMatchElements[self.cost.columns[col_ind[i]]] = self.cost.index[row_ind[i]]

            return (rowMatchElements, colMatchElements)
        elif(
            colType.__contains__("tuple") and
            str(type(sampleColName[0])).__contains__("tuple") and
            str(type(sampleColName[0][0])).__contains__("int")
        ):
            rowMatchAssociations = {}
            colMatchAssociations = {}

            row_ind, col_ind = linear_sum_assignment(self.cost)
            for i in range(len(row_ind)):
                if(self.cost.index[row_ind[i]][0][0] >= 0 and self.cost.columns[col_ind[i]][0][0] >= 0):
                    rowMatchAssociations[self.cost.index[row_ind[i]]] = self.cost.columns[col_ind[i]]
                    colMatchAssociations[self.cost.columns[col_ind[i]]] = self.cost.index[row_ind[i]]

            return (rowMatchAssociations, colMatchAssociations)

    def __makeSquared(self, df):
        newDf = copy.deepcopy(df)
        nrows = len(df)
        ncols = len(df.columns)

        if(nrows > ncols):
            return self.__addNewColumns(newDf, np.abs(nrows - ncols))
        elif(nrows < ncols):
            return self.__addNewRows(newDf, np.abs(nrows - ncols))
        else:
            return newDf
                
    def __addNewColumns(self, df, noOfCols):
        sampleColName = df.columns[0]

        worstValue = df.max(axis=0).max()
        worser = worstValue*3

        newColumns = [self.__getNewRowOrCol(i, sampleColName) for i in range(noOfCols)]
        newDf = pd.DataFrame([[worser]*noOfCols for i in range(len(df))])
        newDf.columns = newColumns
        newDf.index = df.index

        return pd.concat([df, newDf], axis=1)
            
    def __addNewRows(self, df, noOfRows):
        sampleColName = df.columns[0]

        worstValue = df.max(axis=0).max()
        worser = worstValue*3
        newRows =  [self.__getNewRowOrCol(i, sampleColName) for i in range(noOfRows)]
        newDf = pd.DataFrame([[worser]*len(df.columns) for i in range(noOfRows)])
        newDf.columns = df.columns
        newDf.index = newRows

        return pd.concat([df, newDf], axis=0)

    def __getNewRowOrCol(self, i, sampleColName):
        colType = str(type(sampleColName))

        if(colType.__contains__("int")):
            return (1 + i)*-1
        elif(
            colType.__contains__("tuple") and
            str(type(sampleColName[0])).__contains__("int")
        ):
            a = (1 + i)*-1
            b = (1 + i)*-1
            return (a, b)
        elif(
            colType.__contains__("tuple") and
            str(type(sampleColName[0])).__contains__("tuple") and
            str(type(sampleColName[0][0])).__contains__("int")
        ):
            a = ((1 + i)*-1, (1 + i)*-1)
            b = ((1 + i)*-1, (1 + i)*-1)
            return (a, b)
        else:
            raise ValueError("Expected coltype to be in the form; int, (int, int), or ((int, int), (int, int))")

    def __getNeighborPerformance(self, element, matchDict):
        #Checking if neighbors also matched with neighbors

        elementMatch = matchDict[element]
        distances = []
        neighbors = self.__getNeighbors(element)
        for neighborElement in neighbors:
            if(neighborElement in matchDict.keys()):
                neighborElementMatch = matchDict[neighborElement]
                distances.append(np.linalg.norm(np.array(elementMatch) - np.array(neighborElementMatch)))

        return distances
   
    def __getNeighbors(self, element):
        row = element[0]
        col = element[1]

        return [
            (row-1, col-1),
            (row-1, col),
            (row-1, col+1),
            (row, col+1),
            (row+1, col+1),
            (row+1, col),
            (row+1, col-1),
            (row, col-1)
        ]

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

    def __getCanvasSize(self):
        maxVal = 0

        for column in self.cost.columns:
            if(column[0] > maxVal):
                maxVal = column[0]
            elif(column[1] > maxVal):
                maxVal = column[1]
        
        for row in self.cost.index:
            if(row[0] > maxVal):
                maxVal = row[0]
            elif(row[1] > maxVal):
                maxVal = row[1]

        return maxVal + 1

    def __replaceNanWithWorst(self, dataframe):
        worstVal = dataframe.max(axis=0).max()*3
        #The 3 there is arbitrary. Anything greter than 1 would work
        return dataframe.fillna(worstVal)
