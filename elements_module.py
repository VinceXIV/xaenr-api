import numpy as np
from cmath import nan


class ElementOperations:
    def __init__(self, ref_sample = None):
        #ref_sample is the np array sample representing digit (in case of mnist digits)
        #that this object should use as a reference when doing subsequent operations
        #on elements
        self.ref_sample = ref_sample

    def createItem(self, row, col, ref_sample = None):
        if self.ref_sample == None and ref_sample == None:
            raise ValueError("This object was not initialized with a reference sample. As such, you need to provide one when calling this method. Here is an example <this object instance>.createItem(10, 10, your_sample)")
        elif self.ref_sample == None and ref_sample != None:
            return self.__getItem(ref_sample, row, col)
        elif self.ref_sample != None and ref_sample == None:
            return self.__getItem(self.ref_sample, row, col)


    def getAngle(self, item, range_start = 0, range_end = 100):
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
        angle_normalized = angle_in_rads/(2 * np.pi)
        angle_range = range_end - range_start

        return round((angle_normalized * angle_range) + range_start, 2)

    def getDirection(self, element1, element2, range_start = 0, range_end = 100):
        hor = abs(element1[0] - element2[0])
        vert = abs(element1[1] - element2[1])
        
        machine_epsilon = 10**(-16)
        vert = vert + machine_epsilon
                
        angle_in_rads = np.arctan(hor / vert)
        angle_normalized = angle_in_rads/(2 * np.pi)
        angle_range = range_end - range_start

        return round((angle_normalized * angle_range) + range_start, 2)

    def __getItem(self, ref_sample, row, col):
        a = ref_sample[row, col]
        b = ref_sample[row, col + 1]
        c = ref_sample[row + 1, col]
        d = ref_sample[row + 1, col + 1]

        result = np.array([a, b, c, d])
        return result.reshape(2, 2)