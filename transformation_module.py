import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

class Transformation:
    # The match_dict object is in the form 
    def __init__(self, match_dict):
        self.match_dict = match_dict
        self.variables = self.__getVariables()

        eqn1 = self.__getRegressionEquation("x2", ["x1", "y1"])
        eqn2 = self.__getRegressionEquation("y2", ["x1", "y1"])

        self.regressionResults1 = self.__getBetaValues(eqn1)
        self.regressionResults2 = self.__getBetaValues(eqn2)

    def showTransformation(self):
        params1 = self.regressionResults1.params
        params2 = self.regressionResults2.params

        # params1 and params2 contain the beta coefficient and
        # the y-intercept of the regression equations
        # we have the following equation where (x1, y1) was mapped onto (x2, y2)
        # we use the values in params1 and params2 to get the values of a, b, c, d, and k1, k2
        # [[a, b], [c, d]][x1, y1] + [k1, k2] = [x2, y2]
        a = round(params1["x1"], 4)
        b = round(params1["y1"], 4)
        c = round(params2["x1"], 4)
        d = round(params2["y1"], )
        k1 = round(params1["Intercept"], 4)
        k2 = round(params2["Intercept"], 4)

        return [[a, b], [c, d]], [k1, k2]

    def getVariables(self):
        return pd.DataFrame(self.variables)

    def __getVariables(self, match_dict = None):
        match_dict = self.match_dict if match_dict == None else match_dict

        result = {}
        result["x1"] = []
        result["y1"] = []
        result["x2"] = []
        result["y2"] = []

        for ref_element in match_dict:
            result["x1"].append(ref_element[0])
            result["y1"].append(ref_element[1])
            result["x2"].append(match_dict[ref_element][0])
            result["y2"].append(match_dict[ref_element][1])

        return result
    
    def __getBetaValues(self, regressionEquation, data = None):
        if(data == None):
            data = self.getVariables()
        
        return smf.ols(regressionEquation, data=data).fit()

    def __getRegressionEquation(self, y, x_arr):
        regressionEquation = "{y} ~".format(y = y)

        for x, i in zip(x_arr, range(len(x_arr))):
            regressionEquation = regressionEquation + " {x}{eql}".format(x = x, eql= " +" if i < len(x_arr) - 1 else "")

        return regressionEquation

