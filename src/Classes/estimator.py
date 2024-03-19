#Import basic system modules
import numpy as np

class Estimation():
    def __init__(self):
        self.phi = []
        self.y = []
        self.x = np.zeros((2,1))
    
    def regressorUpdate(self, y_i, ax_i, ay_i):

        self.y.append(ax_i*np.sin(y_i) - ay_i*np.cos(y_i))
        C = [np.sin(y_i), -np.cos(y_i)]
        self.phi.append(C)

    def computeState(self,table):
        for i in range(len(table)):
            input_meas = table[i]
            self.regressorUpdate(input_meas[0],input_meas[1],input_meas[2])
