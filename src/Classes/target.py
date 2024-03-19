#Import basic system modules
import numpy as np

class Target():
    def __init__(self, x, dt, P=[]):
        'INPUT: estimate state of the target, eventualy associate P for uscented transform'
        self.x = x
        self.dt = dt
        self.P = P
        self.F = np.matrix([[1,0,self.dt,0], # Target State Transition Matrix - CV
                        [0,1,0,self.dt],
                        [0,0,1,0],
                        [0,0,0,1]])
