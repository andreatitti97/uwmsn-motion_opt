#Import basic system modules
import numpy as np

class Target():
    def __init__(self, x, dt, P=[]):
        'INPUT: estimate state of the target, eventualy associate P for uscented transform'
        
        self.dt = dt
        # newton - eulero
        self.x = [0,0,x[2],x[3]]
        self.x[0] = x[0] + x[2]*self.dt
        self.x[1] = x[1] + x[3]*self.dt

        self.P = P
        self.F = np.matrix([[1,0,self.dt,0], # Target State Transition Matrix - CV
                        [0,1,0,self.dt],
                        [0,0,1,0],
                        [0,0,0,1]])
