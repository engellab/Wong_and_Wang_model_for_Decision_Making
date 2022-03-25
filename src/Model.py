import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from copy import copy
from scipy.optimize import fsolve
from tqdm.auto import tqdm
from scipy.optimize import minimize

def in_the_list(x, x_list):
    for i in range(len(x_list)):
        diff = np.linalg.norm(x-x_list[i],2)
        if diff < 1e-3:
            return True
    return False

class DM_model():
    def __init__(self, dt=0.0005):
        self.dt = dt
        self.tau_s = 0.1
        self.tau_AMPA = 0.002
        self.a = 270
        self.b = 108
        self.d = 0.154
        self.gamma = 0.641
        self.sigma_noise = 0.02
        self.J_E = 0.2609
        self.J_I = -0.0497
        self.J_ext = 0.0156
        self.J = np.array([[self.J_E, self.J_I], [self.J_I, self.J_E]])
        self.Ib = 0.3255*np.ones((2,1))

        def fI(x):
            tmp = (self.a*x - self.b)
            return tmp/(1 - np.exp(-self.d*tmp))

        def der_fI(x):
            tmp = (self.a * x - self.b)
            term1 = self.a / (1 - np.exp(-self.d * tmp))
            term2 = - self.a * tmp * self.d * np.exp(-self.d * tmp) / (1 - np.exp(-self.d * tmp)) ** 2
            return term1 + term2

        self.fI = fI
        self.der_fI = der_fI

        # self.s_init = 0.4244552*np.ones((2,1))
        self.s_init = np.zeros((2,1))
        self.s = copy(self.s_init)
        self.I_noise = 0.1 * np.random.randn(2,1)
        self.s_variable_buffer = [copy(self.s)]

    def rhs(self, s, c):
        self.I_noise += (self.dt/self.tau_AMPA) * (-self.I_noise + np.random.randn(2,1) * np.sqrt(self.tau_AMPA * self.sigma_noise**2))
        x = self.J @ s + self.Ib + self.J_ext * np.array([1+c,1-c]).reshape(-1, 1) + self.I_noise
        rhs_vect = - s/self.tau_s + self.gamma * (1 - s) * self.fI(x)
        return rhs_vect

    def run(self, T, c):
        num_steps = int(np.ceil(T / self.dt))
        for i in (range(num_steps)):
            self.s = self.s + self.dt * self.rhs(self.s, c)
            self.s_variable_buffer.append(copy(self.s)) # so it is N x T
        return None

    def get_history(self):
        return np.hstack(self.s_variable_buffer)  # N

    def clear_history(self):
        self.s_variable_buffer = []
        self.s = self.s_init
        return None

    def plot_outputs(self):
        ss = self.get_history().T
        fig = plt.figure(figsize=(12,3))
        plt.plot(ss[:, 0] - ss[:, 1], color='r', label = 'choice variable')
        plt.ylim([-1,1])
        plt.grid(True)
        return fig

if __name__ == '__main__':
    m = DM_model()
    T = 2
    c = np.random.rand()*2 - 1
    m.run(T, c)
    fig = m.plot_outputs()
    plt.plot(c * np.ones(int(T/m.dt)), color = 'b', label = "input signal")
    plt.legend(fontsize=15)
    plt.show()

