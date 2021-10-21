# Copyright (c) 2021, BOXVIA Developers
# All rights reserved.
# Code released under the BSD 3-clause license.

import GPy
import GPyOpt
from numpy.random import seed
import numpy as np

class BO:

    def __init__(self, n_axis, vmaxmin):

        self.n_axis = n_axis
        self.vmaxmin = vmaxmin


    def setBO(self, initial_data, axis_name, acquisition, maximize, exact_fval, de_duplication, jwparam, batch_size, kernel, cont_or_disc, stride, constr=None):

        Y_data = initial_data[:, -1].reshape(-1, 1)
        if maximize:
            Y_data = -Y_data

        self.Y_data = Y_data

        bounds = []
        for i in range(self.n_axis):
            if cont_or_disc[i] == 'continuous':
                domain = [self.vmaxmin[i][1], self.vmaxmin[i][0]]
            elif cont_or_disc[i] == 'discrete':
                domain = []
                a = self.vmaxmin[i][1]
                b = self.vmaxmin[i][0]
                while a < b:
                    domain.append(a)
                    a += stride[i]
                domain.append(self.vmaxmin[i][0])
            bounds.append({'name': axis_name[i], 'type': cont_or_disc[i], 'domain': domain})

        constraint = []
        if constr:
            for j in range(len(constr)):
                for i in range(self.n_axis):
                    constr[j] = constr[j].replace(axis_name[i], 'x[:,'+str(i)+']')
                constraint.append({'name': 'constraint_'+str(j), 'constraint': constr[j]})
        else:
            constraint = constr

        seed(123)
        kern = eval('GPy.kern.'+kernel+'('+str(self.n_axis)+')')
        self.BO_parallel = GPyOpt.methods.BayesianOptimization(f=None,
                                                               domain=bounds,
                                                               acquisition_type=acquisition,
                                                               exact_feval=exact_fval,
                                                               normalize_Y=True,
                                                               X=initial_data[:, :self.n_axis],
                                                               Y=Y_data,
                                                               evaluator_type='local_penalization',
                                                               batch_size=batch_size,
                                                               acquisition_jitter=jwparam,
                                                               acquisition_weight=jwparam,
                                                               constraints=constraint,
                                                               kernel=kern,
                                                               de_duplication=de_duplication,
                                                               )

    def runBO(self):

        suggest_points = self.BO_parallel.suggest_next_locations()

        self.suggest_points = np.round(suggest_points, 3)


    def normalize(self, Y):

        return GPyOpt.util.general.normalize(Y)
