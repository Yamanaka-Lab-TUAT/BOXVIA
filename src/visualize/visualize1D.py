# Copyright (c) 2021, BOXVIA Developers
# All rights reserved.
# Code released under the BSD 3-clause license.

import numpy as np
import plotly.graph_objects as go

from runBO import BO

class visualize(BO):

    def __init__(self, axis_name, n_axis):

        self.axis_name = axis_name
        self.n_axis = n_axis

    def initial_vis(self, initial_data):

        self.initial_data = initial_data

    def runBO_visualize(self, vmaxmin, acquisition, exact_fval, de_duplication, batch_size, jwparam, constraint, kernel, maximize, cont_or_disc, stride):

        self.vmaxmin = vmaxmin
        self.batch_size = batch_size
        self.maximize = maximize

        self.BOpt = BO(self.n_axis, vmaxmin)
        self.BOpt.setBO(self.initial_data, self.axis_name, acquisition, maximize, exact_fval, de_duplication, jwparam, batch_size, kernel, [cont_or_disc], [stride], constraint)
        self.BOpt.runBO()

    def makegraph(self):

        if self.maximize:
            ind = np.where(np.isclose(self.initial_data[:, -1], self.initial_data[:, -1].max()))
        else:
            ind = np.where(np.isclose(self.initial_data[:, -1], self.initial_data[:, -1].min()))

        self.indata_best = self.initial_data[ind]
        self.indata_wobest = np.delete(self.initial_data, ind, axis=0)

        xyz = [np.linspace(self.vmaxmin[0][1], self.vmaxmin[0][0], 1001)]

        pp = [np.meshgrid(*tuple(xyz))[0]]
        xyz_shape = xyz[0].shape[0]

        ppos = [pp[0].reshape(xyz_shape, 1)]
        pos = np.hstack((ppos))

        model = self.BOpt.BO_parallel.model
        acqu = self.BOpt.BO_parallel.acquisition.acquisition_function(pos)
        acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
        m, v = model.predict(pos)

        M = m
        Y_data = self.BOpt.Y_data
        std = Y_data.std()
        if std > 0:
            M *=  std
        M += Y_data.mean()
        m = M
        v *= std

        reshape = [xyz[0].shape[0]]

        if self.maximize:
            self.m = -m.reshape(reshape)
        else:
            self.m = m.reshape(reshape)
        self.v = v.reshape(reshape)
        self.acqu = acqu_normalized.reshape(reshape)

        self.XX = np.linspace(self.vmaxmin[0][1], self.vmaxmin[0][0], 1001)

    def setInputData(self, marker_size, best):
        if best:
            data1d = self.indata_best
            name = 'Input_Best'
            color = '#00ff00'
        else:
            data1d = self.indata_wobest
            name = 'Input'
            color = '#0000ff'

        x = data1d[:, 0]
        y = data1d[:, -1]

        trace_input = go.Scatter(x=x, y=y,
                                 mode='markers',
                                 name=name,
                                 marker = dict(color=color, size=marker_size),
                                 showlegend=True,
                                 # hovertext=data1d[:,-1],
                                 )

        return trace_input

    def setSuggestData(self, i):
        data1d = self.BOpt.suggest_points

        X = (data1d[:, 0])

        return X[i]

    def setMean(self):
        trace = go.Scatter(x=self.XX,
                           y=self.m,
                           mode='lines',
                           line = dict(width=4),
                           name='Mean',
                           )

        return trace

    def setStDevUp(self):
        trace = go.Scatter(x=self.XX,
                           y=self.m+self.v,
                           mode='lines',
                           line = dict(width=0),
                           marker=dict(color="#FF69B4"),
                           name='Mean+StDev',
                           showlegend=False,
                           )

        return trace

    def setStDevDown(self):
        trace = go.Scatter(x=self.XX,
                           y=self.m-self.v,
                           mode='lines',
                           line = dict(width=0),
                           marker=dict(color="#FF69B4"),
                           name='Mean-StDev',
                           fillcolor='rgba(255,105,180, 0.2)',
                           fill='tonexty',
                           showlegend=False
                           )

        return trace


    def setAcqu(self):
        trace_ac = go.Scatter(x=self.XX,
                              y=self.acqu,
                              mode='lines',
                              line = dict(color='#ff0000', width=4),
                              name='Acquisition',
                              )

        return trace_ac
