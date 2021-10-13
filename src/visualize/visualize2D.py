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
        self.BOpt.setBO(self.initial_data, self.axis_name, acquisition, maximize, exact_fval, de_duplication, jwparam, batch_size, kernel, cont_or_disc, stride, constraint)
        self.BOpt.runBO()

    def makegraph(self, resolution):

        if self.maximize:
            ind = np.where(np.isclose(self.initial_data[:, -1], self.initial_data[:, -1].max()))
        else:
            ind = np.where(np.isclose(self.initial_data[:, -1], self.initial_data[:, -1].min()))

        self.indata_best = self.initial_data[ind]
        self.indata_wobest = np.delete(self.initial_data, ind, axis=0)

        xyz = [np.linspace(self.vmaxmin[0][1], self.vmaxmin[0][0], resolution+1),
               np.linspace(self.vmaxmin[1][1], self.vmaxmin[1][0], resolution+1)]

        pp = [np.meshgrid(*tuple(xyz))[0], np.meshgrid(*tuple(xyz))[1]]
        xyz_shape = xyz[0].shape[0]*xyz[1].shape[0]

        ppos = [pp[0].reshape(xyz_shape, 1), pp[1].reshape(xyz_shape, 1)]
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

        reshape = [xyz[0].shape[0], xyz[1].shape[0]]

        if self.maximize:
            self.m = -m.reshape(reshape)
        else:
            self.m = m.reshape(reshape)
        self.v = v.reshape(reshape)
        self.acqu = acqu_normalized.reshape(reshape)

        self.XX = np.linspace(self.vmaxmin[0][1], self.vmaxmin[0][0], resolution+1)
        self.YY = np.linspace(self.vmaxmin[1][1], self.vmaxmin[1][0], resolution+1)


    def setInputData(self, marker_size, best, legend):
        if best:
            data2d = self.indata_best
            name = 'Input_Best'
            color = '#00ff00'
        else:
            data2d = self.indata_wobest
            name = 'Input'
            color = '#0000ff'

        x = data2d[:, 0]
        y = data2d[:, 1]

        trace_input = go.Scatter(x=x, y=y,
                                 mode='markers',
                                 name=name,
                                 marker = dict(color=color, size=marker_size),
                                 legendgroup=name,
                                 showlegend=legend,
                                 # hovertext=data2d[:,-1],
                                 )

        return trace_input

    def setSuggestData(self, marker_size, legend):
        data2d = self.BOpt.suggest_points

        X = (data2d[:, 0])
        Y = (data2d[:, 1])

        trace_suggest = go.Scatter(x=X, y=Y,
                                   mode='markers',
                                   name='Suggest',
                                   marker = dict(color='#ff0000', size=marker_size),
                                   legendgroup='Suggest',
                                   showlegend=legend,
                                   # hovertext=text,
                                   )

        return trace_suggest

    def setMean(self):
        trace_m = go.Heatmap(x=self.XX,
                             y=self.YY,
                             z=self.m,
                             zmin=self.m.min(),
                             zmax=self.m.max(),
                             name='Mean',
                             colorscale='viridis',
                             colorbar=dict(len=0.45, x=0.425, y=0.79),
                             reversescale=not(self.maximize)
                             )
        return trace_m

    def setStDev(self):
        trace_v = go.Heatmap(x=self.XX,
                             y=self.YY,
                             z=self.v,
                             zmin=self.v.min(),
                             zmax=self.v.max(),
                             name='StDev',
                             colorscale='inferno',
                             colorbar=dict(len=0.45, x=1.0, y=0.79),
                             )

        return trace_v


    def setAcqu(self):
        trace_ac = go.Heatmap(x=self.XX,
                              y=self.YY,
                              z=self.acqu,
                              zmin=self.acqu.min(),
                              zmax=self.acqu.max(),
                              name='Acquisition',
                              colorscale='cividis',
                              colorbar=dict(len=0.45, x=0.425, y=0.215),
                              )

        return trace_ac


    def setInputData3D(self, marker_size, best, legend):
        if best:
            data2d = self.indata_best
            name = 'Input_Best'
            color = '#00ff00'
        else:
            data2d = self.indata_wobest
            name = 'Input'
            color = '#0000ff'

        x = data2d[:, 0]
        y = data2d[:, 1]
        z = data2d[:,-1]

        trace_input = go.Scatter3d(x=x, y=y, z=z,
                                   mode='markers',
                                   name=name,
                                   marker = dict(color=color, size=marker_size/4),
                                   legendgroup=name,
                                   showlegend=legend,
                                   # hovertext=data2d[:,-1],
                                   )

        return trace_input

    def setSuggestData3D(self, marker_size, legend, type):
        data2d = self.BOpt.suggest_points

        X = np.hstack([data2d[0, 0],data2d[0, 0]])
        Y = np.hstack([data2d[0, 1],data2d[0, 1]])

        if type == 'mean':
            mmin = self.m.min()
            mmax = self.m.max()
        elif type == 'sd':
            mmin = self.v.min()
            mmax = self.v.max()
        elif type == 'acqu':
            mmin = 0
            mmax = 1

        Z = [mmin, mmax]

        trace_suggest = go.Scatter3d(x=X, y=Y, z=Z,
                                     mode='lines',
                                     name='First suggest',
                                     line=dict(width=2, color='#ff0000'),
                                     legendgroup='Suggest',
                                     showlegend=legend,
                                     hoverinfo='none',
                                     )

        return trace_suggest


    def setMean3D(self):
        trace_m3d = go.Surface(x=self.XX,
                               y=self.YY,
                               z=self.m,
                               cmin=self.m.min(),
                               cmax=self.m.max(),
                               name='Mean',
                               colorscale='viridis',
                               colorbar=dict(len=0.45, x=0.425, y=0.79),
                               reversescale=not(self.maximize),
                               contours_z=dict(project_z=True)
                               )

        return trace_m3d


    def setStDev3D(self):
        trace_v3d = go.Surface(x=self.XX,
                               y=self.YY,
                               z=self.v,
                               cmin=self.v.min(),
                               cmax=self.v.max(),
                               name='StDev',
                               colorscale='inferno',
                               colorbar=dict(len=0.45, x=1.0, y=0.79),
                               )

        return trace_v3d


    def setAcqu3D(self):
        trace_ac3d = go.Surface(x=self.XX,
                                y=self.YY,
                                z=self.acqu,
                                cmin=self.acqu.min(),
                                cmax=self.acqu.max(),
                                name='Acquisition',
                                colorscale='cividis',
                                colorbar=dict(len=0.45, x=0.425, y=0.215),
                                )

        return trace_ac3d
