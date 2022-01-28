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

    def setAxis(self, select_axis, unselect_axis):
        self.select_axis = select_axis
        self.unselect_axis = unselect_axis

        self.vmaxmin_s = [[],[],[]]
        self.vmaxmin_s[0] = self.vmaxmin[select_axis[0]]
        self.vmaxmin_s[1] = self.vmaxmin[select_axis[1]]
        self.vmaxmin_s[2] = self.vmaxmin[select_axis[2]]

    def plot_setting(self, resolution, datatype, marker_size):
        self.resolution = resolution
        self.resolution2d = resolution*5
        self.datatype = datatype
        self.marker_size = marker_size

        if self.maximize:
            ind = np.where(np.isclose(self.initial_data[:, -1], self.initial_data[:, -1].max()))
        else:
            ind = np.where(np.isclose(self.initial_data[:, -1], self.initial_data[:, -1].min()))

        self.indata_best = self.initial_data[ind]
        self.indata_wobest = np.delete(self.initial_data, ind, axis=0)

    def type_minmax(self):
        r = int(self.resolution**(3./self.n_axis))
        xyz = [np.linspace(self.vmaxmin[i][1], self.vmaxmin[i][0], r) for i in range(self.n_axis)]
        pp = [np.meshgrid(*tuple(xyz))[i] for i in range(self.n_axis)]
        xyz_shape = 1
        for i in range(self.n_axis):
            xyz_shape *= xyz[i].shape[0]

        ppos = [pp[i].reshape(xyz_shape, 1) for i in range(self.n_axis)]

        pos = np.hstack((ppos))

        model = self.BOpt.BO_parallel.model
        acqu = self.BOpt.BO_parallel.acquisition.acquisition_function(pos)
        self.acqu0 = acqu
        m, v = model.predict(pos)

        M = m
        Y_data = self.BOpt.Y_data
        std = Y_data.std()
        if std > 0:
            M *=  std
        M += Y_data.mean()
        m = M
        v *= std

        if self.datatype == 'mean':
            if self.maximize:
                self.val_max, self.val_min = -m.min(), -m.max()
            else:
                self.val_max, self.val_min = m.max(), m.min()
        elif self.datatype == 'sd':
            self.val_max, self.val_min = v.max(), v.min()
        elif self.datatype == 'acqu':
            self.val_max, self.val_min = 1., 0.
        else:
            self.val_max, self.val_min = None, None

    def setInputData(self, dim_ex, best):
        if best:
            data3d = self.indata_best
            name = 'Input_Best'
            color = '#00ff00'
        else:
            data3d = self.indata_wobest
            name = 'Input'
            color = '#0000ff'
        for i in range(len(self.unselect_axis)):
            ind_3d = np.where(np.isclose(data3d[:, self.unselect_axis[i]], dim_ex[i],
                                         atol=(self.vmaxmin[self.unselect_axis[i]][0] - self.vmaxmin[self.unselect_axis[i]][1])/(5.*self.resolution))
                              )
            data3d = data3d[ind_3d]


        x = data3d[:, self.select_axis[0]]
        y = data3d[:, self.select_axis[1]]
        z = data3d[:, self.select_axis[2]]

        trace_input = go.Scatter3d(x=x, y=y, z=z,
                                   mode='markers',
                                   name=name,
                                   marker = dict(color=color, size=self.marker_size),
                                   legendgroup=name,
                                   showlegend=True,
                                   hovertext=data3d[:,-1],
                                   )

        return trace_input

    def setSuggestData(self, dim_ex):
        text = np.arange(1,self.batch_size+1)
        data3d = self.BOpt.suggest_points
        for i in range(len(self.unselect_axis)):
            ind_3d = np.where(np.isclose(data3d[:, self.unselect_axis[i]], dim_ex[i],
                                         atol=(self.vmaxmin[self.unselect_axis[i]][0] - self.vmaxmin[self.unselect_axis[i]][1])/(5.*self.resolution))
                              )
            data3d = data3d[ind_3d]
        X = data3d[:, self.select_axis[0]]
        Y = data3d[:, self.select_axis[1]]
        Z = data3d[:, self.select_axis[2]]
        trace_suggest = go.Scatter3d(x=X, y=Y, z=Z,
                                     mode='markers',
                                     name='Suggest',
                                     marker = dict(color='#ff0000', size = self.marker_size),
                                     legendgroup='Suggest',
                                     showlegend=True,
                                     # hovertext=text,
                                     )

        return trace_suggest

    def setnullData(self):
        trace_null = go.Scatter3d(x=[self.vmaxmin_s[0][1]],
                                  y=[self.vmaxmin_s[1][1]],
                                  z=[self.vmaxmin_s[2][1]],
                                  name='',
                                  opacity=0.,
                                  showlegend=True,
                                  )

        return trace_null

    def make3Dgraph(self, dim_ex):

        xyz = []
        j = 0
        k = 0
        for i in range(self.n_axis):
            if i in self.unselect_axis:
                xyz.append(np.array([dim_ex[j]]))
                j += 1
            else:
                xyz.append(np.linspace(self.vmaxmin_s[k][1], self.vmaxmin_s[k][0], self.resolution+1))
                k += 1

        pp = [np.meshgrid(*tuple(xyz))[i] for i in range(self.n_axis)]
        xyz_shape = 1
        for i in range(self.n_axis):
            xyz_shape *= xyz[i].shape[0]

        ppos = [pp[i].reshape(xyz_shape, 1) for i in range(self.n_axis)]

        pos = np.hstack((ppos))

        model = self.BOpt.BO_parallel.model
        acqu = self.BOpt.BO_parallel.acquisition.acquisition_function(pos)
        acqu_normalized = (-acqu - min(-self.acqu0))/(max(-self.acqu0 - min(-self.acqu0)))
        m, v = model.predict(pos)

        M = m
        Y_data = self.BOpt.Y_data
        std = Y_data.std()
        if std > 0:
            M *=  std
        M += Y_data.mean()
        m = M
        v *= std

        reshape = []
        for i in range(self.n_axis):
            if i in self.select_axis:
                reshape.append(xyz[i].shape[0])

        if 0 in self.select_axis and 1 in self.select_axis:
            t0, t1 = 0, 1
        else:
            t0, t1 = 1, 0

        if self.maximize:
            self.m = -m.reshape(reshape).transpose(t0,t1,2)
        else:
            self.m = m.reshape(reshape).transpose(t0,t1,2)
        self.v = v.reshape(reshape).transpose(t0,t1,2)
        self.acqu = acqu_normalized.reshape(reshape).transpose(t0,t1,2)

        xx = np.linspace(self.vmaxmin_s[0][1], self.vmaxmin_s[0][0], self.resolution+1)
        yy = np.linspace(self.vmaxmin_s[1][1], self.vmaxmin_s[1][0], self.resolution+1)
        zz = np.linspace(self.vmaxmin_s[2][1], self.vmaxmin_s[2][0], self.resolution+1)
        self.XX, self.YY, self.ZZ = np.meshgrid(xx,yy,zz)


    def setMean(self, m_min=None, m_max=None):
        if m_min == None or m_max == None:
            m_min, m_max = self.val_min, self.val_max
        trace_m = go.Volume(x=self.XX.flatten(),
                            y=self.YY.flatten(),
                            z=self.ZZ.flatten(),
                            value=self.m.flatten(),
                            isomin=m_min,
                            isomax=m_max,
                            opacity=0.2,
                            surface_count=25,
                            name='Mean',
                            colorscale='viridis',
                            showscale=False,
                            reversescale=not(self.maximize)
                            )

        return trace_m


    def setStDev(self, v_min=None, v_max=None):
        if v_min == None or v_max == None:
            v_min, v_max = self.val_min, self.val_max
        trace_v = go.Volume(x=self.XX.flatten(),
                            y=self.YY.flatten(),
                            z=self.ZZ.flatten(),
                            value=self.v.flatten(),
                            isomin=v_min,
                            isomax=v_max,
                            opacity=0.2,
                            surface_count=25,
                            name='StDev',
                            colorscale='inferno',
                            showscale=False,
                            )

        return trace_v


    def setAcqu(self, ac_min=None, ac_max=None):
        if ac_min == None or ac_max == None:
            ac_min, ac_max = self.val_min, self.val_max
        trace_ac = go.Volume(x=self.XX.flatten(),
                             y=self.YY.flatten(),
                             z=self.ZZ.flatten(),
                             value=self.acqu.flatten(),
                             isomin=ac_min,
                             isomax=ac_max,
                             opacity=0.2,
                             surface_count=25,
                             name='Acquisition',
                             colorscale='cividis',
                             showscale=False,
                             )

        return trace_ac

    def setAxis2D(self, display_plane):
        self.display_plane = display_plane
        if display_plane == 'plane23':
            self.select_axis2d = [self.select_axis[1],self.select_axis[2]]
            self.unselect_axis2d = [self.select_axis[0]]
        elif display_plane == 'plane13':
            self.select_axis2d = [self.select_axis[0],self.select_axis[2]]
            self.unselect_axis2d = [self.select_axis[1]]
        elif display_plane == 'plane12':
            self.select_axis2d = [self.select_axis[0],self.select_axis[1]]
            self.unselect_axis2d = [self.select_axis[2]]

        self.vmaxmin_s2d = [[],[]]
        self.vmaxmin_s2d[0] = self.vmaxmin[self.select_axis2d[0]]
        self.vmaxmin_s2d[1] = self.vmaxmin[self.select_axis2d[1]]

    def setInputData2D(self, slicevalue, dim_ex, best):
        if best:
            data3d = self.indata_best
            name = 'Input_Best'
            color = '#00ff00'
        else:
            data3d = self.indata_wobest
            name = 'Input'
            color = '#0000ff'
        for i in range(len(self.unselect_axis)):
            ind_3d = np.where(np.isclose(data3d[:, self.unselect_axis[i]], dim_ex[i],
                                         atol=(self.vmaxmin[self.unselect_axis[i]][0] - self.vmaxmin[self.unselect_axis[i]][1])/(5.*self.resolution))
                              )
            data3d = data3d[ind_3d]

        ind2d = np.where(np.isclose(data3d[:, self.unselect_axis2d[0]], slicevalue,
                                    atol=(self.vmaxmin[self.unselect_axis2d[0]][0] - self.vmaxmin[self.unselect_axis2d[0]][1])/(5.*self.resolution))
                         )
        data2d = data3d[ind2d]

        x = data2d[:, self.select_axis2d[0]]
        y = data2d[:, self.select_axis2d[1]]

        trace_input = go.Scatter(x=x, y=y,
                                 mode='markers',
                                 name=name,
                                 marker=dict(color=color, size=self.marker_size*2),
                                 legendgroup=name,
                                 showlegend=False,
                                 hovertext=data2d[:,-1],
                                 )

        return trace_input


    def setSuggestData2D(self, slicevalue, dim_ex):
        text = np.arange(1,self.batch_size+1)
        data3d = self.BOpt.suggest_points
        for i in range(len(self.unselect_axis)):
            ind_3d = np.where(np.isclose(data3d[:, self.unselect_axis[i]], dim_ex[i],
                                         atol=(self.vmaxmin[self.unselect_axis[i]][0] - self.vmaxmin[self.unselect_axis[i]][1])/(5.*self.resolution))
                              )
            data3d = data3d[ind_3d]

        ind2d = np.where(np.isclose(data3d[:, self.unselect_axis2d[0]], slicevalue,
                                    atol=(self.vmaxmin[self.unselect_axis2d[0]][0] - self.vmaxmin[self.unselect_axis2d[0]][1])/(5.*self.resolution))
                                    )
        data2d = data3d[ind2d]

        X = data2d[:, self.select_axis2d[0]]
        Y = data2d[:, self.select_axis2d[1]]
        trace_suggest = go.Scatter(x=X, y=Y,
                                   mode='markers',
                                   name='Suggest',
                                   marker=dict(color='#ff0000', size=self.marker_size*2),
                                   legendgroup='Suggest',
                                   showlegend=False,
                                   # hovertext=text,
                                   )

        return trace_suggest

    def setnullData2D(self):
        trace_null = go.Scatter(x=[self.vmaxmin_s2d[0][1]],
                                y=[self.vmaxmin_s2d[1][1]],
                                name='',
                                opacity=0.,
                                showlegend=True,
                                )

        return trace_null

    def make2Dgraph(self, slicevalue, dim_ex):

        xyz = []
        j = 0
        k = 0
        for i in range(self.n_axis):
            if i in self.unselect_axis:
                xyz.append(np.array([dim_ex[j]]))
                j += 1
            elif i in self.unselect_axis2d:
                xyz.append(np.array([slicevalue]))
            else:
                xyz.append(np.linspace(self.vmaxmin_s2d[k][1], self.vmaxmin_s2d[k][0], self.resolution2d+1))
                k += 1

        pp = [np.meshgrid(*tuple(xyz))[i] for i in range(self.n_axis)]
        xyz_shape = 1
        for i in range(self.n_axis):
            xyz_shape *= xyz[i].shape[0]

        ppos = [pp[i].reshape(xyz_shape, 1) for i in range(self.n_axis)]

        pos = np.hstack((ppos))

        model = self.BOpt.BO_parallel.model
        acqu = self.BOpt.BO_parallel.acquisition.acquisition_function(pos)
        acqu_normalized = (-acqu - min(-self.acqu0))/(max(-self.acqu0 - min(-self.acqu0)))
        m, v = model.predict(pos)

        M = m
        Y_data = self.BOpt.Y_data
        std = Y_data.std()
        if std > 0:
            M *=  std
        M += Y_data.mean()
        m = M
        v *= std

        reshape = []
        for i in range(self.n_axis):
            if i in self.select_axis:
                reshape.append(xyz[i].shape[0])

        if self.maximize:
            self.m2d = -m.reshape(reshape)
        else:
            self.m2d = m.reshape(reshape)
        self.v2d = v.reshape(reshape)
        self.acqu2d = acqu_normalized.reshape(reshape)

        if self.display_plane == 'plane12':
            self.m2d = self.m2d.T
            self.v2d = self.v2d.T
            self.acqu2d = self.acqu2d.T

        xx = np.linspace(self.vmaxmin_s2d[0][1], self.vmaxmin_s2d[0][0], self.resolution2d+1)
        yy = np.linspace(self.vmaxmin_s2d[1][1], self.vmaxmin_s2d[1][0], self.resolution2d+1)
        self.X2d, self.Y2d = np.meshgrid(xx,yy)


    def setMean2D(self, m_min=None, m_max=None):
        if m_min == None or m_max == None:
            m_min, m_max = self.val_min, self.val_max
        trace_m2d = go.Heatmap(x=self.X2d.flatten(),
                               y=self.Y2d.flatten(),
                               z=self.m2d.T.flatten(),
                               zmin=m_min,
                               zmax=m_max,
                               name='Mean',
                               colorscale='viridis',
                               colorbar=dict(len=0.9, x=1.05),
                               reversescale=not(self.maximize)
                               )

        return trace_m2d


    def setStDev2D(self, v_min=None, v_max=None):
        if v_min == None or v_max == None:
            v_min, v_max = self.val_min, self.val_max
        trace_v2d = go.Heatmap(x=self.X2d.flatten(),
                               y=self.Y2d.flatten(),
                               z=self.v2d.T.flatten(),
                               zmin=v_min,
                               zmax=v_max,
                               name='StDev',
                               colorscale='inferno',
                               colorbar=dict(len=0.9, x=1.05),
                               )

        return trace_v2d


    def setAcqu2D(self, ac_min=None, ac_max=None):
        if ac_min == None or ac_max == None:
            ac_min, ac_max = self.val_min, self.val_max
        trace_ac2d = go.Heatmap(x=self.X2d.flatten(),
                                y=self.Y2d.flatten(),
                                z=self.acqu2d.T.flatten(),
                                zmin=ac_min,
                                zmax=ac_max,
                                name='Acquisition',
                                colorscale='cividis',
                                colorbar=dict(len=0.9, x=1.05),
                                )

        return trace_ac2d


    def setPlane(self, slicevalue):

        if self.display_plane == 'plane23':
            x = slicevalue*np.ones(self.resolution)
            y = np.linspace(self.vmaxmin_s2d[0][1], self.vmaxmin_s2d[0][0], self.resolution)
            z = self.vmaxmin_s2d[1][1]*np.ones(self.resolution)
            for i in range(self.resolution-1):
                z = np.vstack([z, self.vmaxmin_s2d[1][1]+(self.vmaxmin_s2d[1][0]-self.vmaxmin_s2d[1][1])/(self.resolution-1)*(i+1)*np.ones(self.resolution)])

            trace = go.Surface(x=x,
                               y=y,
                               z=z.T,
                               showlegend=False,
                               showscale=False,
                               colorscale=[[0, '#ff0000'], [1, '#ff0000']],
                               opacity=0.5,
                               )

        elif self.display_plane == 'plane13':
            x = np.linspace(self.vmaxmin_s2d[0][1], self.vmaxmin_s2d[0][0], self.resolution)
            y = slicevalue*np.ones(self.resolution)
            z = self.vmaxmin_s2d[1][1]*np.ones(self.resolution)
            for i in range(self.resolution-1):
                z = np.vstack([z, self.vmaxmin_s2d[1][1]+(self.vmaxmin_s2d[1][0]-self.vmaxmin_s2d[1][1])/(self.resolution-1)*(i+1)*np.ones(self.resolution)])

            trace = go.Surface(x=x,
                               y=y,
                               z=z,
                               showlegend=False,
                               showscale=False,
                               colorscale=[[0, '#ff0000'], [1, '#ff0000']],
                               opacity=0.5,
                               )

        elif self.display_plane == 'plane12':
            x = np.linspace(self.vmaxmin_s2d[0][1], self.vmaxmin_s2d[0][0], self.resolution)
            y = np.linspace(self.vmaxmin_s2d[1][1], self.vmaxmin_s2d[1][0], self.resolution)
            z = slicevalue*np.ones((self.resolution,self.resolution))

            trace = go.Surface(x=x,
                               y=y,
                               z=z,
                               showlegend=False,
                               showscale=False,
                               colorscale=[[0, '#ff0000'], [1, '#ff0000']],
                               opacity=0.5,
                               )
        return trace
