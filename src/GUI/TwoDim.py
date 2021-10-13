# Copyright (c) 2021, BOXVIA Developers
# All rights reserved.
# Code released under the BSD 3-clause license.

import os
import sys
import csv
import numpy as np

import base64
import io
import pandas as pd

import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from base import app
from visualize.visualize2D import visualize

if getattr(sys, 'frozen', False):
    program_directory = os.path.dirname(os.path.abspath(sys.executable))
    if os.getcwd() == '/':
        program_directory = os.path.dirname(os.path.dirname(os.path.dirname(program_directory)))
else:
    program_directory = os.path.dirname(os.path.abspath(__file__))[:-4]

n_max = 2
n_axis = n_max

axis_range0 = [dbc.Row([dbc.Col(html.Div(str(i+1)+'-axis',
                                         id="dataname-2D_"+str(i),
                                         style={'width': '100%', 'color': 'Red'},
                                         className='mt-1',
                                         ),
                                width=2,
                                ),
                        dbc.Col(dbc.Button('Add',
                                           id="add_c-2D_"+str(i),
                                           n_clicks=0,
                                           color='warning',
                                           size='sm',
                                           ),
                                width=1,
                                ),
                        dbc.Col(html.Div('Min:',
                                         style={'width': '100%', 'text-align': 'right'},
                                         className='mt-1',
                                         ),
                                width=1,
                                ),
                        dbc.Col(html.Div(dcc.Input(id="datamin-2D_"+str(i),
                                                   type='number',
                                                   value='',
                                                   style={'width': '150%'},
                                                   ),
                                         ),
                                width=1,
                                ),
                        dbc.Col(html.Div('Max:',
                                         style={'width': '100%', 'text-align': 'right'},
                                         className='mt-1',
                                         ),
                                width=1,
                                ),
                        dbc.Col(html.Div(dcc.Input(id="datamax-2D_"+str(i),
                                                   type='number',
                                                   value='',
                                                   style={'width': '150%'},
                                                   ),
                                         ),
                                width=1,
                                ),
                        dbc.Col(dcc.Dropdown(id="cont_or_disc-2D_"+str(i),
                                             options=[{'label': 'Continuous', 'value': 'continuous'},
                                                      {'label': 'Discrete',   'value': 'discrete'},
                                                      ],
                                             value='continuous',
                                             clearable=False,
                                             ),
                                width={'size': 2, 'offset': 1},
                                ),
                        dbc.Col(html.Div(dcc.Input(id="interval-2D_"+str(i),
                                                   type='number',
                                                   value=1,
                                                   min=0,
                                                   style={'width': '100%'},
                                                   ),
                                         ),
                                width=1
                                ),
                      ],
                  className='mt-3',
                  ) for i in range(n_max)]

layout = dbc.Container([
    dbc.Row([dbc.Col([html.H6('Batch size',
                              className='mt-1',
                              style={'display': 'inline-block'},
                              ),
                      dbc.Badge('?',
                                color='light',
                                className='ml-1',
                                pill=True,
                                id="batch_text-2D_info",
                                style={'display': 'inline-block', 'margin-bottom': '3px'},
                                ),
                      dbc.Tooltip('Number of candidate points to be proposed by BO (using Local Penalization)',
                                  target="batch_text-2D_info",
                                  placement='top',
                                  ),
                      ],
                     width=2,
                     ),
             dbc.Col(dcc.Input(id="batch_size-2D",
                               type='number',
                               value=1,
                               min=1,
                               style={'width': '100%'},
                               ),
                     width=2,
                     ),
             ],
             className='mt-2'
            ),
    dbc.Row([dbc.Col(html.H6('Acquisition type'),
                     width=2,
                     ),
             dbc.Col(dbc.RadioItems(id="actype-2D",
                                    options=[{'label': 'EI',  'value': 'EI'},
                                             {'label': 'LCB', 'value': 'LCB'}
                                             ],
                                    value='EI',
                                    inline=True,
                                    ),
                     width=4,
                     ),
             ],
             className='mt-2'
            ),
    dbc.Row([dbc.Col([html.H6('Jitter or Weight',
                              className='mt-1',
                              id="jwname-2D",
                              style={'display': 'inline-block'},
                              ),
                      dbc.Badge('?',
                                color='light',
                                className='ml-1',
                                pill=True,
                                id="jwname-2D_info",
                                style={'display': 'inline-block', 'margin-bottom': '3px'},
                                ),
                      dbc.Tooltip('Determine the balance between exploration and exploitation (Large number encourages exploration).',
                                  target="jwname-2D_info",
                                  placement='top',
                                  ),
                      ],
                     width=2,
                     ),
             dbc.Col(dcc.Input(id="jwparam-2D",
                               type='number',
                               min=0,
                               style={'width': '100%'},
                               ),
                     width=2,
                     ),
             ],
             className='mt-2'
            ),
    dbc.Row([dbc.Col([html.H6('Kernel function',
                              className='mt-1',
                              style={'display': 'inline-block'},
                              ),
                      dbc.Badge('?',
                                color='light',
                                className='ml-1',
                                pill=True,
                                id="kernel-2D_info",
                                style={'display': 'inline-block', 'margin-bottom': '3px'},
                                ),
                      dbc.Tooltip('Select the type of kernel function to be used for Gaussian process regression. Matern52 or RBF are common.',
                                  target="kernel-2D_info",
                                  placement='top',
                                  ),
                      ],
                      width=2,
                     ),
             dbc.Col(dcc.Dropdown(id="kern-dropdown-2D",
                                  options=[{'label': 'Exponential', 'value': 'Exponential'},
                                           {'label': 'Linear', 'value': 'Linear'},
                                           {'label': 'Matern 3/2', 'value': 'Matern32'},
                                           {'label': 'Matern 5/2', 'value': 'Matern52'},
                                           {'label': 'Radial Basis Function (RBF)', 'value': 'RBF'},
                                           {'label': 'Rational Quadratic', 'value': 'RatQuad'},
                                           {'label': 'Standard Periodic', 'value': 'StdPeriodic'},
                                           ],
                                  value='Matern52',
                                  clearable=False,
                                  ),
                     width=4,
                     ),
             ],
             className='mt-2'
            ),
    dbc.Row([dbc.Col([dbc.Checklist(id="maximize-2D",
                                    options=[{'label': 'Maximize', 'value': 'val_max'}],
                                    value=[],
                                    style={'display': 'inline-block'},
                                    ),
                      dbc.Badge('?',
                                color='light',
                                className='ml-1',
                                pill=True,
                                id="maximize-2D_info",
                                style={'display': 'inline-block'},
                                ),
                      dbc.Tooltip('Check the box if objective parameter is to be maximazed.',
                                  target="maximize-2D_info",
                                  placement='top',
                                  ),
                       ],
                      width=2,
                      ),
             dbc.Col([dbc.Checklist(id="exact_fval-2D",
                                    options=[{'label': 'Noiseless', 'value': 'val_efval'}],
                                    value=[],
                                    style={'display': 'inline-block'},
                                    ),
                      dbc.Badge('?',
                                color='light',
                                className='ml-1',
                                pill=True,
                                id="exact_fval-2D_info",
                                style={'display': 'inline-block'},
                                ),
                      dbc.Tooltip('Check the box if noiseless evaluation is available.',
                                  target="exact_fval-2D_info",
                                  placement='top',
                                  ),
                      ],
                     width=2,
                     ),
              dbc.Col([dbc.Checklist(id="de_duplication-2D",
                                     options=[{'label': 'Avoid re-evaluating', 'value': 'dup'}],
                                     value=[],
                                     style={'display': 'inline-block'},
                                     ),
                       dbc.Badge('?',
                                 color='light',
                                 className='ml-1',
                                 pill=True,
                                 id="de_duplication-2D_info",
                                 style={'display': 'inline-block'},
                                 ),
                       dbc.Tooltip('Check the box if the location where the data already exists is not to be re-evaluated.',
                                   target="de_duplication-2D_info",
                                   placement='top',
                                   ),
                       ],
                      width=3,
                      ),
             ],
             className='mt-2'
            ),
    dbc.Row([dbc.Col([html.H6('Constraints',
                              className='mt-1',
                              style={'display': 'inline-block'},
                              ),
                      dbc.Badge('?',
                                color='light',
                                className='ml-1',
                                pill=True,
                                id="constraint-2D_info",
                                style={'display': 'inline-block', 'margin-bottom': '3px'},
                                ),
                      dbc.Tooltip(['Constraints can be added as inequalities such that the left-hand side is less than or equal to zero. ',
                                   'Multiple constraints are added by writing them on new lines.'],
                                  target="constraint-2D_info",
                                  placement='bottom',
                                  ),
                      ],
                     width=2,
                     ),
             dbc.Col(dcc.Textarea(id="constraint-2D",
                                  value='',
                                  style={'width': '105%', 'height': '100px'}),
                     width=4,
                     ),
             dbc.Col(html.H6(' <= 0',
                             className='mt-1',
                             style={'height': '100px', 'display': 'flex', 'align-items': 'center'},
                             ),
                     width=2,
                     ),
             ],
             className='mt-3'
            ),
    dbc.Row([dbc.Col(html.H6('Axis name',
                             className='mt-2',
                             ),
                     width=2,
                     ),
             dbc.Col(html.H6('Const.',
                             className='mt-2',
                             ),
                     width=1,
                     ),
             dbc.Col([html.H6('Range',
                              className='mt-2',
                              style={'display': 'inline-block'},
                              ),
                      dbc.Badge('?',
                                color='light',
                                className='ml-1',
                                pill=True,
                                id="range-2D_info",
                                style={'display': 'inline-block', 'margin-bottom': '3px'},
                                ),
                      dbc.Tooltip('Set the range for each parameter. The set range can be saved & loaded.',
                                  target="range-2D_info",
                                  placement='top',
                                  ),
                      ],
                     style={'text-align': 'right'},
                     width=2,
                     ),
             dbc.Col(dbc.Button('Load',
                                id="button_load-2D",
                                n_clicks=0,
                                color='success',
                                ),
                     width=1,
                     ),
             dbc.Col(dbc.Button('Save',
                                id="button_save-2D",
                                n_clicks=0,
                                color='info'
                                ),
                     width=1,
                     ),
             dbc.Col(html.H6('',
                             className='mt-2',
                             id="save_label-2D",
                             style={'width': '100%', 'color': 'Red'},
                             ),
                     width=1,
                     ),
             dbc.Col(html.H6('Type',
                             className='mt-2',
                             style={'width': '100%'},
                             ),
                     width=2,
                     ),
             dbc.Col([html.H6('Interval',
                              className='mt-2',
                              style={'display': 'inline-block'},
                              ),
                      dbc.Badge('?',
                                color='light',
                                className='ml-1',
                                pill=True,
                                id="interval-2D_info",
                                style={'display': 'inline-block', 'margin-bottom': '3px'},
                                ),
                      dbc.Tooltip('Set the discretizing interval for each parameter.',
                                  target="interval-2D_info",
                                  placement='top',
                                  ),
                       ],
                      width='auto',
                      ),
             ],
             className='mt-2'
            ),
    html.Div(children=axis_range0),
    html.P(),
    html.Center(dbc.Button('Run Bayesian Optimization',
                           id="button_runBO-2D",
                           n_clicks=0,
                           style={'width': '100%'},
                           className='mt-3',
                           color='primary',
                           size='lg',
                           ),
                ),
    html.Center([dcc.Loading(id="run_loading-2D",
                             type='default',
                             children=html.Center('',
                                                  id="done_run-2D",
                                                  style={'width': '100%', 'height': '20px', 'color': 'Red'},
                                                  ),
                            ),
                ],
                className='mt-3',
                ),
    html.P(),
    dbc.Row([dbc.Col([html.H5('Suggest data table',
                              style={'display': 'inline-block'},
                              ),
                      dbc.Badge('?',
                                color='light',
                                className='ml-1',
                                pill=True,
                                id="suggest_text-2D_info",
                                style={'display': 'inline-block', 'margin-bottom': '4px'},
                                ),
                      dbc.Tooltip('The table is editable.',
                                  target="suggest_text-2D_info",
                                  placement='top',
                                  ),
                      ],
                      width='auto',
                      ),
             dbc.Col([dbc.Button('Export',
                                 id="export_suggest-2D",
                                 color='dark',
                                 outline=True,
                                 size='sm',
                                 ),
                      dbc.Tooltip('Download the table data as a Suggest.csv.',
                                  target="export_suggest-2D",
                                  placement='top',
                                  ),
                      dcc.Download(id="download_suggest-2D"),
                      ],
                      style={'margin-top': '4px'},
                      width='auto',
                      ),
             ],
            justify='between',
            ),
    dash_table.DataTable(id={'type': "suggest_table", 'index': 2},
                         style_table={'height': '200px', 'overflowY': 'auto'},
                         style_header={'fontWeight': 'bold'},
                         editable=True,
                         css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
                         ),
    dbc.Row(dbc.Col([dbc.Button(html.Div('Add suggests to imported data table',
                                         id="add-2D_info",
                                         ),
                                id={'type': "add", 'index': 2},
                                color='primary',
                                outline=True,
                                ),
                     dbc.Tooltip('Add the suggest table data to the bottom of the input data table.',
                                 target="add-2D_info",
                                 placement='bottom',
                                 ),
                      ],
                     width='auto',
                     ),
            justify='end'
            ),
    html.Hr(),
    dbc.Row([dbc.Col(html.H6('Resolution',
                             className='mt-1',
                             ),
                     width=2,
                     ),
             dbc.Col(dcc.Input(id="resolution-2D",
                               type='number',
                               value=100,
                               step=10,
                               min=10, max=1000,
                               style={'width': '100%'},
                               ),
                     width=2,
                     ),
             dbc.Col(html.H6('Marker size',
                             className='mt-1',
                             ),
                     width={"size": 2, "offset": 1},
                     ),
             dbc.Col(dcc.Input(id="marker_size-2D",
                               type='number',
                               value=10,
                               min=1, max=100,
                               style={'width': '100%'},
                               ),
                     width=2,
                     ),
             ],
             className='mt-2'
            ),
    html.P(),
    dbc.Checklist(id="3Dsurface",
                  options=[{'label': '3D surface', 'value': 'surface'}],
                  value=[],
                  switch=True,
                  ),
    html.Center(dcc.Loading(id="make2D_loading",
                            type='circle',
                            children=html.Div(id="graphspace_2D",
                                              children=html.Div(''),
                                              style={'height': '800px'},
                                              ),
                            ),
                ),
    dbc.Row(dbc.Col([dbc.Button('Export figure data',
                               color='dark',
                               outline=True,
                               size='sm',
                               id="export_figdata-2D",
                               ),
                     dbc.Tooltip('Export the figure data as Figure_data.csv.',
                                 target="export_figdata-2D",
                                 placement='top',
                                 ),
                     dcc.Download(id="download_figdata-2D"),
                     ],
                    width='auto',
                    ),
            justify='end',
            ),
    html.P(),
    ],
)

@app.callback([Output("jwname-2D", 'children'),
               Output("jwparam-2D", 'value')],
              [Input("actype-2D", 'value')]
              )
def jwlabel(actype):
    if actype == 'EI':
        return ['Jitter', 0.01]
    elif actype == 'LCB':
        return ['Weight', 2]


@app.callback([Output("dataname-2D_"+str(i), 'children') for i in range(n_axis)],
              [Output({'type': "suggest_table", 'index': 2}, 'columns'),
               Output({'type': "suggest_table", 'index': 2}, 'style_cell'),
               Output({'type': "suggest_table", 'index': 2}, 'style_data_conditional')],
              [Input("input_table", 'columns')],
              )
def read_initdata(columns):
    global vis, axis_name

    axis_name = [columns[i]['name'] for i in range(len(columns)-1)]
    if not len(columns)-1 == 2:
        axis_name = ['', '']

    style_cell=[{'width': '{}%'.format(len(columns)),
                'textOverflow': 'ellipsis',
                'overflow': 'hidden',
                }]

    style_data_conditional=[[{'if': {'column_id': columns[-1]['name']},
                              'backgroundColor': 'rgba(153,204,255,0.2)',
                              },
                             ]]

    vis = visualize(axis_name, n_axis)

    return axis_name+[columns]+style_cell+style_data_conditional


@app.callback(Output("constraint-2D", 'value'),
              [Input("add_c-2D_"+str(i), 'n_clicks') for i in range(n_axis)],
              State("constraint-2D", 'value'),
              State("input_table", 'columns'),
              prevent_initial_call=True)
def add_c(n_clicks0, n_clicks1, constraint, columns):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if "add_c-2D_0" in changed_id:
        return constraint+columns[0]['name']
    if "add_c-2D_1" in changed_id:
        return constraint+columns[1]['name']

@app.callback([Output("interval-2D_"+str(i), 'disabled') for i in range(n_axis)],
              [Input("cont_or_disc-2D_"+str(i), 'value') for i in range(n_axis)],
              )
def interval_disable(*args):
    cont_or_disc = list(args)

    return [cont_or_disc[0] == 'continuous', cont_or_disc[1] == 'continuous']

@app.callback([Output("datamax-2D_"+str(i), 'value') for i in range(n_axis)],
              [Output("datamin-2D_"+str(i), 'value') for i in range(n_axis)],
              [Input("button_load-2D", 'n_clicks')],
              [State("upload-data", 'filename')],
              prevent_initial_call=True)
def load_range(n_click, filename):

    try:
        filename = filename[:-4]
        is_file = os.path.isfile(program_directory+'/'+filename+'_range.csv')
        if is_file:

            maxmin = np.loadtxt(program_directory+'/'+filename+'_range.csv', delimiter=',', skiprows=1, usecols=[1, 2])
            maxmin = maxmin.T

            range_max = [maxmin[i][0] for i in range(n_axis)]
            range_min = [maxmin[i][1] for i in range(n_axis)]

            return range_max+range_min

        else:
            range_max = ['' for i in range(n_max)]
            range_min = range_max
            return range_max+range_min

    except:
         range_max = ['' for i in range(n_max)]
         range_min = range_max
         return range_max+range_min


@app.callback(Output("save_label-2D", 'children'),
              [Input("button_save-2D", 'n_clicks')],
              [State("upload-data", 'filename')],
              [State("datamax-2D_"+str(i), 'value') for i in range(n_axis)],
              [State("datamin-2D_"+str(i), 'value') for i in range(n_axis)],
              prevent_initial_call=True)
def save_range(n_click, filename, *args):

    try:
        filename = filename[:-4]

        maxmin = []
        for i in range(n_axis):
            value_max, value_min = float(args[i]), float(args[i+n_max])
            maxmin.append([value_max, value_min])

        header = [header[:] for header in axis_name]
        header.insert(0,'')
        maxrow = [str(maxrow[0]) for maxrow in maxmin]
        maxrow.insert(0,'Max')
        minrow = [str(minrow[1]) for minrow in maxmin]
        minrow.insert(0,'Min')
        savedata = [header, maxrow, minrow]
        with open(program_directory+'/'+filename+'_range.csv','w') as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(savedata)

        return 'Saved!'

    except:
        return 'Error!'


@app.callback([Output("done_run-2D", 'children'),
               Output({'type': "suggest_table", 'index': 2}, 'data')],
              [Input("button_runBO-2D", 'n_clicks')],
              [State("input_table", 'data'),
               State("input_table", 'columns'),
               State("batch_size-2D", 'value'),
               State("actype-2D", 'value'),
               State("jwparam-2D", 'value'),
               State("kern-dropdown-2D", 'value'),
               State("exact_fval-2D", 'value'),
               State("de_duplication-2D", 'value'),
               State("maximize-2D", 'value'),
               State("constraint-2D", 'value')],
              [State("datamax-2D_"+str(i), 'value') for i in range(n_axis)],
              [State("datamin-2D_"+str(i), 'value') for i in range(n_axis)],
              [State("cont_or_disc-2D_"+str(i), 'value') for i in range(n_axis)],
              [State("interval-2D_"+str(i), 'value') for i in range(n_axis)],
              prevent_initial_call=True)
def runBO(n_click, data, columns, batch_size, actype, jwparam, kernel, exact_fval, de_duplication, maximize, constraint, *args):
    global vmaxmin

    try:
        df = pd.DataFrame(data=data, columns=[columns[i]['name'] for i in range(len(columns))])
        vmaxmin = []
        cont_or_disc = []
        interval = []
        for i in range(n_axis):
            value_max, value_min = float(args[i]), float(args[i+n_max])
            vmaxmin.append([value_max, value_min])
            cont_or_disc.append(args[i+2*n_max])
            interval.append(args[i+3*n_max])

        exact_fval = (exact_fval == ['val_efval'])
        de_duplication = (de_duplication == ['dup'])
        maximize = (maximize == ['val_max'])

        vis.initial_vis(df.values)
        vis.runBO_visualize(vmaxmin, actype, exact_fval, de_duplication, batch_size, jwparam, constraint.splitlines(), kernel, maximize, cont_or_disc, interval)

        suggest_points = vis.BOpt.suggest_points
        results = [[''] for i in range(batch_size)]
        suggest_points = np.hstack([suggest_points, results])
        index = [str(i) for i in range(batch_size)]
        data = pd.DataFrame(data=suggest_points, index=index, columns=df.columns)

        done_text = '----- Optimization Done ('+str(n_click)+' th trial) -----'

        return [done_text, data.to_dict('records')]

    except:

        return ['Error!', None]


@app.callback(Output("graphspace_2D", 'children'),
              [Input("done_run-2D", 'children'),
               Input("resolution-2D", 'value'),
               Input("marker_size-2D", 'value'),
               Input("3Dsurface", 'value')],
              [State("input_table", 'data'),
               State("input_table", 'columns'),
               State("batch_size-2D", 'value')],
              )
def make2D(txt, resolution, marker_size, surface, data, columns, batch_size):
    if txt:
        df = pd.DataFrame(data=data, columns=[columns[i]['name'] for i in range(len(columns))])

        vis.initial_vis(df.values)
        vis.makegraph(resolution)

        if not surface == ['surface']:
            fig_m = [vis.setInputData(marker_size, best=True, legend=True),
                     vis.setInputData(marker_size, best=False, legend=True),
                     vis.setSuggestData(marker_size, True)]
            fig_v = [vis.setInputData(marker_size, best=True, legend=False),
                     vis.setInputData(marker_size, best=False, legend=False),
                     vis.setSuggestData(marker_size, False)]
            fig_ac = [vis.setInputData(marker_size, best=True, legend=False),
                      vis.setInputData(marker_size, best=False, legend=False),
                      vis.setSuggestData(marker_size, False)]

            fig_m.append(vis.setMean())
            fig_v.append(vis.setStDev())
            fig_ac.append(vis.setAcqu())
            spec = [[{}, {}],[{}, None]]
        else:
            fig_m = [vis.setInputData3D(marker_size, best=True, legend=True),
                     vis.setInputData3D(marker_size, best=False, legend=True),
                     vis.setSuggestData3D(marker_size, True, 'mean')]
            fig_v = [vis.setSuggestData3D(marker_size, False, 'sd')]
            fig_ac = [vis.setSuggestData3D(marker_size, False, 'acqu')]

            fig_m.append(vis.setMean3D())
            fig_v.append(vis.setStDev3D())
            fig_ac.append(vis.setAcqu3D())
            spec = [[{'type': 'scene'}, {'type': 'scene'}],[{'type': 'scene'}, None]]

        fig = make_subplots(rows=2, cols=2,
                            specs=spec,
                            horizontal_spacing=0.15,
                            vertical_spacing=0.15,
                            subplot_titles=('Mean', 'StDev', 'Acquisition'))

        if not surface == ['surface']:
            for i in range(4):
                fig.add_trace(fig_m[i], row=1, col=1)
                fig.add_trace(fig_v[i], row=1, col=2)
                fig.add_trace(fig_ac[i], row=2, col=1)

            ratio = (vmaxmin[0][0]-vmaxmin[0][1])/(vmaxmin[1][0]-vmaxmin[1][1])

            fig.update_xaxes(title=axis_name[0],
                             range=[vmaxmin[0][1],vmaxmin[0][0]],
                             scaleanchor='y',
                             scaleratio=1/ratio,
                             constrain='domain',
                             constraintoward= 'right',
                             )
            fig.update_yaxes(title=axis_name[1],
                             range=[vmaxmin[1][1],vmaxmin[1][0]],
                             zeroline=False,
                             constrain='domain',
                             )
            fig.layout.annotations[0].update(x=0.27)
            fig.layout.annotations[1].update(x=0.85)
            fig.layout.annotations[2].update(x=0.27)

        else:
            for i in range(4):
                fig.add_trace(fig_m[i], row=1, col=1)

            for i in range(2):
                fig.add_trace(fig_v[i], row=1, col=2)
                fig.add_trace(fig_ac[i], row=2, col=1)

            camera = dict(up=dict(x=0, y=0, z=1),
                          center=dict(x=0, y=0, z=0),
                          eye=dict(x=1.25, y=-1.25, z=1.25)
                          )

            fig.update_layout(scene1 = dict(xaxis = dict(title=axis_name[0], range=[vmaxmin[0][1],vmaxmin[0][0]]),
                                            yaxis = dict(title=axis_name[1], range=[vmaxmin[1][1],vmaxmin[1][0]]),
                                            zaxis = dict(title='Mean', range=[vis.m.min(), vis.m.max()]),
                                            camera=camera,
                                            aspectmode='cube',
                                            ),
                              scene2 = dict(xaxis = dict(title=axis_name[0], range=[vmaxmin[0][1],vmaxmin[0][0]]),
                                            yaxis = dict(title=axis_name[1], range=[vmaxmin[1][1],vmaxmin[1][0]]),
                                            zaxis = dict(title='StDev', range=[vis.v.min(), vis.v.max()]),
                                            camera=camera,
                                            aspectmode='cube',
                                            ),
                              scene3 = dict(xaxis = dict(title=axis_name[0], range=[vmaxmin[0][1],vmaxmin[0][0]]),
                                            yaxis = dict(title=axis_name[1], range=[vmaxmin[1][1],vmaxmin[1][0]]),
                                            zaxis = dict(title='Acquisition', range=[0, 1]),
                                            camera=camera,
                                            aspectmode='cube',
                                            ),
                              margin=dict(r=20, l=10, b=10, t=10),
                              )

        fig.update_layout(height=800,
                          width=1000,
                          legend=dict(x=0.65, y=0.25),
                          )




        show_figure = [dcc.Graph(id="graph",
                                 figure=fig,
                                 style={'height': '800px'},
                                 ),
                       ]




        return show_figure

@app.callback(Output("download_suggest-2D", 'data'),
              [Input("export_suggest-2D", 'n_clicks')],
              [State({'type': "suggest_table", 'index': 2}, 'data'),
               State({'type': "suggest_table", 'index': 2}, 'columns')],
              prevent_initial_call=True)
def download(n_clicks, data, columns):
    df = pd.DataFrame(data=data, columns=[columns[i]['name'] for i in range(len(columns))])
    return dcc.send_data_frame(df.to_csv, 'Suggest.csv', index=False)


@app.callback(Output("download_figdata-2D", 'data'),
              [Input("export_figdata-2D", 'n_clicks')],
              [State({'type': "suggest_table", 'index': 2}, 'data'),
               State({'type': "suggest_table", 'index': 2}, 'columns')],
              prevent_initial_call=True)
def download_figdata(n_clicks, data, columns):
    df = pd.DataFrame(data=data, columns=[columns[i]['name'] for i in range(len(columns))])

    X = vis.XX.repeat(len(vis.YY)).reshape(len(vis.XX),len(vis.YY)).T.flatten()
    Y = vis.YY.repeat(len(vis.XX))

    figdata = np.array([X, Y, vis.m.flatten(), vis.v.flatten(), vis.acqu.flatten()])
    columns2 = [df.columns[0], df.columns[1], 'Mean', 'StDev', 'Acquisition']
    fd = pd.DataFrame(figdata.T, columns=columns2)
    return dcc.send_data_frame(fd.to_csv, 'Figure_data.csv', index=False)
