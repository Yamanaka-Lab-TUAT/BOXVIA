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
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from base import app
from visualize.visualize1D import visualize

if getattr(sys, 'frozen', False):
    program_directory = os.path.dirname(os.path.abspath(sys.executable))
    if os.getcwd() == '/':
        program_directory = os.path.dirname(os.path.dirname(os.path.dirname(program_directory)))
else:
    program_directory = os.path.dirname(os.path.abspath(__file__))[:-4]

n_max = 1
n_axis = n_max

axis_range0 = [dbc.Row([dbc.Col(html.Div('axis_name',
                                         id="dataname-1D",
                                         style={'width': '100%', 'color': 'Red'},
                                         className='mt-1',
                                         ),
                                width=2,
                                ),
                        dbc.Col(dbc.Button('Add',
                                           id="add_c-1D",
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
                        dbc.Col(html.Div(dcc.Input(id="datamin-1D",
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
                        dbc.Col(html.Div(dcc.Input(id="datamax-1D",
                                                   type='number',
                                                   value='',
                                                   style={'width': '150%'},
                                                   ),
                                         ),
                                width=1
                                ),
                        dbc.Col(dcc.Dropdown(id="cont_or_disc-1D",
                                             options=[{'label': 'Continuous', 'value': 'continuous'},
                                                      {'label': 'Discrete',   'value': 'discrete'},
                                                      ],
                                             value='continuous',
                                             clearable=False,
                                             ),
                                width={'size': 2, 'offset': 1},
                                ),
                        dbc.Col(html.Div(dcc.Input(id="interval-1D",
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
                  ),
                ]

layout = dbc.Container([
    dbc.Row([dbc.Col([html.H6('Batch size',
                              className='mt-1',
                              style={'display': 'inline-block'},
                              ),
                      dbc.Badge('?',
                                color='light',
                                className='ml-1',
                                pill=True,
                                id="batch_text-1D_info",
                                style={'display': 'inline-block', 'margin-bottom': '3px'},
                                ),
                      dbc.Tooltip('Number of candidate points to be suggested by BO.',
                                  target="batch_text-1D_info",
                                  placement='top',
                                  ),
                      ],
                     width=2,
                     ),
             dbc.Col(dcc.Input(id="batch_size-1D",
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
             dbc.Col(dbc.RadioItems(id="actype-1D",
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
                             id="jwname-1D",
                             style={'display': 'inline-block'},
                             ),
                      dbc.Badge('?',
                                color='light',
                                className='ml-1',
                                pill=True,
                                id="jwname-1D_info",
                                style={'display': 'inline-block', 'margin-bottom': '3px'},
                                ),
                      dbc.Tooltip('Balance between exploration and exploitation (Large number encourages exploration).',
                                  target="jwname-1D_info",
                                  placement='top',
                                  ),
                      ],
                     width=2,
                     ),
             dbc.Col(dcc.Input(id="jwparam-1D",
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
                                id="kernel-1D_info",
                                style={'display': 'inline-block', 'margin-bottom': '3px'},
                                ),
                      dbc.Tooltip('Select the type of kernel function to be used for Gaussian process regression. Matern5/2 or RBF are commonly used.',
                                  target="kernel-1D_info",
                                  placement='top',
                                  ),
                      ],
                      width=2,
                     ),
             dbc.Col(dcc.Dropdown(id="kern-dropdown-1D",
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
    dbc.Row([dbc.Col([dbc.Checklist(id="maximize-1D",
                                    options=[{'label': 'Maximize', 'value': 'val_max'}],
                                    value=[],
                                    style={'display': 'inline-block'},
                                    ),
                      dbc.Badge('?',
                                color='light',
                                className='ml-1',
                                pill=True,
                                id="maximize-1D_info",
                                style={'display': 'inline-block'},
                                ),
                      dbc.Tooltip('Check the box if you treat maximization problem.',
                                  target="maximize-1D_info",
                                  placement='top',
                                  ),
                      ],
                     width=2,
                     ),
             dbc.Col([dbc.Checklist(id="exact_fval-1D",
                                    options=[{'label': 'Noiseless', 'value': 'val_efval'}],
                                    value=[],
                                    style={'display': 'inline-block'},
                                    ),
                      dbc.Badge('?',
                                color='light',
                                className='ml-1',
                                pill=True,
                                id="exact_fval-1D_info",
                                style={'display': 'inline-block'},
                                ),
                      dbc.Tooltip('Check the box if noiseless evaluation is available.',
                                  target="exact_fval-1D_info",
                                  placement='top',
                                  ),
                      ],
                     width=2,
                     ),
              dbc.Col([dbc.Checklist(id="de_duplication-1D",
                                     options=[{'label': 'Avoid re-evaluating', 'value': 'dup'}],
                                     value=[],
                                     style={'display': 'inline-block'},
                                     ),
                       dbc.Badge('?',
                                 color='light',
                                 className='ml-1',
                                 pill=True,
                                 id="de_duplication-1D_info",
                                 style={'display': 'inline-block'},
                                 ),
                       dbc.Tooltip('Check the box if the location where the data already exists is not to be re-evaluated.',
                                   target="de_duplication-1D_info",
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
                                id="constraint-1D_info",
                                style={'display': 'inline-block', 'margin-bottom': '3px'},
                                ),
                      dbc.Tooltip(['Constraints can be added as inequalities such that the left-hand side is less than or equal to zero. ',
                                   'Multiple constraints can be added by writing them on new lines.'],
                                  target="constraint-1D_info",
                                  placement='bottom',
                                  ),
                      ],
                     width=2,
                     ),
             dbc.Col(dcc.Textarea(id="constraint-1D",
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
                                id="range-1D_info",
                                style={'display': 'inline-block', 'margin-bottom': '3px'},
                                ),
                      dbc.Tooltip('Set the range for input parameter. The set range can be saved & loaded.',
                                  target="range-1D_info",
                                  placement='top',
                                  ),
                      ],
                     style={'text-align': 'right'},
                     width=2,
                     ),
             dbc.Col(dbc.Button('Load',
                                id="button_load-1D",
                                n_clicks=0,
                                color='success',
                                ),
                     width=1,
                     ),
             dbc.Col(dbc.Button('Save',
                                id="button_save-1D",
                                n_clicks=0,
                                color='info',
                                ),
                     width=1,
                     ),
             dbc.Col(html.H6('',
                             className='mt-2',
                             id="save_label-1D",
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
                                id="interval-1D_info",
                                style={'display': 'inline-block', 'margin-bottom': '3px'},
                                ),
                      dbc.Tooltip('Set the discretizing interval for parameter to be suggested.',
                                  target="interval-1D_info",
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
                           id="button_runBO-1D",
                           n_clicks=0,
                           style={'width': '100%'},
                           className='mt-3',
                           color='primary',
                           size='lg',
                           ),
                ),
    html.Center([dcc.Loading(id="run_loading-1D",
                             type='default',
                             children=html.Center('',
                                                  id="done_run-1D",
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
                                id="suggest_text-1D_info",
                                style={'display': 'inline-block', 'margin-bottom': '4px'},
                                ),
                      dbc.Tooltip('The table is editable.',
                                  target="suggest_text-1D_info",
                                  placement='top',
                                  ),
                      ],
                      width='auto',
                      ),
             dbc.Col([dbc.Button('Export',
                                 id="export_suggest-1D",
                                 color='dark',
                                 outline=True,
                                 size='sm',
                                 ),
                      dbc.Tooltip('Download the table data as a Suggest.csv.',
                                  target="export_suggest-1D",
                                  placement='top',
                                  ),
                      dcc.Download(id="download_suggest-1D"),
                      ],
                      style={'margin-top': '4px'},
                      width='auto',
                      ),
             ],
            justify='between',
            ),
    dash_table.DataTable(id={'type': "suggest_table", 'index': 1},
                         style_table={'height': '200px', 'overflowY': 'auto'},
                         style_header={'fontWeight': 'bold'},
                         editable=True,
                         css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
                         ),
    dbc.Row(dbc.Col([dbc.Button(html.Div('Add suggests to imported data table',
                                         id="add-1D_info",
                                         ),
                                id={'type': "add", 'index': 1},
                                color='primary',
                                outline=True,
                                ),
                     dbc.Tooltip('Add the suggest table data to the bottom of the input data table.',
                                 target="add-1D_info",
                                 placement='bottom',
                                 ),
                     ],
                    width='auto',
                    ),
            justify='end',
            ),
    html.Hr(),
    dbc.Row([dbc.Col(html.H6('Marker size',
                             className='mt-1',
                             ),
                     width=2,
                     ),
             dbc.Col(dcc.Input(id="marker_size-1D",
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
    html.Center(dcc.Loading(id="make1D_loading",
                            type='circle',
                            children=html.Div(id="graphspace_1D",
                                              children=html.Div(''),
                                              style={'height': '800px'},
                                              ),
                            ),
                ),
    dbc.Row(dbc.Col([dbc.Button('Export figure data',
                               color='dark',
                               outline=True,
                               size='sm',
                               id="export_figdata-1D",
                               ),
                     dbc.Tooltip('Export the figure data as Figure_data.csv.',
                                 target="export_figdata-1D",
                                 placement='top',
                                 ),
                     dcc.Download(id="download_figdata-1D"),
                     ],
                    width='auto',
                    ),
            justify='end',
            ),
    html.P(),
    ],
)

@app.callback([Output("jwname-1D", 'children'),
               Output("jwparam-1D", 'value')],
              [Input("actype-1D", 'value')],
              )
def jwlabel(actype):
    if actype == 'EI':
        return ['Jitter', 0.01]
    elif actype == 'LCB':
        return ['Weight', 2]


@app.callback([Output("dataname-1D", 'children'),
               Output({'type': "suggest_table", 'index': 1}, 'columns'),
               Output({'type': "suggest_table", 'index': 1}, 'style_cell'),
               Output({'type': "suggest_table", 'index': 1}, 'style_data_conditional')],
              [Input("input_table", 'columns')],
              )
def read_initdata(columns):
    global vis

    style_cell=[{'width': '{}%'.format(len(columns)),
                'textOverflow': 'ellipsis',
                'overflow': 'hidden',
                }]
    style_data_conditional=[[{'if': {'column_id': columns[-1]['name']},
                              'backgroundColor': 'rgba(153,204,255,0.2)',
                              },
                             ]]

    vis = visualize([columns[0]['name']], n_axis)

    return [columns[0]['name']]+[columns]+style_cell+style_data_conditional


@app.callback(Output("constraint-1D", 'value'),
              Input("add_c-1D", 'n_clicks'),
              State("constraint-1D", 'value'),
              State("input_table", 'columns'),
              prevent_initial_call=True)
def add_c(n_clicks, constraint, columns):

    return constraint+columns[0]['name']

@app.callback(Output("interval-1D", 'disabled'),
              Input("cont_or_disc-1D", 'value'),
              )
def interval_disable(cont_or_disc):
    return cont_or_disc == 'continuous'


@app.callback([Output("datamax-1D", 'value'),
               Output("datamin-1D", 'value')],
              [Input("button_load-1D", 'n_clicks')],
              [State("upload-data", 'filename')],
              prevent_initial_call=True)
def load_range(n_click, filename):

    try:
        filename = filename[:-4]
        is_file = os.path.isfile(program_directory+'/'+filename+'_range.csv')
        if is_file:

            maxmin = np.loadtxt(program_directory+'/'+filename+'_range.csv', delimiter=',', skiprows=1, usecols=1)

            range_max = [maxmin[0]]
            range_min = [maxmin[1]]

            return range_max+range_min

        else:
            range_max = ['']
            range_min = range_max
            return range_max+range_min

    except:
         range_max = ['']
         range_min = range_max
         return range_max+range_min


@app.callback(Output("save_label-1D", 'children'),
              [Input("button_save-1D", 'n_clicks')],
              [State("upload-data", 'filename'),
               State("input_table", 'columns'),
               State("datamax-1D", 'value'),
               State("datamin-1D", 'value')],
              prevent_initial_call=True)
def save_range(n_click, filename, columns, value_max, value_min):

    try:
        filename = filename[:-4]

        maxmin = [[value_max, value_min]]

        header = [header[:] for header in columns[0]['name']]
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
        return program_directory


@app.callback([Output("done_run-1D", 'children'),
               Output({'type': "suggest_table", 'index': 1}, 'data')],
              [Input("button_runBO-1D", 'n_clicks')],
              [State("input_table", 'data'),
               State("input_table", 'columns'),
               State("batch_size-1D", 'value'),
               State("actype-1D", 'value'),
               State("jwparam-1D", 'value'),
               State("kern-dropdown-1D", 'value'),
               State("exact_fval-1D", 'value'),
               State("de_duplication-1D", 'value'),
               State("maximize-1D", 'value'),
               State("constraint-1D", 'value'),
               State("datamax-1D", 'value'),
               State("datamin-1D", 'value'),
               State("cont_or_disc-1D", 'value'),
               State("interval-1D", 'value')],
              prevent_initial_call=True)
def runBO(n_click, data, columns, batch_size, actype, jwparam, kernel, exact_fval, de_duplication, maximize, constraint, value_max, value_min, cont_or_disc, interval):
    global vmaxmin

    try:
        df = pd.DataFrame(data=data, columns=[columns[i]['name'] for i in range(len(columns))])
        vmaxmin = [[value_max, value_min]]

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

@app.callback(Output("graphspace_1D", 'children'),
              [Input("done_run-1D", 'children'),
               Input("marker_size-1D", 'value')],
              [State("input_table", 'data'),
               State("input_table", 'columns'),
               State("batch_size-1D", 'value')],
              )
def make1D(txt, marker_size, data, columns, batch_size):
    if txt:
        df = pd.DataFrame(data=data, columns=[columns[i]['name'] for i in range(len(columns))])
        graph = []
        graph_ac = []

        vis.initial_vis(df.values)
        vis.makegraph()

        graph.append(vis.setInputData(marker_size, True))
        graph.append(vis.setInputData(marker_size, False))

        graph.append(vis.setMean())
        graph.append(vis.setStDevUp())
        graph.append(vis.setStDevDown())

        graph_ac.append(vis.setAcqu())

        fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25], shared_xaxes=True)
        for i in range(5):
            fig.add_trace(graph[i], row=1, col=1)
        fig.add_trace(graph_ac[0], row=2, col=1)

        fig.update_xaxes(range=[vmaxmin[0][1],vmaxmin[0][0]], row=1, col=1)
        fig.update_yaxes(title='Mean and StDev', row=1, col=1)
        fig.update_xaxes(title=df.columns[0], range=[vmaxmin[0][1],vmaxmin[0][0]], row=2, col=1)
        fig.update_yaxes(title='Acquisition function', range=[0,1], row=2, col=1)
        fig.update_layout(height=800,
                          width=1000,
                          shapes = [dict(x0=vis.setSuggestData(i), x1=vis.setSuggestData(i), y0=0, y1=1,
                                         xref='x',
                                         yref='paper',
                                         line_width=2,
                                         line_color='#ff0000') for i in range(batch_size)],
                          )

        show_figure = [dcc.Graph(id="graph",
                                 figure=fig,
                                 style={'height': '800px'},
                                 ),
                       ]


        return show_figure

@app.callback(Output("download_suggest-1D", 'data'),
              [Input("export_suggest-1D", 'n_clicks')],
              [State({'type': "suggest_table", 'index': 1}, 'data'),
               State({'type': "suggest_table", 'index': 1}, 'columns')],
              prevent_initial_call=True)
def download(n_clicks, data, columns):
    df = pd.DataFrame(data=data, columns=[columns[i]['name'] for i in range(len(columns))])
    return dcc.send_data_frame(df.to_csv, 'Suggest.csv', index=False)


@app.callback(Output("download_figdata-1D", 'data'),
              [Input("export_figdata-1D", 'n_clicks')],
              [State({'type': "suggest_table", 'index': 1}, 'data'),
               State({'type': "suggest_table", 'index': 1}, 'columns')],
              prevent_initial_call=True)
def download_figdata(n_clicks, data, columns):
    df = pd.DataFrame(data=data, columns=[columns[i]['name'] for i in range(len(columns))])

    figdata = np.array([vis.XX, vis.m, vis.v, vis.acqu])
    columns2 = [df.columns[0], 'Mean', 'StDev', 'Acquisition']
    fd = pd.DataFrame(figdata.T, columns=columns2)
    return dcc.send_data_frame(fd.to_csv, 'Figure_data.csv', index=False)
