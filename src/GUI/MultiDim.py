# Copyright (c) 2021, BOXVIA Developers
# All rights reserved.
# Code released under the BSD 3-clause license.

import os
import sys
import csv
import numpy as np
import itertools

import base64
import io
import pandas as pd

import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash.exceptions import PreventUpdate

from base import app
from visualize.visualize import visualize

if getattr(sys, 'frozen', False):
    program_directory = os.path.dirname(os.path.abspath(sys.executable))
    if os.getcwd() == '/':
        program_directory = os.path.dirname(os.path.dirname(os.path.dirname(program_directory)))
else:
    program_directory = os.path.dirname(os.path.abspath(__file__))[:-4]

n_max = 5
n_axis = n_max
unselect_max = n_max - 3

axis_range0 = [dbc.Row([dbc.Col(html.Div(str(i+1)+'-axis',
                                         id="dataname_"+str(i),
                                         style={'width': '100%', 'color': 'Red'},
                                         className='mt-1',
                                         ),
                                width=2,
                                ),
                        dbc.Col(dbc.Button('Add',
                                           id="add_c_"+str(i),
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
                        dbc.Col(html.Div(dcc.Input(id="datamin_"+str(i),
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
                        dbc.Col(html.Div(dcc.Input(id="datamax_"+str(i),
                                                   type='number',
                                                   value='',
                                                   style={'width': '150%'},
                                                   ),
                                         ),
                                width=1,
                                ),
                         dbc.Col(dcc.Dropdown(id="cont_or_disc_"+str(i),
                                              options=[{'label': 'Continuous', 'value': 'continuous'},
                                                       {'label': 'Discrete',   'value': 'discrete'},
                                                       ],
                                              value='continuous',
                                              clearable=False,
                                              ),
                                 width={'size': 2, 'offset': 1},
                                 ),
                         dbc.Col(html.Div(dcc.Input(id="interval_"+str(i),
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

slider_c = [html.Div(id="colorbar")]
slider_c.append(html.Center(id="set_range_slider"))
sliders = []
for i in range(n_max-3):
    sliders.append([dcc.Slider(id="dim_"+str(i+4),
                               className='mt-3',
                               updatemode='drag',
                               ),
                    html.Center(id="value_dim_"+str(i+4),
                                ),
                    ])


slider = [html.Div(id="dimdim_"+str(i+4), children=sliders[i], style={'display': 'none'}) for i in range(n_max-3)]

slider2d = [dcc.Slider(id="dim_2d",
                       className='mt-3',
                       updatemode='drag',
                       )]
slider2d.append(html.Center(id="value_dim_2d",
                            ),
                )


layout = dbc.Container([
    dbc.Row([dbc.Col([html.H6('Batch size',
                              className='mt-1',
                              id="batch_text",
                              style={'display': 'inline-block'},
                              ),
                      dbc.Badge('?',
                                color='light',
                                className='ml-1',
                                pill=True,
                                id="batch_text_info",
                                style={'display': 'inline-block', 'margin-bottom': '3px'},
                                ),
                      dbc.Tooltip('Number of candidate points to be suggested by BO',
                                  target="batch_text_info",
                                  placement='top',
                                  ),
                      ],
                     width=2,
                     ),
             dbc.Col(dcc.Input(id="batch_size",
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
             dbc.Col(dbc.RadioItems(id="actype",
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
                              id="jwname",
                              className='mt-1',
                              style={'display': 'inline-block'},
                              ),
                      dbc.Badge('?',
                                color='light',
                                className='ml-1',
                                pill=True,
                                id="jwname_info",
                                style={'display': 'inline-block', 'margin-bottom': '3px'},
                                ),
                      dbc.Tooltip('Balance between exploration and exploitation (Large number encourages exploration).',
                                  target="jwname_info",
                                  placement='top',
                                  ),
                      ],
                     width=2,
                     ),
             dbc.Col(dcc.Input(id="jwparam",
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
                                id="kernel_info",
                                style={'display': 'inline-block', 'margin-bottom': '3px'},
                                ),
                      dbc.Tooltip('Select the type of kernel function to be used for Gaussian process regression. Matern5/2 or RBF are commonly used.',
                                  target="kernel_info",
                                  placement='top',
                                  ),
                      ],
                      width=2,
                     ),
             dbc.Col(dcc.Dropdown(id="kern-dropdown",
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
    dbc.Row([dbc.Col([dbc.Checklist(id="maximize",
                                    options=[{'label': 'Maximize', 'value': 'val_max'}],
                                    value=[],
                                    style={'display': 'inline-block'},
                                    ),
                      dbc.Badge('?',
                                color='light',
                                className='ml-1',
                                pill=True,
                                id="maximize_info",
                                style={'display': 'inline-block'},
                                ),
                      dbc.Tooltip('Check the box if you treat maximization problem.',
                                  target="maximize_info",
                                  placement='top',
                                  ),
                      ],
                     width=2,
                     ),
             dbc.Col([dbc.Checklist(id="exact_fval",
                                    options=[{'label': 'Noiseless', 'value': 'val_efval'}],
                                    value=[],
                                    style={'display': 'inline-block'},
                                    ),
                      dbc.Badge('?',
                                color='light',
                                className='ml-1',
                                pill=True,
                                id="exact_fval_info",
                                style={'display': 'inline-block'},
                                ),
                      dbc.Tooltip('Check the box if noiseless evaluation is available.',
                                  target="exact_fval_info",
                                  placement='top',
                                  ),
                      ],
                     width=2,
                     ),
              dbc.Col([dbc.Checklist(id="de_duplication",
                                     options=[{'label': 'Avoid re-evaluating', 'value': 'dup'}],
                                     value=[],
                                     style={'display': 'inline-block'},
                                     ),
                       dbc.Badge('?',
                                 color='light',
                                 className='ml-1',
                                 pill=True,
                                 id="de_duplication_info",
                                 style={'display': 'inline-block'},
                                 ),
                       dbc.Tooltip('Check the box if the location where the data already exists is not to be re-evaluated.',
                                   target="de_duplication_info",
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
                                id="constraint_info",
                                style={'display': 'inline-block', 'margin-bottom': '3px'},
                                ),
                      dbc.Tooltip(['Constraints can be added as inequalities such that the left-hand side is less than or equal to zero. ',
                                   'Multiple constraints can be added by writing them on new lines.'],
                                  target="constraint_info",
                                  placement='bottom',
                                  ),
                      ],
                     width=2,
                     ),
             dbc.Col(dcc.Textarea(id="constraint",
                                  value='',
                                  style={'width': '105%', 'height': '100px'}),
                     width=4,
                     ),
             dbc.Col(html.Div(' <= 0',
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
                                id="range_info",
                                style={'display': 'inline-block', 'margin-bottom': '3px'},
                                ),
                      dbc.Tooltip('Set the range for input parameter. The set range can be saved & loaded.',
                                  target="range_info",
                                  placement='top',
                                  ),
                      ],
                     style={'text-align': 'right'},
                     width=2,
                     ),
             dbc.Col(dbc.Button('Load',
                                id="button_load",
                                n_clicks=0,
                                color='success',
                                ),
                     width=1,
                     ),
             dbc.Col(dbc.Button('Save',
                                id="button_save",
                                n_clicks=0,
                                color='info',
                                ),
                     width=1,
                     ),
             dbc.Col(html.H6('',
                             className='mt-2',
                             id="save_label",
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
                                id="interval_info",
                                style={'display': 'inline-block', 'margin-bottom': '3px'},
                                ),
                      dbc.Tooltip('Set the discretizing interval for parameter to be suggested.',
                                  target="interval_info",
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
                           id="button_runBO",
                           n_clicks=0,
                           style={'width': '100%'},
                           className='mt-3',
                           color='primary',
                           size='lg',
                           ),
                ),
    html.Center([dcc.Loading(id="run_loading",
                             type='default',
                             children=html.Center('',
                                                  id="done_run",
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
                                id="suggest_text_info",
                                style={'display': 'inline-block', 'margin-bottom': '4px'},
                                ),
                      dbc.Tooltip('The table is editable.',
                                  target="suggest_text_info",
                                  placement='top',
                                  ),
                      ],
                      width='auto',
                      ),
             dbc.Col([dbc.Button('Export',
                                 id="export_suggest",
                                 color='dark',
                                 outline=True,
                                 size='sm',
                                 ),
                      dbc.Tooltip('Download the table data as a Suggest.csv.',
                                  target="export_suggest",
                                  placement='top',
                                  ),
                      dcc.Download(id="download_suggest"),
                      ],
                      style={'margin-top': '4px'},
                      width='auto',
                      ),
              ],
             justify='between',
             ),
    dash_table.DataTable(id={'type': "suggest_table", 'index': 3},
                         style_table={'height': '200px', 'overflowY': 'auto'},
                         style_header={'fontWeight': 'bold'},
                         editable=True,
                         css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
                         ),
    dbc.Row(dbc.Col([dbc.Button(html.Div('Add to the table of imported data',
                                         id="add_info",
                                         ),
                                id={'type': "add", 'index': 3},
                                color='primary',
                                outline=True,
                                ),
                      dbc.Tooltip('Add the suggest table data to the bottom of the input data table.',
                                  target="add_info",
                                  placement='bottom',
                                  ),
                       ],
                     width='auto',
                     ),
            justify='end'
            ),
    html.Hr(),
    html.Div([html.H6('Display axis',
                      style={'display': 'inline-block'},
                      ),
              dbc.Badge('?',
                        color='light',
                        className='ml-1',
                        pill=True,
                        id="display_axis_info",
                        style={'display': 'inline-block', 'margin-bottom': '4px'},
                        ),
              dbc.Tooltip('Select 3 parameters to be the axes that construct 3D graph space.',
                          target="display_axis_info",
                          placement='top',
                          ),
              ],
             ),
    dbc.Row(dbc.Col(dbc.Checklist(id="axis_set",
                                  options=[{'label': '', 'value': 'set_'+str(i)} for i in range(n_max)],
                                  value=['set_0', 'set_1', 'set_2'],
                                  inline=True,
                                  labelStyle={'color': 'Red'},
                                  ),
                    width=10,
                    ),
            className='mt-2',
            ),
    dbc.Row([dbc.Col(html.H6('Resolution',
                             className='mt-1',
                             ),
                     width=2,
                     ),
             dbc.Col(dcc.Input(id="resolution",
                               type='number',
                               value=20,
                               min=1, max=100,
                               style={'width': '100%'}
                               ),
                     width=2,
                     ),
             dbc.Col(html.H6('Marker size',
                             className='mt-1',
                             ),
                     width={"size": 2, "offset": 1},
                     ),
             dbc.Col(dcc.Input(id="marker_size",
                               type='number',
                               value=5,
                               min=1, max=100,
                               style={'width': '100%'},
                               ),
                     width=2,
                     ),
             ],
             className='mt-3'
            ),
    html.P(),
    dbc.Row([dbc.Col(html.H6('Display data type'),
                     width=2,
                     ),
             dbc.Col(dbc.RadioItems(id="display_type",
                                    options=[{'label': 'None', 'value': 'none'},
                                             {'label': 'Mean', 'value': 'mean'},
                                             {'label': 'StDev', 'value': 'sd'},
                                             {'label': 'Acquisition', 'value': 'acqu'},
                                             ],
                                    value='none',
                                    inline=True,
                                    ),
                     width=7,
                     ),
             ],
             className='mt-2'
            ),
    dbc.Row([dbc.Col(html.H6('Display 2D plane'),
                     width=2,
                     ),
             dbc.Col(dbc.RadioItems(id="display_plane",
                                    options=[{'label': '2nd vs 3rd', 'value': 'plane23'},
                                             {'label': '1st vs 3rd', 'value': 'plane13'},
                                             {'label': '1st vs 2nd', 'value': 'plane12'},
                                             ],
                                    value='plane23',
                                    inline=True,
                                    ),
                     width=7,
                     ),
              ],
             className='mt-2'
             ),
    dbc.Row([dbc.Col(dbc.Button('Make / Reload graph',
                                id="button_reload",
                                n_clicks=0,
                                style={'width': '100%'},
                                className='mt-3',
                                color='primary',
                                ),
                     width=3,
                     ),
             dbc.Col(dbc.Checklist(id="show_plane",
                                   options=[{'label': 'Show plane', 'value': 'show'}],
                                   value=['show'],
                                   className='mt-4',
                                   switch=True,
                                   ),
                     width=2,
                     ),
              ],
             ),
    html.Center(dcc.Loading(id="make3D_loading",
                            type='circle',
                            children=html.Div(id="graphspace_3D",
                                              children=html.Div(''),
                                              style={'height': '800px'},
                                              ),
                            ),
                ),
    html.Div(id="slider2D", children=slider2d, style={'display': 'none'}),
    dcc.Store(id="value_2ds"),
    html.Div(id="slider_c", children=slider_c, style={'display': 'none'}),
    html.Div(children=slider),
    dcc.Store(id="values_s"),
    html.P(),
    ],
)

@app.callback([Output("jwname", 'children'),
               Output("jwparam", 'value')],
              [Input("actype", 'value')]
              )
def jwlabel(actype):
    if actype == 'EI':
        return ['Jitter', 0.01]
    elif actype == 'LCB':
        return ['Weight', 2]


@app.callback([Output("dataname_"+str(i), 'children') for i in range(n_axis)],
              [Output("axis_set", 'options'),
               Output({'type': "suggest_table", 'index': 3}, 'columns'),
               Output({'type': "suggest_table", 'index': 3}, 'style_cell'),
               Output({'type': "suggest_table", 'index': 3}, 'style_data_conditional')],
              [Output("add_c_"+str(i), 'disabled') for i in range(n_axis)],
              [Input("input_table", 'columns')],
              )
def read_initdata(columns):
    global vis, n_axis, axis_name

    axis_name = [columns[i]['name'] for i in range(len(columns)-1)]
    n_axis = len(axis_name)

    for i in range(n_axis, n_max):
        axis_name.append('')

    options = [[{'label': axis_name[i], 'value': 'set_'+str(i)} for i in range(n_axis)]]

    style_cell=[{'width': '{}%'.format(len(columns)),
                'textOverflow': 'ellipsis',
                'overflow': 'hidden',
                }]
    style_data_conditional=[[{'if': {'column_id': columns[-1]['name']},
                              'backgroundColor': 'rgba(153,204,255,0.2)',
                              },
                             ]]

    button_disable =[False for i in range(n_axis)]
    for i in range(n_axis, n_max):
        button_disable.append(True)

    vis = visualize(axis_name, n_axis)

    return axis_name+options+[columns]+style_cell+style_data_conditional+button_disable


@app.callback([Output("display_plane", 'options')],
              [Input("axis_set", 'value'),
               Input("axis_set", 'options')],
              )
def checklist_label(axis_set, op):
    global select_axis, unselect_axis

    if len(axis_set)==3:
        select_axis = [int(axis_set[i][-1]) for i in range(3)]
        select_axis = sorted(select_axis)
        unselect_axis = []
        for i in range(n_axis):
            if not i in select_axis:
                unselect_axis.append(i)

        options=[{'label': axis_name[select_axis[1]]+' vs '+axis_name[select_axis[2]], 'value': 'plane23'},
                 {'label': axis_name[select_axis[0]]+' vs '+axis_name[select_axis[2]], 'value': 'plane13'},
                 {'label': axis_name[select_axis[0]]+' vs '+axis_name[select_axis[1]], 'value': 'plane12'},
                 ]
    else:
        options=[{'label': '2nd vs 3rd', 'value': 'plane23'},
                 {'label': '1st vs 3rd', 'value': 'plane13'},
                 {'label': '1st vs 2nd', 'value': 'plane12'},
                 ]

    return [options]


@app.callback(Output("constraint", 'value'),
              [Input("add_c_"+str(i), 'n_clicks') for i in range(n_axis)],
              State("constraint", 'value'),
              State("input_table", 'columns'),
              prevent_initial_call=True)
def add_c(*args):

    constraint = args[-2]
    columns = args[-1]
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    for i in range(n_axis):
        if "add_c_"+str(i) in changed_id:
            return constraint+columns[i]['name']


@app.callback([Output("interval_"+str(i), 'disabled') for i in range(n_axis)],
              [Input("cont_or_disc_"+str(i), 'value') for i in range(n_axis)],
              )
def interval_disable(*args):
    cont_or_disc = list(args)

    return [cont_or_disc[i] == 'continuous' for i in range(n_max)]

@app.callback([Output("datamax_"+str(i), 'value') for i in range(n_axis)],
              [Output("datamin_"+str(i), 'value') for i in range(n_axis)],
              [Input("button_load", 'n_clicks')],
              [State("upload-data", 'filename')],
              prevent_initial_call=True)
def load_range(n_click, filename):

    try:
        filename = filename[:-4]
        is_file = os.path.isfile(program_directory+'/'+filename+'_range.csv')
        if is_file:

            cols = [i+1 for i in range(n_axis)]
            maxmin = np.loadtxt(program_directory+'/'+filename+'_range.csv', delimiter=',', skiprows=1, usecols=cols)

            maxmin = maxmin.T

            range_max = [maxmin[i][0] for i in range(n_axis)]
            range_min = [maxmin[i][1] for i in range(n_axis)]
            for i in range(n_axis, n_max):
                range_max.append('')
                range_min.append('')

            return range_max+range_min

        else:
            range_max = ['' for i in range(n_max)]
            range_min = range_max
            return range_max+range_min

    except:
         range_max = ['' for i in range(n_max)]
         range_min = range_max
         return range_max+range_min


@app.callback(Output("save_label", 'children'),
              [Input("button_save", 'n_clicks')],
              [State("upload-data", 'filename')],
              [State("datamax_"+str(i), 'value') for i in range(n_axis)],
              [State("datamin_"+str(i), 'value') for i in range(n_axis)],
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


@app.callback([Output("done_run", 'children'),
               Output({'type': "suggest_table", 'index': 3}, 'data')],
              [Input("button_runBO", 'n_clicks')],
              [State("input_table", 'data'),
               State("input_table", 'columns'),
               State("batch_size", 'value'),
               State("actype", 'value'),
               State("jwparam", 'value'),
               State("kern-dropdown", 'value'),
               State("exact_fval", 'value'),
               State("de_duplication", 'value'),
               State("maximize", 'value'),
               State("constraint", 'value')],
              [State("datamax_"+str(i), 'value') for i in range(n_axis)],
              [State("datamin_"+str(i), 'value') for i in range(n_axis)],
              [State("cont_or_disc_"+str(i), 'value') for i in range(n_axis)],
              [State("interval_"+str(i), 'value') for i in range(n_axis)],
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


@app.callback([Output("graphspace_3D", 'children'),
               Output("slider_c", 'style'),
               Output("colorbar", 'children')],
              list(itertools.chain.from_iterable([
              (Output("dimdim_"+str(i+4), 'style'),
               Output("dim_"+str(i+4), 'max'),
               Output("dim_"+str(i+4), 'min'),
               Output("dim_"+str(i+4), 'step')) for i in range(unselect_max)])),
               [Output("slider2D", 'style'),
                Output("dim_2d", 'max'),
                Output("dim_2d", 'min'),
                Output("dim_2d", 'step')],
              [Output("set_range_slider", 'children')],
              [Output("value_dim_"+str(i+4), 'children') for i in range(unselect_max)],
              [Output("value_dim_2d", 'children')],
              [Input("button_reload", 'n_clicks')],
              [State("input_table", 'data'),
               State("input_table", 'columns'),
               State("resolution", 'value'),
               State("display_type", 'value'),
               State("display_plane", 'value'),
               State("axis_set", 'value'),
               State("marker_size", 'value'),
               State("values_s", 'modified_timestamp'),
               State("values_s", 'data'),
               State("value_2ds", 'modified_timestamp'),
               State("value_2ds", 'data')],
              prevent_initial_call=True)
def make3D(n_click, data, columns, resolution, display_type, display_plane, axis_set, marker_size, ts, datas, ts2ds, data2ds):
    global axis_set_s, display_type_s

    axis_set_s = axis_set

    if len(axis_set) == 3:

        df = pd.DataFrame(data=data, columns=[columns[i]['name'] for i in range(len(columns))])
        vis.initial_vis(df.values)

        display_type_s = display_type

        vis.setAxis(select_axis, unselect_axis)
        vis.setAxis2D(display_plane)
        vis.plot_setting(resolution, display_type, marker_size)

        vis.type_minmax()

        children = [html.Div(id="graph_3D")]

        if display_type == 'none':
            range_max = 0
            range_min = 0
            vstep = 0
            disable=True
        else:
            range_max = float('{:.3g}'.format(vis.val_max))
            range_min = float('{:.3g}'.format(vis.val_min))
            vstep = (vis.val_max-vis.val_min)/100
            disable=False

        colorbar = [dcc.RangeSlider(id="colorbar_range",
                                    className='mt-3',
                                    updatemode='drag',
                                    max=range_max,
                                    min=range_min,
                                    step=vstep,
                                    value=[range_min, range_max],
                                    ),
                                   ]

        colorbar_text = [[html.Div('Colorbar range: Min: ',
                                   style={'display': 'inline-block'},
                                   ),
                          dcc.Input(id="crange_min",
                                    type='number',
                                    # max=range_max,
                                    # min=range_min,
                                    value=range_min,
                                    disabled=disable,
                                    debounce=True,
                                    style={'width': '10%', 'display': 'inline-block', 'margin-left': '10px'}),
                          html.Div('Max: ',
                                   style={'display': 'inline-block', 'margin-left': '10px'},
                                   ),
                          dcc.Input(id="crange_max",
                                    type='number',
                                    # max=range_max,
                                    # min=range_min,
                                    value=range_max,
                                    disabled=disable,
                                    debounce=True,
                                    style={'width': '10%', 'display': 'inline-block', 'margin-left': '10px'})]]

        set_slider = []
        for i in range(len(unselect_axis)):
            set_slider.append({})
            set_slider.append(float('{:.3g}'.format(vmaxmin[unselect_axis[i]][0])))
            set_slider.append(float('{:.3g}'.format(vmaxmin[unselect_axis[i]][1])))
            set_slider.append((vmaxmin[unselect_axis[i]][0]-vmaxmin[unselect_axis[i]][1])/100.)

        for i in range(unselect_max-len(unselect_axis)):
            set_slider.append({'display': 'none'})
            set_slider.append(0)
            set_slider.append(0)
            set_slider.append(0)

        values_s = [None for i in range(unselect_max)]
        for i in range(len(unselect_axis)):
            if ts2ds == None:
                values_s[i] = float('{:3g}'.format(vmaxmin[unselect_axis[i]][1]))
            else:
                values_s[i] = datas.get('values_s'+str(i+4))


        slider_text = []
        for i in range(len(unselect_axis)):
            slider_text.append([html.Div(axis_name[unselect_axis[i]]+' = ',
                                         id="dim_text_"+str(i+4),
                                         style={'display': 'inline-block'},
                                         ),
                                dcc.Input(id="dim_value_"+str(i+4),
                                          type='number',
                                          max=float('{:3g}'.format(vmaxmin[unselect_axis[i]][0])),
                                          min=float('{:3g}'.format(vmaxmin[unselect_axis[i]][1])),
                                          value=values_s[i],
                                          style={'width': '10%', 'display': 'inline-block', 'margin-left': '10px'},
                                          debounce=True,
                                          ),
                                ],
                                )

        for i in range(unselect_max-len(unselect_axis)):
            slider_text.append(html.Div(id="dim_value_"+str(i+len(unselect_axis)+4)))


        slider2d = [{}]
        slider2d.append(float('{:.3g}'.format(vmaxmin[vis.unselect_axis2d[0]][0])))
        slider2d.append(float('{:.3g}'.format(vmaxmin[vis.unselect_axis2d[0]][1])))
        slider2d.append((vmaxmin[vis.unselect_axis2d[0]][0]-vmaxmin[vis.unselect_axis2d[0]][1])/100.)

        if ts2ds == None:
            value_2ds = float('{:.3g}'.format(vmaxmin[vis.unselect_axis2d[0]][1]))
        else:
            value_2ds = data2ds.get('value_2ds')

        text2d = [[html.Div(axis_name[vis.unselect_axis2d[0]]+' = ',
                            id="dim_text2d",
                            style={'display': 'inline-block'},
                            ),
                   dcc.Input(id="dim_value2d",
                             type='number',
                             max=float('{:.3g}'.format(vmaxmin[vis.unselect_axis2d[0]][0])),
                             min=float('{:.3g}'.format(vmaxmin[vis.unselect_axis2d[0]][1])),
                             value=value_2ds,
                             style={'width': '10%', 'display': 'inline-block', 'margin-left': '10px'},
                             debounce=True,
                             ),
                   ],
                  ]


        return children+[{}]+colorbar+set_slider+slider2d+colorbar_text+slider_text+text2d

    else:
        children = ['Select 3 parameters !']

        colorbar = ['']
        set_slider = []
        for i in range(unselect_max):
            set_slider.append({'display': 'none'})
            set_slider.append(0)
            set_slider.append(0)
            set_slider.append(0)

        slider2d = [{'display': 'none'}, 0, 0, 0]

        colorbar_text = ['']

        slider_text = []
        for i in range(unselect_max):
            slider_text.append('')
            slider_text.append(0)
            slider_text.append(0)

        text2d = ['', 0, 0]

        return children+[{}]+colorbar+set_slider+slider2d+colorbar_text+slider_text+text2d




@app.callback(Output("graph_3D", 'children'),
              [Input("colorbar_range", 'value'),
               Input("dim_2d", 'value'),
               Input("show_plane", 'value')],
              [Input("dim_"+str(i+4), 'value') for i in range(unselect_max)],
              prevent_initial_call=True)
def make3D_slider(colorbar_range, slicevalue, show_plane, *args):

    if not len(axis_set_s) == 3:
        return ['Number of selected axes is not 3.']+['' for i in range(2+unselect_max)]

    dim_ex = list(args)

    if slicevalue > vmaxmin[vis.unselect_axis2d[0]][0]:
        slider_value = vmaxmin[vis.unselect_axis2d[0]][0]
    elif slicevalue < vmaxmin[vis.unselect_axis2d[0]][1]:
        slider_value = vmaxmin[vis.unselect_axis2d[0]][1]
    else:
        slider_value = slicevalue

    fig_point = [vis.setnullData(),
                 vis.setInputData(dim_ex, best=True),
                 vis.setInputData(dim_ex, best=False),
                 vis.setSuggestData(dim_ex)]
    fig_point2d = [vis.setnullData2D(),
                   vis.setInputData2D(slider_value, dim_ex, best=True),
                   vis.setInputData2D(slider_value, dim_ex, best=False),
                   vis.setSuggestData2D(slider_value, dim_ex)]

    vis.make3Dgraph(dim_ex)
    vis.make2Dgraph(slider_value, dim_ex)

    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type': 'scene'}, {}]],
                        horizontal_spacing=0.05)

    camera = dict(up=dict(x=0, y=0, z=1),
                  center=dict(x=0, y=0, z=0),
                  eye=dict(x=2, y=-2, z=2)
                  )

    for i in range(4):
        fig.add_trace(fig_point[i], row=1, col=1)
        fig.add_trace(fig_point2d[i], row=1, col=2)

    if display_type_s == 'mean':
        fig.add_trace(vis.setMean(colorbar_range[0], colorbar_range[1]), row=1, col=1)
        fig.add_trace(vis.setMean2D(colorbar_range[0], colorbar_range[1]), row=1, col=2)
    elif display_type_s == 'sd':
        fig.add_trace(vis.setStDev(colorbar_range[0], colorbar_range[1]), row=1, col=1)
        fig.add_trace(vis.setStDev2D(colorbar_range[0], colorbar_range[1]), row=1, col=2)
    elif display_type_s == 'acqu':
        fig.add_trace(vis.setAcqu(colorbar_range[0], colorbar_range[1]), row=1, col=1)
        fig.add_trace(vis.setAcqu2D(colorbar_range[0], colorbar_range[1]), row=1, col=2)

    if show_plane == ['show']:
        fig.add_trace(vis.setPlane(slider_value))

    fig.update_layout(scene = dict(xaxis = dict(title=axis_name[select_axis[0]], range = [vmaxmin[select_axis[0]][1],vmaxmin[select_axis[0]][0]],
                                                dtick=int((vmaxmin[select_axis[0]][0]-vmaxmin[select_axis[0]][1])/10)),
                                   yaxis = dict(title=axis_name[select_axis[1]], range = [vmaxmin[select_axis[1]][1],vmaxmin[select_axis[1]][0]],
                                                dtick=int((vmaxmin[select_axis[1]][0]-vmaxmin[select_axis[1]][1])/10)),
                                   zaxis = dict(title=axis_name[select_axis[2]], range = [vmaxmin[select_axis[2]][1],vmaxmin[select_axis[2]][0]],
                                                dtick=int((vmaxmin[select_axis[2]][0]-vmaxmin[select_axis[2]][1])/10))),
                      scene_aspectmode='cube',
                      scene_camera=camera,
                      margin=dict(r=20, l=10, b=10, t=10)
                      )

    ratio = (vis.X2d.max()-vis.X2d.min())/(vis.Y2d.max()-vis.Y2d.min())
    text_xaxis = axis_name[vis.select_axis2d[0]]
    text_yaxis = axis_name[vis.select_axis2d[1]]

    fig.update_xaxes(title=text_xaxis,
                     range=[vis.X2d.min(),vis.X2d.max()],
                     scaleanchor='y',
                     scaleratio=1/ratio,
                     constrain='domain',
                     constraintoward= 'right',
                     row=1, col=2)
    fig.update_yaxes(title=text_yaxis,
                     range=[vis.Y2d.min(),vis.Y2d.max()],
                     scaleanchor='x',
                     scaleratio=ratio,
                     zeroline=False,
                     constrain='domain',
                     row=1, col=2)

    fig.update_layout(legend=dict(x=0.),
                      scene_aspectmode='cube',
                      height=800,
                      )

    show_figure = [dcc.Graph(id="graph",
                             figure=fig,
                             style={'height': '800px'},
                             ),
                   ]

    return show_figure

@app.callback([Output("crange_min", 'value'),
               Output("crange_max", 'value')],
              [Input("colorbar_range", 'value')],
              prevent_initial_call=True)
def c_slider_value(colorbar_range):

    if colorbar_range[0] == None:
        raise PreventUpdate
    if colorbar_range[1] == None:
        raise PreventUpdate

    return [float('{:.3g}'.format(colorbar_range[0])), float('{:.3g}'.format(colorbar_range[1]))]


@app.callback(Output("colorbar_range", 'value'),
              [Input("crange_min", 'value'),
               Input("crange_max", 'value')],
              prevent_initial_call=True)
def c_slider_value_in(crange_min, crange_max):

    if crange_min == None:
        raise PreventUpdate
    if crange_max == None:
        raise PreventUpdate

    return [crange_min, crange_max]


@app.callback([Output("dim_value_"+str(i+4), 'value') for i in range(unselect_max)],
              [Input("dim_"+str(i+4), 'value') for i in range(unselect_max)],
              prevent_initial_call=True)
def sliders_value(*args):

    dim_ex = list(args)

    values = []
    for i in range(len(unselect_axis)):
        if dim_ex[i] == None:
            raise PreventUpdate
        values.append(float('{:.3g}'.format(dim_ex[i])))

    if len(unselect_axis) < unselect_max:
        for i in range(unselect_max-len(unselect_axis)):
            values.append(0)

    return values

@app.callback([Output("dim_"+str(i+4), 'value') for i in range(unselect_max)],
              [Input("dim_value_"+str(i+4), 'value') for i in range(unselect_max)],
              prevent_initial_call=True)
def sliders_value_in(*args):

    values = list(args)

    for i in range(len(unselect_axis)):
        if values[i] == None:
            raise PreventUpdate

    return values


@app.callback([Output("dim_value2d", 'value')],
              [Input("dim_2d", 'value')],
              prevent_initial_call=True)
def slider2d_value(slicevalue):

    if slicevalue == None:
        raise PreventUpdate

    value2d = [float('{:.3g}'.format(slicevalue))]

    return value2d


@app.callback(Output("dim_2d", 'value'),
              [Input("dim_value2d", 'value')],
              prevent_initial_call=True)
def slider2d_value_in(value2d):

    if value2d == None:
        raise PreventUpdate

    return value2d


@app.callback(Output("values_s", 'data'),
              [Input("dim_value_"+str(i+4), 'value') for i in range(unselect_max)],
              prevent_initial_call=True)
def value_2ds(*args):

    values = list(args)

    values_dict = {}
    for i in range(unselect_max):
        values_dict['values_s'+str(i+4)] = values[i]
    data = values_dict

    return data


@app.callback(Output("value_2ds", 'data'),
              Input("dim_value2d", 'value'),
              prevent_initial_call=True)
def value_2ds(value2d):

    data = {'value_2ds': value2d}

    return data


@app.callback(Output("download_suggest", 'data'),
              [Input("export_suggest", 'n_clicks')],
              [State({'type': "suggest_table", 'index': 3}, 'data'),
               State({'type': "suggest_table", 'index': 3}, 'columns')],
              prevent_initial_call=True)
def download(n_clicks, data):
    df = pd.DataFrame(data=data, columns=[columns[i]['name'] for i in range(len(columns))])
    return dcc.send_data_frame(df.to_csv, 'Suggest.csv', index=False)
