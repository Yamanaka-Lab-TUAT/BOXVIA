# Copyright (c) 2021, BOXVIA Developers
# All rights reserved.
# Code released under the BSD 3-clause license.

import base64
import datetime
import io
import pandas as pd

import webbrowser
from threading import Timer

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
from flask import request

from base import app, port
from GUI import OneDim, TwoDim, MultiDim

app.title = 'BOXVIA'
app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.Img(src='assets/title.png'),
                     width=10,
                     ),
             dbc.Col(dbc.Button('Shutdown',
                                color='danger',
                                outline=True,
                                id="shutdown",
                                style={'margin-top': '120px'},
                                ),
                     width='auto',
                     ),
             ],
             justify='between',
             className="h-30",
            ),
    dbc.Alert([html.H5('Notification: To quit the application, be sure to press upper right Shutdown button.', className='alert-heading'),
               html.Hr(),
               html.P(['The application will continue to run in the background unless you press the Shutdown button. '
                      'If you close the web browser without doing the exit process, connect directly to ',
                      html.A('http://localhost:8050', href='http://localhost:8050', className='alert-link'),
                      '.'],
                      className='mb-0',
                      ),
               ],
               color='info',
               dismissable=True,
               is_open=True,
               ),
    html.Hr(),
    dbc.Row([dbc.Col(html.H5('Import data',
                             ),
                     width='auto',
                     ),
             dbc.Col([dbc.Badge('?',
                                color='light',
                                className='ml-1',
                                pill=True,
                                id="input_text_info",
                                ),
                      dbc.Tooltip('Import a .CSV file that contains inputs/output data.　See the sample .CSV files for how to fill the data into a CSV file.',
                                  target="input_text_info",
                                  placement='right',
                                  autohide=False,
                                  ),
                      ],
                     width='auto',
                     ),
              ],
             className='mt-2',
             no_gutters=True,
             ),
    html.Div([dcc.Upload(id="upload-data",
                         children=html.Div(['Drag and Drop or ',
                                            html.A('Select Files')
                                            ]
                                           ),
                         style={'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                                },
                         multiple=False
                         ),
              html.Div(id="output-data-upload"),
              ],
             ),
    html.P(),
    html.Div([dcc.Location(id="url", refresh=False),
              html.Div(id="page-content")
              ]),
    html.Hr(),
    html.P(),
    html.Div('Copyright (c) 2021, BOXVIA Developers'),
    html.Div('All rights reserved.'),
    html.Div(['Cheif Developer: A. Ishii (', html.A('Yamanaka lab. @TUAT', href='http://web.tuat.ac.jp/~yamanaka/'), ')']),
    html.P(),
    ])

def parse_contents(contents, filename, date):
    global df
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        print(e)
        return html.Div(['There was an error processing this file.'])

    return html.Div([html.H5(filename),
                     dbc.Row([dbc.Col(html.H6(datetime.datetime.fromtimestamp(date)),
                                      width='auto',
                                      ),
                              dbc.Col([dbc.Button('Export',
                                                  color='dark',
                                                  outline=True,
                                                  size='sm',
                                                  id="export_input",
                                                  ),
                                      dbc.Tooltip('Download this table data as "Input.csv."',
                                                  target="export_input",
                                                  placement='top',
                                                  ),
                                       ],
                                       style={'margin-bottom': '3px'},
                                       width={'size': 'auto', 'order': 'last'}
                                      ),
                              dcc.Download(id="download-data-csv"),
                              ],
                             justify='between',
                             ),
                     dash_table.DataTable(id="input_table",
                                          data=df.to_dict('records'),
                                          columns=[{'name': i, 'id': i} for i in df.columns],
                                          style_table={'height': '335px', 'overflowY': 'auto', 'margin-right': '5px'},
                                          style_header={'fontWeight': 'bold'},
                                          cell_selectable=False,
                                          editable=False,
                                          sort_action='native',
                                          css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
                                          style_cell={'width': '{}%'.format(len(df.columns)),
                                                      'textOverflow': 'ellipsis',
                                                      'overflow': 'hidden',
                                                      },
                                          style_data_conditional=[{'cursor': 'not-allowed'}],
                                          ),
                     dbc.Row(dbc.Col([dbc.Button('Delete bottom data',
                                                color='secondary',
                                                size='sm',
                                                id="delete",
                                                style={'margin-top': '3px'},
                                                ),
                                      dbc.Tooltip('Delete the bottom data in this table.',
                                                  target="delete",
                                                  placement='bottom',
                                                  ),
                                      ],
                                     width='auto',
                                     ),
                             justify='end',
                             ),
                     html.Hr(),
                     ],
                    ), len(df.columns)-1


@app.callback([Output("output-data-upload", 'children'),
               Output("url", 'pathname')],
              Input("upload-data", 'contents'),
              State("upload-data", 'filename'),
              State("upload-data", 'last_modified'))
def read_initdata(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children, dimension = parse_contents(list_of_contents, list_of_names, list_of_dates)

        if dimension == 1:
            path = '/OneDim'
        elif dimension == 2:
            path = '/TwoDim'
        elif dimension >= 3:
            path = '/MultiDim'
        else:
            path = ''

        return [children, path]

    else:
        return ['', '/']


@app.callback(Output("page-content", 'children'),
              [Input("url", 'pathname')])
def display_page(pathname):
    if pathname == '/OneDim':
        return OneDim.layout
    elif pathname == '/TwoDim':
        return TwoDim.layout
    elif pathname == '/MultiDim':
        return MultiDim.layout
    elif pathname == '/': #　defaultPageのURLの設定(以下の直接書き込みも可能)
        return html.Div('', style={'height': '500px'})
    else:
        return html.Div('Import failed', style={'color': 'Blue'})


@app.callback(Output("download-data-csv", 'data'),
              [Input("export_input", 'n_clicks')],
              [State("upload-data", 'filename'),
               State("input_table", 'data'),
               State("input_table", 'columns')],
              prevent_initial_call=True)
def download(n_clicks, filename, data, columns):

    data = pd.DataFrame(data=data, columns=[columns[i]['name'] for i in range(len(columns))])
    return dcc.send_data_frame(data.to_csv, filename, index=False)


@app.callback(Output("input_table", 'data'),
              [Input({'type': "add", 'index': ALL}, 'n_clicks'),
               Input("delete", 'n_clicks')],
              [State({'type': "suggest_table", 'index': ALL}, 'data')],
              prevent_initial_call=True)
def add_suggest(n_clicks_a, n_clicks_d, data):
    global df, a, d

    if n_clicks_a[0] == None:
        a = 0
        n_clicks_a[0] = 0
    if n_clicks_d == None:
        d = 0
        n_clicks_d = 0

    if a < n_clicks_a[0]:
        for sub in data[0]:
            for key in sub:
                sub[key] = float(sub[key])
        df = pd.DataFrame(df.to_dict('records')+data[0])
        a = n_clicks_a[0]

        return df.to_dict('records')

    if d < n_clicks_d:
        if len(df) <= 1:
            return df.to_dict('records')

        df = pd.DataFrame(df[:-1].to_dict('records'))
        d = n_clicks_d

        return df.to_dict('records')

    return df.to_dict('records')


def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.callback([Output("shutdown" ,'children'),
               Output("shutdown" ,'outline')],
              Input("shutdown", 'n_clicks'),
              prevent_initial_call=True)
def shut_button(click):
    shutdown()
    return ['STOP', False]


def open_browser():
	webbrowser.open_new("http://localhost:{}".format(port))


if __name__ == '__main__':
    Timer(1, open_browser).start();
    app.run_server(debug=False, port=port)
