# Copyright (c) 2021, BOXVIA Developers
# All rights reserved.
# Code released under the BSD 3-clause license.

import dash
import dash_bootstrap_components as dbc

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.LITERA], suppress_callback_exceptions=True)
server = app.server
port = 8050
