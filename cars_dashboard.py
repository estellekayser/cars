import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import plotly.graph_objects as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv("data/carData.csv")

svar = ('Year', 'Owner','Selling_Price', 'Present_Price', 'Kms_Driven')
vquanti = ('Selling_Price', 'Present_Price', 'Kms_Driven')

## Stat


app.layout = html.Div([
    html.H1(''' Analyse bivariée '''),
    html.Div([
        html.Div([
            html.Div('''Abscisse'''),
            dcc.Dropdown(
                id='xaxis-column',
                options=[{'label': i, 'value': i} for i in svar],
                value='Year'
            ),
            
        ],
        style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            html.Div('''Ordonnée'''),
            dcc.Dropdown(
                id='yaxis-column',
                options=[{'label': i, 'value': i} for i in vquanti],
                value='Selling_Price'
            ),
           
        ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),
    dcc.Graph(id='linear'),
])

## Callback
@app.callback(
    Output('linear', 'figure'),
    [Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value')])
def update_graph(xaxis_column_name, yaxis_column_name):

    x = df[xaxis_column_name].values.reshape(-1, 1)
    y = df[yaxis_column_name].values
    model = LinearRegression().fit(x, y)

    x_range = np.linspace(x.min(), x.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    fig = px.scatter(x=df[xaxis_column_name],
                     y=df[yaxis_column_name], title='Relation entre %s et %s' %(xaxis_column_name, yaxis_column_name))
    fig.add_traces(go.Scatter(x=x_range, y=y_range, name=('Y = %f.X + %f' %(model.coef_, model.intercept_))))
    fig.update_xaxes(title=xaxis_column_name)
    fig.update_yaxes(title=yaxis_column_name)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)