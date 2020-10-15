import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

## Correction et ajout variables
df = pd.read_csv("data/carData.csv")

svar = ('Year', 'Owner','Selling_Price', 'Present_Price', 'Kms_Driven')
vquanti = ('Selling_Price', 'Present_Price', 'Kms_Driven')
vquali = ('Car_Name','Year','Fuel_Type', 'Seller_Type', 'Transmission')

app.layout = html.Div([

    html.Div([
        html.Div([
            html.H1(''' Analyse univariée ''')
            ]),

        html.Div([
            html.Div([
                html.H2(''' Variables qualitatives '''),
                html.Div([
                    html.Div('''Choisir une variable'''),
                    dcc.Dropdown(
                        id='vquali_choose',
                        options=[{'label': i, 'value': i} for i in vquali],
                        value='Year'
                    ),
                ],
                style={'width': '48%', 'display': 'inline-block'}),
                dcc.Graph(id='univariate_quali'),
            ]),

            html.Div([
                html.H2(''' Variables quantitatives '''),
                html.Div([
                    html.Div('''Choisir une variable'''),
                    dcc.Dropdown(
                        id='vquanti_choose',
                        options=[{'label': i, 'value': i} for i in vquanti],
                        value='Selling_Price'
                    ),
                ],
                style={'width': '48%', 'display': 'inline-block'}),
                dcc.Graph(id='univariate_quanti'),
            ]),
            ], style={'columnCount': 2,} 
        ),
    ], 
    ),

    html.Div([
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
    ]),

    # html.Div([
    #     html.H1(''' Analyse multivariee '''),
    #     html.Div([
            
    #     ]),
    #     dcc.Graph(id='multivariee'),
    # ])
])

## Callback
@app.callback(
    Output('linear', 'figure'),
    [Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value')])
def update_graph2(xvar, yvar):

    x = df[xvar].values.reshape(-1, 1)
    y = df[yvar].values
    model = LinearRegression().fit(x, y)

    x_range = np.linspace(x.min(), x.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    fig = px.scatter(x=df[xvar],
                     y=df[yvar], title='Relation entre %s et %s' %(xvar, yvar))
    fig.add_traces(go.Scatter(x=x_range, y=y_range, name=('Y = %f.X + %f' %(model.coef_, model.intercept_))))
    fig.update_xaxes(title=xvar)
    fig.update_yaxes(title=yvar)

    return fig


@app.callback(
    Output('univariate_quali', 'figure'),
    [Input('vquali_choose', 'value')])
def update_graph1(var):
    fig = px.histogram(df, x=var)
    return fig 

@app.callback(
    Output('univariate_quanti', 'figure'),
    [Input('vquanti_choose', 'value')])
def update_graph3(var):
    fig = px.box(df, x=var)
    return fig 

if __name__ == '__main__':
    app.run_server(debug=True)
