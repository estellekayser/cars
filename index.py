#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import plotly.graph_objects as go
from sklearn import svm
from sklearn.model_selection import train_test_split
from app import app
from app import server

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
                html.Div('''Les boxplots permettent de voir la répartition des données. 
                    On voit d'un seul coup d'oeil où se situent le min, Q1, la médiane, Q3 et le max, ainsi que les valeurs extrêmes. 
                '''),
            ]),
            ], style={'columnCount': 2,} 
        ),
    ], 
    ),

    html.Div([
        html.H1(''' Regression linéaire et SVM'''),
        html.Div([
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
                ],
                style={'width': '48%', 'display': 'inline-block'}
                ),
            ]),
        ]),

        html.Div([

            html.Div([
                html.H2(''' Regression linéaire univariée'''),
                html.Div(''' N'ayant observé aucune différence entre les différentes méthodes de régression linéaire univariée, la regression ci-dessous a été réalisé avec sklearn. 
            '''),
                dcc.Graph(id='linear'),
                
            ]),

            html.Div([
                html.H2(''' SVM '''),
                dcc.Graph(id='svm'),
            ]),

        ],  
        style={'columnCount': 2,} ),
        html.Div('''Différence entre regression linéaire univariée et SVR univariée : La SVR cherche a maximiser les marges.
        Alors que la régression linéaire cherche à minimiser les erreurs. 
        
            '''),
        
    ]),

    # html.Div([
    #     html.Div([
    #         html.H1('''Regression linéaire multiple'''),
    #         html.Div([
    #         ], id = 'regmulti' ),
    #     ]),
    # ]),

    html.Div('''Question bonus - Quelles données manque-il à votre analyse ? -
    On pourrait regarder la date du contrôle technique et regarder s'il contient une ou plusieurs reparations à faire. 
    On peut aussi se demander à quelle série appartient la voiture ? 
    Est-ce que la voiture a t-elle été déjà accidentée ?
    Est-ce que la courroie de distribution a t-elle été changée ? 
    ...
    Les données manquantes sont de ce fait, des données techniques sur la voiture. 
            '''),

])

## Callback
@app.callback(
    Output('linear', 'figure'),
    [Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value')])
def graph_reg(xvar, yvar):

    x = df[xvar].values.reshape(-1, 1)
    y = df[yvar].values

    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)
    
    model = LinearRegression().fit(x, y)

    x_range = np.linspace(x.min(), x.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))


    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train, name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test, name='test', mode='markers'),
        go.Scatter(x=x_range, y=y_range, name=('Y = %f.X + %f' %(model.coef_, model.intercept_)))
    ])
    fig.update_xaxes(title=xvar)
    fig.update_yaxes(title=yvar)

    return fig


@app.callback(
    Output('univariate_quali', 'figure'),
    [Input('vquali_choose', 'value')])
def graph_quali(var):
    fig = px.histogram(df, x=var)
    return fig 

@app.callback(
    Output('univariate_quanti', 'figure'),
    [Input('vquanti_choose', 'value')])
def graph_quanti(var):
    fig = px.box(df, x=var)
    return fig 


@app.callback(
    Output('svm', 'figure'),
    [Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value')])
def graph_svm(xvar, yvar):

    x = df[xvar].values.reshape(-1, 1)
    y = df[yvar].values

    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)
    
    model =  svm.SVR(kernel="linear").fit(x, y)

    x_range = np.linspace(x.min(), x.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))


    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train, name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test, name='test', mode='markers'),
        go.Scatter(x=x_range, y=y_range, name=('Y = %f.X + %f' %(model.coef_, model.intercept_)))
    ])
    fig.update_xaxes(title=xvar)
    fig.update_yaxes(title=yvar)

    return fig

# @app.callback(Output('regmulti', 'children'))
# def graph_multi():
#     data = pd.DataFrame(df, columns=['Year','Kms_Driven', 'Selling_Price']),

#     col_transform = pd.DataFrame(df, columns=['Transmission'])
#     dummies = pd.get_dummies(col_transform)

#     xm = np.array(data.join(dummies))

#     ym = df['Selling_Price'].values

#     reg = LinearRegression().fit(xm, ym)
#     print('Sklearn multiple - Coefficients:', reg.coef_)
#     b1 = (reg.intercept_)


#     return  b1

####
if __name__ == '__main__':
    app.run_server(debug=True)
