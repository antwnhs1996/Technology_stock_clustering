
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
import pandas as pd
import plotly.graph_objects as go

stock_vis = pd.read_csv("../DashApp/data/stock_vis")
stock_vis = stock_vis.drop(columns =['Unnamed: 0'])
kpi_vis = pd.read_csv('../DashApp/data/kpi_vis')
nasdaq_vis = pd.read_csv('../DashApp/data/nasdaq_vis')
sp_vis = pd.read_csv("../DashApp/data/sp500_vis")


def line_fig(stock, stock_vis = stock_vis, sp_vis=sp_vis, nasdaq_vis = nasdaq_vis):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_vis['date'].values, y=stock_vis[stock].values,
                        mode='lines',
                        name=stock))
    fig.add_trace(go.Scatter(x=stock_vis['date'].values, y=sp_vis[sp_vis['date']>='2015-12-02' ]['price'].values,
                        mode='lines',
                        name='S&P 500'))
    fig.add_trace(go.Scatter(x=stock_vis['date'].values, y=nasdaq_vis[nasdaq_vis['date']>='2015-12-02' ]['price'].values,
                        mode='lines', name='Nasdaq 100'))

    fig.update_layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1'],
                      template='plotly_dark',
                      paper_bgcolor='rgba(0, 0, 0, 0)',
                      plot_bgcolor='rgba(0, 0, 0, 0)',
                      margin={'t': 50},
                      height=500, 
                      hovermode='x',
                      autosize=True,
                      )
    return fig
    
def kpi_fig(stock, algo , kpi_vis = kpi_vis, sp_vis=sp_vis, nasdaq_vis = nasdaq_vis):
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        title = {"text": "Cluster"},
        mode = "number",
        value = kpi_vis[kpi_vis['ticker'] == stock][algo].values[0],
        domain = {'x': [0.33, 0.66], 'y': [0, 1]}
        ))

    fig.add_trace(go.Indicator(
        mode = "number",
        value = nasdaq_vis.iloc[1279]['price']*100,
        title = {"text": "Nasdaq 100 return"},
        domain = {'x': [0, 0.33], 'y': [0, 0.5]},
        number = {'suffix': "%"}))

    fig.add_trace(go.Indicator(
        mode = "number",
        value = sp_vis.iloc[1279]['price']*100,
        title = {"text": "S&P 500 return"},
        domain = {'x': [0.66, 1], 'y': [0, 0.5]},
        number = {'suffix': "%"}))

    fig.add_trace(go.Indicator(
        mode = "number",
        value = kpi_vis[kpi_vis['ticker'] == stock]['return'].values[0]*100,
        title = {"text": "{} total return".format(stock)},
        domain = {'x': [0, 0.33], 'y': [0.5, 1]},
        number = {'suffix': "%"}))

    fig.add_trace(go.Indicator(
        mode = "number",
        value = kpi_vis[kpi_vis['ticker'] == stock]['volatility'].values[0]*100,
        title = {"text": " {} Volatility".format(stock)},
        domain = {'x': [0.66, 1], 'y': [0.5, 1]},
        number = {'suffix': "%"},))


    fig.update_layout(
                      uniformtext_minsize=30,
                      uniformtext_mode='hide',
                      font = dict(color = 'green', family = "Arial"),
                      template='plotly_dark',
                      paper_bgcolor='rgba(0, 0, 0, 0)',
                      plot_bgcolor='rgba(0, 0, 0, 0)',
                      margin={'t': 50},
                      height=400, 
                      hovermode='x'
                     )

    return fig




app = dash.Dash(external_stylesheets = [dbc.themes.BOOTSTRAP])
server = app.server

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background": "rgb(56,56,56)"
    }



CONTENT_STYLE = {
    "margin-left": "16rem",
    "margin-right": "10rem",
    "padding": "3rem 3rem",
    "background": "black"
}

sidebar = html.Div(
    [
        html.H2("Menu", className="display-4"),
        html.Hr(),
        html.P(
            "please select the stock that you want to preview", className="lead"
        ),
        html.H3("Stock selection", className="display-6"),
        dcc.Dropdown(
            id = "stock-dropdown",
            options =[{'label':i, 'value':i} for i in stock_vis.drop(columns = ['date']).keys().tolist()],
            value = stock_vis.keys().tolist()[1]
            ),
        html.H3("Algorithm selection", className="display-6"),
        dcc.Dropdown(
            id = "algo-dropdown",
            options =[{'label':'Hierarchical clustering', 'value':'HC clusters'},{'label':'Kmeans', 'value':'KM clusters'}],
            value = 'HC clusters',
            ),
    ],
    style=SIDEBAR_STYLE,
)

app.layout = html.Div([dcc.Location(id="url"), sidebar, dbc.Container(
    [
        dbc.Row(dbc.Col(html.H2('STOCK OVERVIEW', className='text-center text-primary, mb-3'))),  # header row
        
        dbc.Row([  # start of second row

            dbc.Col([  # first column on second row
            
            dcc.Graph(id='line-fig',
                      style={'height':400}),
            
            html.Hr(),

            ], width={'size': 10, 'offset': 0, 'order': 1}),  # width first column on second row

        ]),  # end of second row
        
        dbc.Row([  # start of third row
            dbc.Col([  # first column on third row
                
                dcc.Graph(id ="kpi-fig",
                            style = {'height':400})
    
            ], width={'size': 10, 'offset': 0, 'order': 1}),  # width first column on second row
        ])  # end of third row
        
    ], fluid=True, style=CONTENT_STYLE) 

])
@app.callback(Output("line-fig", 'figure'),
             [Input('stock-dropdown', 'value')])
def update_line(stock):
    return line_fig(stock)

@app.callback(Output("kpi-fig", 'figure'),
             [Input('stock-dropdown', 'value')],
             [Input('algo-dropdown', 'value')])
def update_kpi(stock, algo):
    return kpi_fig(stock, algo)



if __name__ == "__main__":
    app.run_server(debug=True)
