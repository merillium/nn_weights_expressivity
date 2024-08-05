import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc

def ReLU(x):
    return x * (x > 0)

def f_0(X, W1, W2, w3):
    """Compact form for a two-layer neural network"""
    return ReLU(ReLU(X@W1)@W2)@w3

## meshgrid and reverse meshgrid
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
xxRev, yyRev = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))

# Stack them column-wise to form an n x 2 matrix
X = np.column_stack((xx.flatten(), yy.flatten()))
x0, x1 = X[:,0], X[:,1]

app = Dash(__name__, external_scripts=["https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML"],)

slider_container_style = {'width': '12%', 'margin': '5px'}

def random_weight():
    return np.random.normal(loc=0.0, scale=np.sqrt(2), size=1)[0]

app.layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                    dbc.Row(
                        [
                            html.Div([
                                
                                html.Div([
                                    ## adjust W1 elements
                                    html.H5('W1'),
                                    html.Div(style=slider_container_style),
                                    html.Div(style=slider_container_style),
                                    html.Div(style=slider_container_style),
                                    html.Div(style=slider_container_style),
                                    ## adjust W2 elements
                                    html.H5('W2'),
                                    html.Div(style=slider_container_style),
                                    html.Div(style=slider_container_style),
                                    html.Div(style=slider_container_style),
                                    html.Div(style=slider_container_style),
                                ], style={'display': 'flex', 'flex-direction': 'col', 'justify-content': 'left'}),
                            ], ),
                        ],
                        justify="center",
                    ),
                    dbc.Row(
                        [
                            html.Div([
                                
                                html.Div([
                                    ## adjust W1 elements
                                    
                                    html.Div(dcc.Slider(-5, 5, step=0.01, value=random_weight(), marks={-5: '-5', 0: '0', 5: '5'}, id='W1[1][1]'), style=slider_container_style),
                                    html.Div(dcc.Slider(-5, 5,step=0.01, value=random_weight(), marks={-5: '-5', 0: '0', 5: '5'}, id='W1[1][2]'), style=slider_container_style),
                                    html.Div(dcc.Slider(-5, 5, step=0.01, value=random_weight(), marks={-5: '-5', 0: '0', 5: '5'}, id='W1[1][3]'), style=slider_container_style),
                                    html.Div(dcc.Slider(-5, 5, step=0.01, value=random_weight(), marks={-5: '-5', 0: '0', 5: '5'}, id='W1[1][4]'), style=slider_container_style),
                                    ## adjust W2 elements
                                    
                                    html.Div(dcc.Slider(-5, 5, step=0.01, value=random_weight(), marks={-5: '-5', 0: '0', 5: '5'}, id='W2[1][1]'), style=slider_container_style),
                                    html.Div(dcc.Slider(-5, 5, step=0.01, value=random_weight(), marks={-5: '-5', 0: '0', 5: '5'}, id='W2[1][2]'), style=slider_container_style),
                                    html.Div(dcc.Slider(-5, 5, step=0.01, value=random_weight(), marks={-5: '-5', 0: '0', 5: '5'}, id='W2[1][3]'), style=slider_container_style),
                                    html.Div(dcc.Slider(-5, 5, step=0.01, value=random_weight(), marks={-5: '-5', 0: '0', 5: '5'}, id='W2[1][4]'), style=slider_container_style)
                                ], style={'display': 'flex', 'flex-direction': 'col', 'justify-content': 'left'}),
                            ], ),
                        ],
                        justify="center",
                    ),
                    dbc.Row(
                        [
                            html.Div([
                                html.Div([
                                     ## adjust W1 elements
                                    html.Div(dcc.Slider(-5, 5, step=0.01, value=random_weight(), marks={-5: '-5', 0: '0', 5: '5'}, id='W1[2][1]'), style=slider_container_style),
                                    html.Div(dcc.Slider(-5, 5, step=0.01, value=random_weight(), marks={-5: '-5', 0: '0', 5: '5'}, id='W1[2][2]'), style=slider_container_style),
                                    html.Div(dcc.Slider(-5, 5, step=0.01, value=random_weight(), marks={-5: '-5', 0: '0', 5: '5'}, id='W1[2][3]'), style=slider_container_style),
                                    html.Div(dcc.Slider(-5, 5, step=0.01, value=random_weight(), marks={-5: '-5', 0: '0', 5: '5'}, id='W1[2][4]'), style=slider_container_style),
                                     ## adjust W2 elements
                                    html.Div(dcc.Slider(-5, 5, step=0.01, value=random_weight(), marks={-5: '-5', 0: '0', 5: '5'}, id='W2[2][1]'), style=slider_container_style),
                                    html.Div(dcc.Slider(-5, 5, step=0.01, value=random_weight(), marks={-5: '-5', 0: '0', 5: '5'}, id='W2[2][2]'), style=slider_container_style),
                                    html.Div(dcc.Slider(-5, 5, step=0.01, value=random_weight(), marks={-5: '-5', 0: '0', 5: '5'}, id='W2[2][3]'), style=slider_container_style),
                                    html.Div(dcc.Slider(-5, 5, step=0.01, value=random_weight(), marks={-5: '-5', 0: '0', 5: '5'}, id='W2[2][4]'), style=slider_container_style)
                                ], style={'display': 'flex', 'flex-direction': 'col', 'justify-content': 'left'}),
                            ], ),
                                
                        ],
                        justify="center",
                    ),
                    ]
                ),
                dbc.Col(
                    [
                    dbc.Row(
                        [
                            html.Div([
                                ## adjust W2 elements
                                html.Div([
                                    html.Div(style=slider_container_style),
                                    html.Div(style=slider_container_style),
                                    html.Div(style=slider_container_style),
                                    html.Div(style=slider_container_style),
                                    html.Div(dcc.Slider(-5, 5, step=0.01, value=random_weight(), marks={-5: '-5', 0: '0', 5: '5'}, id='W2[3][1]'), style=slider_container_style),
                                    html.Div(dcc.Slider(-5, 5, step=0.01, value=random_weight(), marks={-5: '-5', 0: '0', 5: '5'}, id='W2[3][2]'), style=slider_container_style),
                                    html.Div(dcc.Slider(-5, 5, step=0.01, value=random_weight(), marks={-5: '-5', 0: '0', 5: '5'}, id='W2[3][3]'), style=slider_container_style),
                                    html.Div(dcc.Slider(-5, 5, step=0.01, value=random_weight(), marks={-5: '-5', 0: '0', 5: '5'}, id='W2[3][4]'), style=slider_container_style)
                                ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'left'}),
                            ], ),
                        ],
                        justify="center",
                    ),
                    dbc.Row(
                        [
                            html.Div([
                                ## adjust W2 elements
                                html.Div([
                                    html.Div(style=slider_container_style),
                                    html.Div(style=slider_container_style),
                                    html.Div(style=slider_container_style),
                                    html.Div(style=slider_container_style),
                                    html.Div(dcc.Slider(-5, 5, step=0.01, value=random_weight(), marks={-5: '-5', 0: '0', 5: '5'}, id='W2[4][1]'), style=slider_container_style),
                                    html.Div(dcc.Slider(-5, 5, step=0.01, value=random_weight(), marks={-5: '-5', 0: '0', 5: '5'}, id='W2[4][2]'), style=slider_container_style),
                                    html.Div(dcc.Slider(-5, 5, step=0.01, value=random_weight(), marks={-5: '-5', 0: '0', 5: '5'}, id='W2[4][3]'), style=slider_container_style),
                                    html.Div(dcc.Slider(-5, 5, step=0.01, value=random_weight(), marks={-5: '-5', 0: '0', 5: '5'}, id='W2[4][4]'), style=slider_container_style)
                                ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'left'}),
                            ], ),
                                
                        ],
                        justify="center",
                    ),
                    ]
                ),
            ],
            justify="center",
        ),
        dcc.Store(id='meshgrid-store'), # this doesn't quite work (yet)
        dcc.Store(id='camera-store'),
        dcc.Graph(id='graph', mathjax=True),
    ],
)


@callback(
    Output('graph', 'figure'),
    Output('meshgrid-store', 'data'),
    Output('camera-store', 'data'),
    ## W1 elements
    Input('W1[1][1]', 'value'),
    Input('W1[1][2]', 'value'),
    Input('W1[1][3]', 'value'),
    Input('W1[1][4]', 'value'),
    Input('W1[2][1]', 'value'),
    Input('W1[2][2]', 'value'),
    Input('W1[2][3]', 'value'),
    Input('W1[2][4]', 'value'),
    ## W2 elements
    Input('W2[1][1]', 'value'),
    Input('W2[1][2]', 'value'),
    Input('W2[1][3]', 'value'),
    Input('W2[1][4]', 'value'),
    Input('W2[2][1]', 'value'),
    Input('W2[2][2]', 'value'),
    Input('W2[2][3]', 'value'),
    Input('W2[2][4]', 'value'),
    Input('W2[3][1]', 'value'),
    Input('W2[3][2]', 'value'),
    Input('W2[3][3]', 'value'),
    Input('W2[3][4]', 'value'),
    Input('W2[4][1]', 'value'),
    Input('W2[4][2]', 'value'),
    Input('W2[4][3]', 'value'),
    Input('W2[4][4]', 'value'),
    Input('graph', 'relayoutData'),
    State('meshgrid-store', 'data'),
    State('camera-store', 'data'),
)
def update_output(
    value1, value2, value3, value4, ## W1 elements
    value5, value6, value7, value8, 
    value9, value10, value11, value12, ## W2 elements
    value13, value14, value15, value16,
    value17, value18, value19, value20, 
    value21, value22, value23, value24,
    relayout_data, meshgrid_store, camera_store
):
    W1 = np.array([
        [value1, value2, value3, value4],
        [value5, value6, value7, value8],
    ])

    W2 = np.array([
        [value9, value10, value11, value12],
        [value13, value14, value15, value16],
        [value17, value18, value19, value20],
        [value21, value22, value23, value24],
    ])

    ## this is arbitrary
    w3 = np.array([[1],[1],[1],[1]])

    Z = f_0(X, W1, W2, w3).reshape(50,50)

    fig = make_subplots(rows=1, cols=2, column_widths=[0.2, 0.8], specs=[
        [{"type": "table"}, {"type": "scatter3d"}], 
    ])
    
    fig.add_trace(row=1, col=1, trace=go.Table(
        header=dict(values=['$$\\text{W}_1$$', ''], fill_color='#FFFFFF'), 
        cells=dict(values=[
            [value1, value5, '', '$$\\text{W}_2$$', value9, value13, value17, value21], 
            [value2, value6, '', '', value10, value14, value18, value22],
            [value3, value7, '', '', value11, value15, value19, value23],
            [value4, value8, '', '', value12, value16, value20, value24],
        ],
        format=[".2",".2",".2",".2"],
        fill_color=[['#F0F8FF','#F0F8FF','#FFFFFF','#FFFFFF','#F0F8FF','#F0F8FF','#F0F8FF','#F0F8FF']*4],
        font=dict(color=[
            ['black','black','white','black','black','black','black','black'],
            ['black','black','white','white','black','black','black','black'],
            ['black','black','white','white','black','black','black','black'],
            ['black','black','white','white','black','black','black','black']
        ])
        ),
        
    ))
    
    fig.add_trace(row=1, col=2, trace=go.Surface(z=Z, x=xx, y=yy),)

    grid_line_marker = dict(color='black', width=2)

    ## add gridlines in one direction using mesh grid
    for i, j, k in zip(xx, yy, Z):
        fig.add_trace(
            trace=go.Scatter3d(x=i, y=j, z=k, mode='lines', line=grid_line_marker, showlegend=False),
            row=1, col=2,
        ) 

    ## add gridlines in perpendicular direction using reverse mesh grid
    for i, j, k in zip(xxRev, yyRev, Z.T):
        fig.add_trace(
            trace=go.Scatter3d(x=j, y=i, z=k, mode='lines', line=grid_line_marker, showlegend=False),
            row=1, col=2, 
        ) 

    
    
    ## add trajectory
    t = np.linspace(0, 2*np.pi, 500)
    r = 4.0
    x_traj, y_traj = r*np.cos(t), r*np.sin(t)
    X_traj = np.array([x_traj, y_traj]).T
    Z_traj = f_0(X_traj, W1, W2, w3).reshape(-1)
    fig.add_trace(
        trace=go.Scatter3d(
            x=x_traj, y=y_traj, z=Z_traj, mode='lines', 
            line=dict(color='yellow'), showlegend=True,
            legend="legend2"
        ), row=1, col=2, 
    )

    no_meshgrid_visible_traces = []
    number_of_traces = len(fig.data)
    for trace in fig.data:
        if trace.type == 'scatter3d':
            no_meshgrid_visible_traces.append(False)
        else:
            no_meshgrid_visible_traces.append(True)

    meshgrid_visible_traces = [True]*number_of_traces

    ## we want the trajectory to be visible by default
    no_meshgrid_visible_traces[-1] = True

    ## stores the current layout
    if meshgrid_store is None:
        meshgrid_store = meshgrid_visible_traces

    if camera_store is None:
        camera_store = dict(center=dict(x=0, y=0, z=0), eye=dict(x=1.25, y=1.25, z=1.25))
    if relayout_data and 'scene.camera' in relayout_data:
        camera_store = relayout_data['scene.camera']

    fig.update_layout(
        scene_camera=camera_store,
        title="$$ \\text{A two-layer neural network with 4 nodes in each layer: } f_0(X) = ((\\text{XW}_1)_{+} \\text{W}_2)_{+} \\textbf{w}_3 $$",
        width=1400, height=800,
        margin=dict(l=65, r=50, b=65, t=90),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.8,
                y=1.1,
                buttons=list([
                    dict(label="Meshgrid On",
                         method="update",
                         args=[{"visible": meshgrid_visible_traces}, {"scene_camera": camera_store}]),
                    dict(label="Meshgrid Off",
                         method="update",
                         args=[{"visible": no_meshgrid_visible_traces}, {"scene_camera": camera_store}]),
                ]),
            )
        ],
        legend2={
            "title": "",
            "xref": "container",
            "yref": "container",
            "x": 0.1,
            "y": 0.5,
        },
    )

    return fig, meshgrid_store, camera_store


if __name__ == '__main__':
    app.run(debug=True)