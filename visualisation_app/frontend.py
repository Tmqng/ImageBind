import pandas as pd
import plotly.express as px
from dash import callback, Dash, html, dcc, Input, Output, State, dash_table

from dash import MATCH, ALL

import dash_bootstrap_components as dbc

from itertools import combinations

import json

import numpy as np

import backend

object_options = [
    {'label': 'Bird', 'value': 'bird'},
    {'label': 'Car', 'value': 'car'},
    {'label': 'Dog', 'value': 'dog'},
    {'label': 'Piano', 'value': 'piano'}
] # ideally this has to update dynamically depending on the content in .assets

model, device = backend.init_imagebind_model()

def create_layout():
    # 2. Define the Frontend (Layout)
    app_layout = html.Div([

        dcc.Store(id='embeddings-store', storage_type='memory'),

        html.H1("Multimodal Embeddings Visualisation using ImageBind encoder", style={'textAlign': 'center', 'color': '#2c3e50'}),

        html.Div([
            html.H4("Choose Objects", style={'marginBottom': '10px'}),
            
            # A Core Component (Dropdown)
            dcc.Dropdown(
                id='object-selector',
                options=object_options,
                value=[opt['value'] for opt in object_options],
                # values=[],
                multi=True
            ),
        ]),

        dbc.Row([
            html.Button(
                    "Compute Embeddings", 
                    id='compute-embeddings-btn', 
                    n_clicks=0,
                    className="btn btn-primary me-3",
                    style={'width': '25%', 'marginRight': '300px'}
            ),

            html.Button(
                "Compute Dot Products", 
                id='compute-products-btn', 
                n_clicks=0,
                className="btn btn-success",
                style={'width': '25%'}
            ),
        ],
        className="justify-content-end mb-3",
        style={'paddingTop': '20px', 'paddingLeft': '600px'}),
        
        dbc.Row(id='main-container', children=[
            # Column 1: Data
            dbc.Col(
                id='multimedia-container',
                width=4,
                style={'display': 'flex', 'flexDirection': 'column', 'minHeight': '100vh'}
            ),

            # Column 2: Embeddings
            dbc.Col(
                id='embeddings-container',
                width=4,
                style={'display': 'flex', 'flexDirection': 'column', 'minHeight': '100vh'}
            ),

            # Column 3: Dot Products
            dbc.Col(
                id='dot-products-container', 
                width=4,
                style={'display': 'flex', 'flexDirection': 'column', 'minHeight': '100vh'}
            )

        ], className="g-3", # Adds consistent spacing (gutters) between columns
        style={
            'padding': '20px', 
            'backgroundColor': '#f9f9f9',
            'minHeight': '100vh',
            'display': 'flex'
        })
    ])

    return app_layout


@callback(
    Output('multimedia-container', 'children'),
    Input('object-selector', 'value'),
)
def update_multimedia_display(selected_objects):
    if not selected_objects:
        return "Select objects to view"

    rows = []
    for obj in selected_objects:
        # Create a Row for each object
        obj_row = dbc.Row([
                    html.H4(f"{obj.capitalize()}", style={'marginBottom': '10px'}),
                    
                    dbc.Col([

                        # 1. Text Component
                        html.P(f"'A {obj.capitalize()}'"),

                        # 2. Image Component
                        html.Img(
                            src=f"my_assets/{obj}_image.jpg", 
                            style={'width': '150px'}
                        ),
                        
                        # 3. Audio Component
                        html.Audio(
                            src=f"my_assets/{obj}_audio.wav", 
                            controls=True,
                            style={},
                        ),
                    ])
                ], style={
                        'display': 'flex', 
                        'flexDirection': 'column',
                        'alignItems': 'center', 
                        'padding': '15px', 
                        'border': '1px solid #ddd', 
                        'marginBottom': '10px', 
                        'backgroundColor': 'white',
                    }
                )
        
        rows.append(obj_row)    

    return rows


@callback(
    Output(
        component_id='embeddings-container',
        component_property='children'
    ),
    Output(
        component_id='embeddings-store',
        component_property='data'
    ),
    Input(
        component_id='compute-embeddings-btn', 
        component_property='n_clicks'
    ),
    Input('object-selector', 'value'),     # The data to use
    prevent_initial_call=True              # Don't run on page load
)
def update_embeddings(n_clicks, selected_objects):
    # Filter the data based on selection
    
    embeddings = backend.get_embeddings(model, device, selected_objects)

    rows = []

    for obj in selected_objects:
        # Create a Row for each object
        obj_row = dbc.Row([

        dash_table.DataTable(
            id='modality-table',
            columns=[
                {"name": "Modality", "id": "modality"},
                {"name": "Embeddings", "id": "embedding"}
            ],
            data=[
                {"modality": "Text", "embedding": str(embeddings['text'][obj])},
                {"modality": "Vision", "embedding": str(embeddings['vision'][obj])},
                {"modality": "Audio", "embedding": str(embeddings['audio'][obj])}
            ],
            style_table={
                'width': '500px',
            },
            style_cell={
                'textAlign': 'left', 
                'padding': '10px',
                'maxWidth': '20px',      # Limits the width
                'textOverflow': 'ellipsis', # Adds "..." at the end
            },
            style_header={
                'backgroundColor': '#f2f2f2',
                'fontWeight': 'bold'
            }
        )
        ], style={
            'marginBottom': '200px'
        })
        
        rows.append(obj_row)    
    
    return rows, embeddings


@callback(
    Output(
        component_id='dot-products-container',
        component_property='children'
    ),
    Input(
        component_id='compute-products-btn', 
        component_property='n_clicks'
    ),
    Input('object-selector', 'value'),     # The data to use
    State('embeddings-store', 'data'),
    prevent_initial_call=True,
    suppress_callback_exceptions=True
)
def update_dot_products(n_clicks, selected_objects, embeddings):
    
    products = backend.compute_dot_products(embeddings, selected_objects)
    
    row = dbc.Row([

        dash_table.DataTable(
            id='product-table',
            columns=[
                {"name": "Modality", "id": "modality"},
                {"name": "Dot product", "id": "dot_product"}
            ],
            data=[
                {"modality": "Vision X Text", "dot_product": format_matrix(products['VT'])},
                {"modality": "Audio X Text", "dot_product": format_matrix(products['AT'])},
                {"modality": "Vision X Audio", "dot_product": format_matrix(products['VA'])}
            ],
            style_table={
                'width': '500px',
            },
            style_data={
                'whiteSpace': 'pre-line',
                'height': 'auto',
            },
            style_cell={
                'textAlign': 'left', 
                'padding': '10px',
                'maxWidth': '20px',      # Limits the width
                'textOverflow': 'ellipsis', # Adds "..." at the end
            },
            style_header={
                'backgroundColor': '#f2f2f2',
                'fontWeight': 'bold'
            }
        )
        ], style={
            'marginBottom': '10px'
        })
    
    return row
    

def format_matrix(tensor):
    # Convertit le tenseur en liste de listes (4x4)
    matrix = tensor.detach().cpu().numpy().tolist()
    
    # Formate chaque ligne avec des crochets et 3 décimales
    formatted_rows = []
    for row in matrix:
        row_str = ", ".join([f"{v: .3f}" for v in row])
        formatted_rows.append(f"[{row_str}]")
    
    # Joint les lignes avec un saut de ligne
    return "\n".join(formatted_rows)