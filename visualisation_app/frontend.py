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
                    style={'width': '25%', 'marginRight': '200px'}
            ),

            html.Button(
                "Compute Dot Products", 
                id='compute-btn', 
                n_clicks=0,
                className="btn btn-success",
                style={'width': '25%'}
            ),
        ],
        className="justify-content-end mb-3",
        style={'paddingTop': '20px', 'paddingLeft': '450px'}),
        
        dbc.Row(id='main-container', children=[
            # Column 1: Data
            dbc.Col(
                id='multimedia-container',
                width=6,
            ),

            # Column 2: Embeddings
            dbc.Col([
                html.H5("Embeddings"),             
                dash_table.DataTable(
                    id='embeddings-table',
                    columns=[
                        {"name": "Object", "id": "object"},
                        {"name": "Modality", "id": "modality"},
                        {"name": "Vector (Preview)", "id": "vector"}
                    ],
                    data=[],
                    style_table={'height': '400px', 'overflowY': 'auto', 'maxWidth': '300px'},
                    style_cell={
                        'textAlign': 'left',
                        'fontFamily': 'monospace',
                        'padding': '10px'
                    },
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    }
                ),
            ], width=4), # Exactly 1/3 of the row

            # Column 3: Dot Products
            dbc.Col([
                html.H5("Alignment Scores", className="mb-3"),
                html.Pre(
                    id='dot-products',
                    style={
                        'border': '1px solid #ccc', 
                        'padding': '10px', 
                        'maxHeight': '600px',
                        'maxWidth': '300px', 
                        'overflowY': 'auto',
                        'backgroundColor': '#fff'
                    }
                ),
            ], width=4) # Exactly 1/3 of the row

        ], className="g-3", # Adds consistent spacing (gutters) between columns
        style={
            'padding': '20px', 
            'backgroundColor': '#f9f9f9',
            'minHeight': '100vh'
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
                        'flexDirection': 'column',  # This stacks them vertically
                        'alignItems': 'center',      # This centers them horizontally in the column
                        'gap': '15px',
                        'padding': '15px', 
                        'border': '1px solid #ddd', 
                        'marginBottom': '10px', 
                        'backgroundColor': 'white',
                        'width': 'fit-content'
                    }
                )
        
        rows.append(obj_row)    

    return rows


@callback(
    Output(
        component_id='embeddings-table',
        component_property='data'
    ),
    Output(
        component_id='embeddings-store',
        component_property='data'
    ),
    Input(
        component_id='compute-embeddings-btn', 
        component_property='n_clicks'
    ),
    State('object-selector', 'value'),     # The data to use
    prevent_initial_call=True              # Don't run on page load
)
def display_embeddings(n_clicks, selected_objects):
    # Filter the data based on selection
    if not selected_objects:
        return "No objects selected."
    
    embeddings = backend.get_embeddings(model, device, selected_objects)

    # Loop through each modality in the embeddings dict
    # and filter the inner dictionary based on selected_objects
    filtered_output = {
        modality: {obj: data[obj] for obj in selected_objects if obj in data}
        for modality, data in embeddings.items()
    }

    # Transform dict to list of rows for the table
    table_rows = []
    for mod, objects in filtered_output.items():
        for obj, vector in objects.items():
            # Format vector preview: [0.12, -0.45, ...]
            preview = "[" + ", ".join([f"{v:.2f}" for v in vector[:5]]) + " ...]"
            table_rows.append({
                "modality": mod,
                "object": obj,
                "vector": preview
            })
    
    return table_rows, embeddings


@callback(
    Output(
        component_id='dot-products',
        component_property='children'
    ),
    Input(
        component_id='compute-btn', 
        component_property='n_clicks'
    ),
    State('object-selector', 'value'),     # The data to use
    State('embeddings-store', 'data'),
    prevent_initial_call=True,
    suppress_callback_exceptions=True
)
def display_all_pairs_dot_products(n_clicks, selected_objects, embeddings):
    
    results = backend.compute_pairwise_dot_products(embeddings, selected_objects)

    if not results:
        return "No data available for selected objects."

    return json.dumps(results, indent=4)