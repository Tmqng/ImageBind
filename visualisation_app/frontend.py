import pandas as pd
import plotly.express as px
from dash import callback, Dash, html, dcc, Input, Output, State, dash_table

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
] # this has to update dynamically depending on the content in .assets

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
            # 1. Left Column: Multimedia (Images/Audio)
            dbc.Col(
                id='multimedia-container', 
                style={
                    'flex': '1.5'  # Slightly wider than the calculation columns
                }
            ),

            # 2. Middle Column: Embeddings Logic
            dbc.Col([
                html.Button(
                    "Compute Embeddings", 
                    id='compute-embeddings-btn', 
                    n_clicks=0,
                    style={'width': '100%', 'marginBottom': '10px'}
                ),
                html.Pre(
                    id='embeddings-vectors', 
                    style={
                        'border': '1px solid #ccc', 
                        'padding': '10px', 
                        'maxHeight': '600px', # Increased height for a vertical column
                        'overflowY': 'auto',
                        'backgroundColor': '#fff'
                    }
                ),
            ], style={'flex': '1'}),

            # 3. Right Column: Dot Product Logic
            dbc.Col([
                html.Button(
                    "Compute Dot Products", 
                    id='compute-btn', 
                    n_clicks=0,
                    style={'width': '100%', 'marginBottom': '10px'}
                ),
                html.Pre(
                    id='dot-products', 
                    style={
                        'border': '1px solid #ccc', 
                        'padding': '10px', 
                        'maxHeight': '600px', 
                        'overflowY': 'auto',
                        'backgroundColor': '#fff'
                    }
                ),
            ], style={'flex': '1'})

        ], style={
            'display': 'flex',        # Makes the three main divs sit side-by-side
            'flexDirection': 'row', 
            'gap' : '5px',
            'padding': '20px', 
            'backgroundColor': '#f9f9f9',
            'minHeight': '100vh'      # Ensures background spans full page height
        })
    ])

    return app_layout


@callback(
    Output('multimedia-container', 'children'),
    Input('object-selector', 'value')
)
def update_multimedia_display(selected_objects):
    if not selected_objects:
        return "Select objects to view"

    rows = []
    for obj in selected_objects:
        # Create a "Card" for each object
        obj_card = dbc.Col([
            html.H4(f"{obj.capitalize()}", style={'marginBottom': '10px'}),
            
            html.Div([

                # 1. Text Component
                html.P(f"'A {obj.capitalize()}'"),

                # 2. Image Component
                html.Img(
                    src=f"my_assets/{obj}_image.jpg", 
                    style={'width': '200px', 'borderRadius': '5px'}
                ),
                
                # 3. Audio Component
                html.Audio(
                    src=f"my_assets/{obj}_audio.wav", 
                    controls=True
                ),

            ], style={
                    'display': 'flex', 
                    'flexDirection': 'column',  # This stacks them vertically
                    'alignItems': 'center',      # This centers them horizontally in the column
                    'gap': '15px',
                    'padding': '15px', 
                    'border': '1px solid #ddd', 
                    'marginBottom': '10px', 
                    'backgroundColor': 'white',
                    'width': 'fit-content'       # Ensures the border hugs the components
                })
        ])

        rows.append(obj_card)

    return rows


@callback(
    Output(
        component_id='embeddings-vectors',
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
    
    return json.dumps(filtered_output), embeddings


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
    prevent_initial_call=True              # Don't run on page load
)
def display_all_pairs_dot_products(n_clicks, selected_objects, embeddings):
    
    results = backend.compute_pairwise_dot_products(embeddings, selected_objects)

    if not results:
        return "No data available for selected objects."

    return json.dumps(results, indent=4)