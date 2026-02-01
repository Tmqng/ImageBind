import frontend

from dash import Dash

import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
assets_path = os.path.join(root_path, ".assets")

app = Dash(
    __name__,
    assets_folder=assets_path,
    assets_url_path='my_assets'
)

app.layout = frontend.create_layout()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8050, debug=True)