"""Vercel WSGI entrypoint exposing the Dash Flask server as `app`."""
from dashboard import app as dash_app

# Vercel looks for a top-level WSGI/ASGI variable named `app`.
app = dash_app.server

if __name__ == "__main__":
    dash_app.run(debug=True, host="127.0.0.1", port=8050)
