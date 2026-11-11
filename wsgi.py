"""Vercel WSGI entrypoint exposing the Dash Flask server as `app`."""
from dashboard import app as dash_app

app = dash_app.server
