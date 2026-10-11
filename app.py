"""Main Streamlit Entrypoint for Cloud & Local Deployment.

Delegates directly to render_dashboard in streamlit_app.py.
"""

from streamlit_app import render_dashboard

if __name__ == "__main__":
    render_dashboard()
