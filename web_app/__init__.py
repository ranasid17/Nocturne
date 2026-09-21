try:
    import jinja2
    from markupsafe import Markup, escape

    if not hasattr(jinja2, "escape"):
        jinja2.escape = escape
    if not hasattr(jinja2, "Markup"):
        jinja2.Markup = Markup
except ImportError:
    pass

from flask import Flask, jsonify

from web_app.api import api_bp


def create_app():
    """
    Create the local Flask application.
    """

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "healthy"})

    app.register_blueprint(api_bp)

    return app
