from flask import Flask, jsonify

from web_app.api import api_bp
from web_app.routes import pages_bp


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
    app.register_blueprint(pages_bp)

    return app
