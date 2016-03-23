from flask import Flask
from flask_restful import Resource, Api
from NeuroticAdHominem import eval

def launch():

    app = Flask(__name__, static_url_path='', static_folder='app')
    app.secret_key = "bGhmsER0KydumulGTRa2"
    api = Api(app)

    @app.route('/')
    def root():
        return app.send_static_file('index.html')

    class EvalString(Resource):
        def get(self, test):
            return {"result": eval(test)}

    api.add_resource(EvalString, "/eval/<string:test>")

    app.run(debug=True, host="0.0.0.0")
