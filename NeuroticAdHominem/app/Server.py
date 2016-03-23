from flask import Flask
from flask_restful import Resource, Api
from NeuroticAdHominem import eval

def launch():

    app = Flask(__name__)
    api = Api(app)

    class HelloWorld(Resource):
        def get(self, test):
            return {"result": eval(test)}

    api.add_resource(HelloWorld, "/<string:test>")

    app.run(debug=True, host="0.0.0.0")
