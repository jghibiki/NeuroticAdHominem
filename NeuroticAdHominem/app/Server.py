from flask import Flask
from flask_socketio import SocketIO, emit
from NeuroticAdHominem import eval

def launch():

    app = Flask(__name__, static_url_path='', static_folder='app')
    app.secret_key = "bGhmsER0KydumulGTRa2"
    socketio = SocketIO(app)


    @app.route('/')
    def root():
        return app.send_static_file('index.html')

    @socketio.on("eval")
    def handle_eval(stringToEval):
        result = eval(stringToEval)
        emit("eval::response", result)

    #start server
    socketio.run(app, debug=True, host="0.0.0.0")
