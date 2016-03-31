from flask import Flask
from flask_socketio import SocketIO, emit
from flask.ext.redis import FlaskRedis
import redis
import threading
import json



app = Flask(__name__, static_url_path='', static_folder='app')
app.secret_key = "bGhmsER0KydumulGTRa2"
redis_store = FlaskRedis(app)
socketio = SocketIO(app, message_queue="redis://redis")
listener = None


@app.route('/')
def root():
    return app.send_static_file('index.html')

@socketio.on("eval")
def handle_eval(stringToEval):
    emit("eval", stringToEval)


class Listener(threading.Thread):
    def __init__(self, r, socketio):
        threading.Thread.__init__(self)
        self.redis = r
        self.socketio = socketio
        self.pubsub = self.redis.pubsub()
        self.pubsub.subscribe(["server"])

    def work(self, data):
        self.socketio.emit("stream:eval", json.loads(data))

    def run(self):
        for item in self.pubsub.listen():
            if item["data"] == "KILL":
                self.pubsub.unsubscribe()
                print(self, "unsubscribed and finished")
                break
            elif isinstance(item["data"], str):
                self.work(item["data"])


#start server
if __name__ == "__main__":
    r = redis.Redis(host="redis")
    s = SocketIO(message_queue="redis://redis")
    listener = Listener(r, s)
    listener.start()
    socketio.run(app, debug=False, host="0.0.0.0")


