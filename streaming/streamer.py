from __future__ import unicode_literals
from tweepy import OAuthHandler
from tweepy import Stream
import tweepy, datetime, signal, sys, json
import redis
import json

# Variables that contains the user credentials to access Twitter API
access_token = "4010739394-F156dCIH53L1pfstMxF7PlqkmfDJEZJScb0qGAv"
access_token_secret = "7ZynMAw4hsnmyFfifGl9z8omFKqBRDbCcbhO1rgiqQW51"
consumer_key = "NhwzdxzKaBRemgIpnnuZFmyNd"
consumer_secret = "EB6OJaK5IGbfXPjMKsk1nRY9GxdgHuEsGZDAKVh0VJubp9iSFO"

trump_id = '25073877'
hilary_id = '1339835893'
sanders_id = '216776631'
bush_id = '113047940'
cruz_id = '23022687'
carson_id = '1180379185'
christie_id = '1347285918'
rubio_id = '15745368'


# Abbeviations
# CT: candidate thought
# OT: original thought (post by non candidate)
# RT: retweet

class StreamListener(tweepy.StreamListener):
    """
    This is a basic listener that just prints received tweets to stdout.
    """
    def __init__(self):
        super(tweepy.StreamListener, self).__init__()
        self.redis = redis.Redis(host="redis")


    def emitEval(self, sentence):
        sentence = sentence.encode('ascii', 'replace')
        self.redis.publish("model", json.dumps(["eval", sentence]))


    def on_data(self, _data):
        data = json.loads(_data)
        if "delete" not in data:
            if ( "retweeted_status" not in data and
                  ( data["user"]["id_str"] == trump_id or
                    data["user"]["id_str"] == hilary_id or
                    data["user"]["id_str"] == sanders_id or
                    data["user"]["id_str"] == bush_id or
                    data["user"]["id_str"] == cruz_id or
                    data["user"]["id_str"] == carson_id or
                    data["user"]["id_str"] == christie_id or
                    data["user"]["id_str"] == rubio_id)):
                #print("CT: %s" % data["text"])
                result = data["text"]
                self.emitEval(data["text"], result)

            elif "retweeted_status" not in data:
                #print("OT: %s" % data["text"])
                self.emitEval(data["text"])

            #elif "retweeted_status" in data:
            #    #print("RT: %s" % data["text"])
            #    result = eval(data["text"])
            #    self.emitEval(data["text"], result)

        return True

    def on_error(self, status):
        print(status)


def GracefulExit(_signal, frame):
    if _signal is signal.SIGINT:
        print("\nShutting down...")
        conn.close()
        sys.exit(0)


if __name__ == "__main__":
    # set up exit handler
    signal.signal(signal.SIGINT, GracefulExit)

    # This handles Twitter authetification and the connection to Twitter Streaming API
    listener = StreamListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, listener)

    # connect to db

    print("Starting Stream...")
    sys.stdout.flush()
    #while(True):
    #    try:
    stream.filter(follow=[trump_id, hilary_id, sanders_id, bush_id, cruz_id, carson_id, christie_id, rubio_id], async=False)
    #    except Exception:
    #        print(e)
