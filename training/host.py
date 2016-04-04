from TextCNN import TextCNN
from store import vocab
import store
from store import options as opts
import preprocess
from EvalModel import EvalModel
import sys

import tensorflow as tf
import numpy as np
import redis
import threading
import json
from store import vocab



def launch():
    """
        Begin the model eval process
    """

    print("Launching evaluator")
    sys.stdout.flush()
    r = redis.Redis(host="redis")
    client = ModelListener(r)
    client.start()

class ModelListener(threading.Thread):

    def __init__(self, r):
        threading.Thread.__init__(self)
        # load trained word embeddings
        vocab.load()
        # load the trained model
        self.model = EvalModel()
        self.redis = r
        self.pubsub = self.redis.pubsub()
        self.pubsub.subscribe(["model"])


    def eval(self, item):
        sentence = preprocess.clean(item)
        padded_sentence = preprocess.pad(sentence)
        word_ids = []

        # get word id's
        for word in padded_sentence:
            id = vocab.getIdFromWord(word)
            word_ids.append(id)

        # run evaluation
        result = self.model.eval(np.array(word_ids))
        print("eval:: {0}:  \"{1}\"".format(result, item))
        import sys
        sys.stdout.flush()
        self.redis.publish("server", json.dumps({"sentence": item, "classification": result}))


    def reload(self):
        model = EvalModel()


    def run(self):
        for item in self.pubsub.listen():
            if item["data"] == "KILL":
                self.pubsub.unsubscribe()
                print(self, "unsubscribe and finished")
                sys.stdout.flush()
                break
            elif isinstance(item["data"], str):
                data = json.loads(item["data"])
                if(data[0] == "eval"):
                    print("eval: %s" % data[1])
                    sys.stdout.flush()
                    self.eval(data[1])
                elif(data[0] == "reload"):
                    store.log("reload")
                    self.reload()
            else:
                print("bad data: %s" % item["data"])
                sys.stdout.flush()
