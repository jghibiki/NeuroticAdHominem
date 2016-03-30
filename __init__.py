from time import sleep
import sys
from multiprocessing import Process, Pipe

#utils

def log(text):
    print(text)
    sys.stdout.flush()


# Globals
vocabulary_inv = None
vocabulary = None

from options import Options
options = Options()
Options = options

from training import preprocess #Vocab depends on preprocess

from models.Vocabulary import Vocabulary
vocabulary = Vocabulary()
Vocabulary = vocabulary
from models.TextCNN import TextCNN

import training
from training.train import train
from training.TrainingManager import processEvents as training_events
from training.TrainingManager import launch as training_manager_launch
from training.TrainingManager import conn as training_conn

from hosting.host import launch as hosting_launch
from hosting.host import eval

from streaming.streamer import launch as streamer_launch

from app.Server import launch as app_launch


def launch():
    hosting_launch()
    streamer_launch()

    log("Starting event loop process.")
    my_con, child_con = Pipe()
    training_manager_launch()

    event_process = Process(target=processEvents, args=(my_con, child_con, training_conn, ))
    event_process.start()


    app_launch()


def processEvents(my_con, cild_con, training_con):
    while True:
        training_events(training_conn)
        sleep(1)

