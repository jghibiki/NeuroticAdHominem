from threading import Event
import NeuroticAdHominem as nah
from NeuroticAdHominem import Options as opts
from NeuroticAdHominem import train
from time import sleep
from multiprocessing import Process, Pipe


conn = None
process = None

def troll(conn):
    while True:
        child_conn.recv()
        train()
        child_conn.send(True)

def processEvents():
    sleep(60)
    retrain()

def retrain(conn):
    nah.log("Starting to retrain model")
    conn.send(True)
    conn.recv() # block until finished training
    nah.log("Finished retraining model")

def launch():
    global conn
    global process
    conn, child_conn = Pipe()
    process = Process(target=troll, args=(child_conn,))
