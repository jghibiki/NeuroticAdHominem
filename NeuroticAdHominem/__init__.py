
# Globals
vocabulary_inv = None
vocabulary = None

from options import Options
options = Options()
Options = options


from models.TextCNN import TextCNN

import training
from training.train import train

from hosting.host import host

import api
from api.ApiController import launch


