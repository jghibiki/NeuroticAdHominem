
# Globals
vocabulary_inv = None
vocabulary = None


from NeuralAdHominem.options import Options
options = Options()
Options = options

from NeuralAdHominem.models.TextCNN import TextCNN

import NeuralAdHominem.training
from NeuralAdHominem.training.train import train

from NeuralAdHominem.hosting.host import host
