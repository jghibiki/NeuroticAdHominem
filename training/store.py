import os, sys


options = {
        "sentence_padding": os.getenv('SENTENCE_PADDING', 50),
        "sentence_padding_token": os.getenv('SENTENCE_PADDING_TOKEN', "<PAD/>"),

        # CNN Parameters
        "embedding_dim": os.getenv('EMBEDDING_DIM', 20),
        "filter_sizes": os.getenv('FILTER_SIZES',  "3,4,5"),
        "num_filters": os.getenv('NUM_FILTERS',  50),
        "dropout_keep_prob": os.getenv('DROPOUT_KEEP_PROB', 0.5),
        "l2_reg_lambda": os.getenv('L2_REG_LAMBDA', 3.0),
        "vocab_oversizing": os.getenv('VOCAB_OVERSIZING', 5000),

        # Training parameters
        "batch_size": os.getenv('BATCH_SIZE', 64),
        "num_epochs": os.getenv('NUM_EPOCHS',  2),
        "evaluate_every": os.getenv('EVALUATE_EVERY',  100),
        "checkpoint_every": os.getenv('CHECKPOINT_EVERY', 100),

        # Export Parameters
        "export_version": os.getenv('EXPORT_VERSION', 1),
        "model_location": os.getenv('MODEL_LOCATION', "/tmp/"),

        # Misc Parameters
        "allow_soft_placement": os.getenv('ALLOW_SOFT_PLACEMENT', True),
        "log_device_placement": os.getenv('LOG_DEVICE_PLACEMENT', False),
        "display_train_steps": os.getenv('DISPLAY_TRAIN_STEPS', False )
}




from Vocabulary import Vocabulary
Vocabulary = Vocabulary()
vocab = Vocabulary
vocabulary = vocab

def log(msg):
    print(msg)
    sys.stdout.flush()
