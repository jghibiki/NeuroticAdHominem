
class Options(object):
    def __init__(self):

        # Preprocessing Parameters
        self.sentence_padding = 20
        self.sentence_padding_token = "<PAD/>"

        # CNN Parameters
        self.embedding_dim = 20 # Dimensionality of character embedding (default: 128)
        self.filter_sizes = "3,4,5" # Comma-separated filter sizes (default: '3,4,5')
        self.num_filters = 50  # Number of filters per filter size (default: 128)
        self.dropout_keep_prob = 0.5 # Dropout keep probability (default: 0.5)
        self.l2_reg_lambda =  3.0 # L2 regularizaion lambda (default: 0.0)

        # Training parameters
        self.batch_size = 64 # Batch Size (default: 64)
        self.num_epochs =  500 # Number of training epochs (default: 200)
        self.evaluate_every =  100 # Evaluate model on dev set after this many steps (default: 100)
        self.checkpoint_every =  100 # Save model after this many steps (default: 100)

        # Export Parameters
        self.export_version = 1 # Version number for exporing the model

        # Misc Parameters
        self.allow_soft_placement = True # Allow device soft device placement
        self.log_device_placement = False # Log placement of ops on devices
        self.display_train_steps = False #"toggles output of training step results")

        self.model_location = "model.ckpt"
