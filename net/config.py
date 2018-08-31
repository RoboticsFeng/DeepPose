 #weights file
PRETRAIN_WEIGHTS = '/save/yolo_weights/YOLO_small.ckpt'
WEIGHTS_FILE = None


# image parameters
IMAGE_SIZE = 448
VIEW_POINTS = 30



# model parameters
CELL_SIZE = 7

LOCATION_SCALE = 2.0
CONFIDENCE_SCALE = 1.0
NOOBJ_CONFIDENCE_SCALE = 0.1
VIEW_POINTS_SCALE = 2.0

LEAKY_RELU_ALPHA = 0.1



# training parameters
DATA_SIZE = 5000
MAX_ITERS = 30000
BATCH_SIZE = 32
SUMMARY_ITERS = 10
SAVE_ITERS = 1000
MAX_TO_KEEP = 10

LEARNING_RATE = 0.005
DECAY_STEPS = 30000
DECAY_RATE = 0.1

