from classifyHistology.train_net import vars_phs_consts_metrics as vars
import tensorflow as tf

# training hyperparameteers
th = {
    'training_iters': 2,
    'learning_rate': 0.001,
    'batch_size': 128,
    'n_input': [32,64,3],
    'n_classes': 2,
    'net':'convNet2',
    'dropout_keep_prob': 0.5}

tf.reset_default_graph()

# create the variables to load
weights,biases,x,y,ph_is_training = vars.definePhVar(th)

# create the op to save and resotre all the variables
saver = tf.train.Saver()

# launch the model and load the restored variables
with tf.Session() as sess:
    saver.restore(sess, "/tmp/model.ckpt")
    print(weights)
    print(biases)