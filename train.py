import tensorflow as tf
import time

from utils.dataset import dataset
from net.network import PoseNet
import net.config as cfg

slim = tf.contrib.slim


training_data = dataset(data_size = cfg.DATA_SIZE)
net = PoseNet()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter(cfg.PATH + '/save', graph = sess.graph)

#loading YOLO weights as pretrained parameters
if cfg.PRETRAIN_WEIGHTS is not None:
	include = []
	for num in [2, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 28, 29, 30]:
		weights = 'yolo/conv_' + str(num) + '/weights'
		biases = 'yolo/conv_' + str(num) + '/biases'
		include.append(weights)
		include.append(biases)
	variables_to_restore = slim.get_variables_to_restore(include = include)
	restorer = tf.train.Saver(variables_to_restore, max_to_keep = None)
	print('Loading pretrained weights from ' + cfg.PRETRAIN_WEIGHTS)
	restorer.restore(sess, cfg.PRETRAIN_WEIGHTS)


print('Start Training.....')
training_start = time.time()
for step in range(1, cfg.MAX_ITERS + 1):

	load_start = time.time()
	image_batch, label_batch = training_data.get_next_batch()
	load_end = time.time()

	feed_dict = {net.images: image_batch, net.labels: label_batch, 
	net.keep_prob: 0.5, 
	net.is_training: True, 
	net.learning_rate: cfg.LEARNING_RATE,
	net.decay_steps: cfg.DECAY_STEPS,
	net.decay_rate: cfg.DECAY_RATE}

	if step % cfg.SUMMARY_ITERS == 0:
		if step % (cfg.SUMMARY_ITERS * 10) == 0:

			train_start = time.time()
			_, loss, summary_str = sess.run([net.train_op, net.total_loss, net.summary_op], feed_dict = feed_dict)
			train_end = time.time()
		
			log = ('Step: {}/{}, Loss: {:.5f},'
				' Load: {:.2f}s/iter, Train: {:.2f}s/iter, Remain: {:.2f}h').format(
				step, cfg.MAX_ITERS,
				loss,
				load_end - load_start,
				train_end - train_start,
				(train_end - load_start) * (cfg.MAX_ITERS - step)/3600)
			print(log)
		else:
			_, loss, summary_str = sess.run([net.train_op, net.total_loss, net.summary_op], feed_dict = feed_dict)
		writer.add_summary(summary_str, step)
	else:
		sess.run(net.train_op, feed_dict = feed_dict)

	if step % cfg.SAVE_ITERS == 0:
		tf.train.Saver(max_to_keep = cfg.MAX_TO_KEEP).save(sess, cfg.PATH + '/save/weights', global_step = net.global_steps)

print('Training Done!')
training_end = time.time()

conf_file = cfg.PATH + '/net/config.py'
conf_txt = open(cfg.PATH + '/save/config.txt', 'w')
with open(conf_file, 'r') as f:
	for lines in f:
		conf_txt.write(lines)
	conf_txt.write( '\nTraining time = ' + str(int((training_end - training_start)/60)) + ' min')
conf_txt.close()