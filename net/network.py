import numpy as np
import tensorflow as tf
import net.config as cfg

slim = tf.contrib.slim


class PoseNet:

	def __init__(self):
		self.image_size = cfg.IMAGE_SIZE
		self.cell_size = cfg.CELL_SIZE
		self.view_points = cfg.VIEW_POINTS

		self.location_scale = cfg.LOCATION_SCALE
		self.rotation_scale = cfg.LOCATION_SCALE
		self.confidence_scale = cfg.CONFIDENCE_SCALE
		self.noobj_confidence_scale = cfg.NOOBJ_CONFIDENCE_SCALE
		self.view_points_scale = cfg.VIEW_POINTS_SCALE

		self.batch_size = cfg.BATCH_SIZE
		self.alpha = cfg.LEAKY_RELU_ALPHA
		
		self.output_size = self.cell_size * self.cell_size * (4 + self.view_points)
		self.boundary1 = self.cell_size * self.cell_size * 3
		self.boundary2 = self.cell_size * self.cell_size + self.boundary1

		self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name = 'images')
		self.labels = tf.placeholder(tf.float32, [None, self.cell_size, self.cell_size, 4 + self.view_points], name = 'labels')

		#hyperparameters for training and inference
		self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
		self.is_training = tf.placeholder(tf.bool, name = 'is_training')
		self.logits = self.build_network(self.images, self.output_size, self.alpha, self.is_training, self.keep_prob)#name is 'yolo/logits/BiasAdd'
		
		self.global_steps = tf.Variable(0, trainable = False)
		self.learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
		self.decay_steps = tf.placeholder(tf.float32, name = 'decay_steps')
		self.decay_rate = tf.placeholder(tf.float32, name = 'decay_rate')
		self.total_loss = self.loss(self.logits, self.labels)#name is 'loss_layer/AddN:0'
		self.train_op = tf.train.GradientDescentOptimizer(
			tf.train.exponential_decay(self.learning_rate, self.global_steps, self.decay_steps, self.decay_rate), 
			name='train_op').minimize(self.total_loss, global_step = self.global_steps)
		self.summary_op = tf.summary.merge_all()#name is Merge/MergeSummary:0


	def build_network(self,
                      images,
                      num_outputs,
                      alpha,
                      is_training = True,
                      keep_prob = 0.5,
                      scope = 'yolo'):
		with tf.variable_scope(scope):
			with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn = leaky_relu(alpha),
                                weights_initializer = tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer = slim.l2_regularizer(0.0002)):
				net = tf.pad(images / 255, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), name = 'pad_1')
				net = slim.conv2d(net, 64, 7, 2, padding = 'VALID', scope = 'conv_2')
				net = slim.max_pool2d(net, 2, padding = 'SAME', scope = 'pool_3')
				net = slim.conv2d(net, 192, 3, scope = 'conv_4')
				net = slim.max_pool2d(net, 2, padding = 'SAME', scope = 'pool_5')
				net = slim.conv2d(net, 128, 1, scope = 'conv_6')
				net = slim.conv2d(net, 256, 3, scope = 'conv_7')
				net = slim.conv2d(net, 256, 1, scope = 'conv_8')
				net = slim.conv2d(net, 512, 3, scope = 'conv_9')
				net = slim.max_pool2d(net, 2, padding = 'SAME', scope = 'pool_10')
				net = slim.conv2d(net, 256, 1, scope = 'conv_11')
				net = slim.conv2d(net, 512, 3, scope = 'conv_12')
				net = slim.conv2d(net, 256, 1, scope = 'conv_13')
				net = slim.conv2d(net, 512, 3, scope = 'conv_14')
				net = slim.conv2d(net, 256, 1, scope = 'conv_15')
				net = slim.conv2d(net, 512, 3, scope = 'conv_16')
				net = slim.conv2d(net, 256, 1, scope = 'conv_17')
				net = slim.conv2d(net, 512, 3, scope = 'conv_18')
				net = slim.conv2d(net, 512, 1, scope = 'conv_19')
				net = slim.conv2d(net, 1024, 3, scope = 'conv_20')
				net = slim.max_pool2d(net, 2, padding = 'SAME', scope='pool_21')
				net = slim.conv2d(net, 512, 1, scope = 'conv_22')
				net = slim.conv2d(net, 1024, 3, scope = 'conv_23')
				net = slim.conv2d(net, 512, 1, scope = 'conv_24')
				net = slim.conv2d(net, 1024, 3, scope = 'conv_25')
				net = slim.conv2d(net, 1024, 3, scope = 'conv_26')
				net = tf.pad(net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name = 'pad_27')
				net = slim.conv2d(net, 1024, 3, 2, padding = 'VALID', scope = 'conv_28')
				net = slim.conv2d(net, 1024, 3, scope = 'conv_29')
				net = slim.conv2d(net, 1024, 3, scope = 'conv_30')
				net = tf.transpose(net, [0, 3, 1, 2], name = 'trans_31')
				net = slim.flatten(net, scope = 'flat_32')
				net = slim.fully_connected(net, 512, scope = 'fc_33')
				net = slim.fully_connected(net, 4096, scope = 'fc_34')
				net = slim.dropout(net, keep_prob = keep_prob, is_training = is_training, scope = 'dropout_35')
				net = slim.fully_connected(net, num_outputs, activation_fn = None, scope = 'logits')
		return net



	def loss(self, y_predict, y_label, scope = 'loss_layer'):
	# y_predict is [batch_size, self.output_size] tensor
	# y_label is [batch_size, cell_size, cell_size, 5 + self.view_points] tensor
		with tf.variable_scope(scope):
			predict_location = tf.reshape(y_predict[:, :self.boundary1], [self.batch_size, self.cell_size, self.cell_size, 3])
			predict_confidence = tf.reshape(y_predict[:, self.boundary1:self.boundary2], [self.batch_size, self.cell_size, self.cell_size])
			predict_view_points = tf.reshape(y_predict[:, self.boundary2:], [self.batch_size, self.cell_size, self.cell_size, self.view_points])

			label_location = y_label[:, :, :, :3]
			delta_obj_reduced = y_label[:, :, :, 3]
			delta_obj = tf.reshape(delta_obj_reduced, [self.batch_size, self.cell_size, self.cell_size, 1])
			label_view_points = y_label[:, :, :, 4:]



			location_delta = delta_obj * (predict_location - label_location)
			location_loss = tf.reduce_mean(tf.reduce_sum(tf.square(location_delta), axis=[1, 2, 3])) * self.location_scale
	

			location_square_error = tf.reduce_sum(tf.square(predict_location - label_location), axis = 3)
			groundtruth_confidence = delta_obj_reduced / tf.exp(location_square_error)
			all_confidence_delta = predict_confidence - groundtruth_confidence
			obj_confidence_loss = tf.reduce_mean(tf.reduce_sum(delta_obj_reduced * tf.square(all_confidence_delta), axis=[1, 2])) * self.confidence_scale
			noobj_confidence_delta = (tf.ones_like(delta_obj_reduced, dtype = tf.float32) - delta_obj_reduced) * all_confidence_delta
			noobj_confidence_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobj_confidence_delta), axis=[1, 2])) * self.noobj_confidence_scale

			view_points_delta = delta_obj * (predict_view_points - label_view_points)
			view_points_loss = tf.reduce_mean(tf.reduce_sum(tf.square(view_points_delta), axis=[1, 2, 3])) * self.view_points_scale
		
			tf.losses.add_loss(location_loss)
			tf.losses.add_loss(obj_confidence_loss)
			tf.losses.add_loss(noobj_confidence_loss)
			tf.losses.add_loss(view_points_loss)
			total_loss = tf.losses.get_total_loss(name = None)
			
			tf.summary.scalar('location_loss', location_loss)
			tf.summary.scalar('obj_confidence_loss', obj_confidence_loss)
			tf.summary.scalar('noobj_confidence_loss', noobj_confidence_loss)
			tf.summary.scalar('view_points_loss', view_points_loss)
			tf.summary.scalar('total_loss', total_loss)

		return total_loss


def leaky_relu(alpha):
	return lambda inputs: tf.maximum(alpha * inputs, inputs, name = 'leaky_relu')