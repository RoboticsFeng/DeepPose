import tensorflow as tf
import numpy as np
import net.config as cfg
import time
from PIL import Image, ImageDraw, ImageFont
import math


imname = 'test/000005.jpg'
MIN_DISTANCE, MAX_DISTANCE = 710, 760
FOCAL_LENGTH = 1515.4

# Load network
sess = tf.Session()
restorer = tf.train.import_meta_graph('save/weights-30000.meta')
restorer.restore(sess,tf.train.latest_checkpoint('save/'))
graph = tf.get_default_graph()
images = graph.get_tensor_by_name("images:0")
logits = graph.get_tensor_by_name("yolo/logits/BiasAdd:0")
drop_out = graph.get_tensor_by_name("keep_prob:0")
is_training = graph.get_tensor_by_name("is_training:0")

# Read RGB image
inputs = np.empty((1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3), dtype = 'float32')
image = Image.open(imname).resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), Image.NEAREST)
image = np.transpose(np.array(image))
inputs[0,:,:,0] = image[0,:,:] #R
inputs[0,:,:,1] = image[1,:,:] #G`
inputs[0,:,:,2] = image[2,:,:] #B


t1 = time.time()
# Network inference
y_predict = sess.run(logits, feed_dict = {images: inputs, drop_out: 1.0, is_training: False})

t2 = time.time()
# Post processing
boundary1 = cfg.CELL_SIZE * cfg.CELL_SIZE * 3
boundary2 = cfg.CELL_SIZE * cfg.CELL_SIZE + boundary1

predict_location = np.reshape(y_predict[0, : boundary1], [cfg.CELL_SIZE, cfg.CELL_SIZE, 3])
predict_confidence = np.reshape(y_predict[0, boundary1 : boundary2], [cfg.CELL_SIZE, cfg.CELL_SIZE])
predict_view_points = np.reshape(y_predict[0, boundary2 :], [cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.VIEW_POINTS])

max_view_points = np.max(predict_view_points, axis = 2)
view_points_index = np.argmax(predict_view_points, axis = 2)
probability_map = predict_confidence * max_view_points
i, j = divmod(np.argmax(probability_map), cfg.CELL_SIZE)

location = predict_location[i, j, :]
location_x = (i + location[0]) * 64 * 1280/448
location_x = int(location_x)
location_y = (j + location[1]) * 64 * 1024/448
location_y = int(location_y)
distance = location[2] * (MAX_DISTANCE - MIN_DISTANCE) + MIN_DISTANCE


viewpoints = view_points_index[i, j]
orientation_id = int(viewpoints/10)
rotation_id = viewpoints%10
rotation = rotation_id * 36 + 18
print(orientation_id, rotation)




print('network output:', location_x, location_y, distance, rotation, orientation_id)
print('estimated position: ')
print(pose)
print(xyz_r)



img = Image.open(imname).convert('RGB').resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), Image.NEAREST)
draw = ImageDraw.Draw(img)
for i in range(1, 8):
	draw.line([64*i, 0, 64*i, 448], fill = (255, 255, 255))
	draw.line([0, 64*i, 448, 64*i], fill = (255, 255, 255))

font = ImageFont.truetype("arial.ttf", 20)
for i in range(cfg.CELL_SIZE):
	for j in range(cfg.CELL_SIZE):
		#prob = probability_map[i, j]
		prob = predict_confidence[i, j]
		if prob > 0.2:
			prob = format(prob, '.2f')
			draw.text((i*64+14, j*64+22), prob, fill = (0,255,0,128), font = font)
		else:
			prob = format(prob, '.2f')
			draw.text((i*64+14, j*64+22), prob, fill = (255,0,0,128), font = font)
img.save('map.png')


image_index = orientation_id
if image_index == 0:
	radius = 732.0
elif image_index == 1:
	radius = 734.0
elif image_index == 2:
	radius = 730.9 
		

def show_result(scene, body, position, distance, rotation, radius):
	FOCAL_LENGTH = 1515.4
	ALPHA = 0.5
	'''Mix the object and background
		scene---background
		body---object
		position---object postion in backgound [x in pixels, y in pixels]
		distance---distance to the camera origin in mm
		rotation---rataton angel in degree [degree]
	'''
	resize_ratio = radius / distance
	size = (int(resize_ratio * body.size[0]), int(resize_ratio * body.size[1]))
	body = body.resize(size, Image.NEAREST).convert('RGB')
	body = body.rotate(rotation - 90).transpose(Image.FLIP_LEFT_RIGHT)
	body = np.array(body)
	scene = np.array(scene.convert('RGB'))
	
	offset_x = int(position[0] - 0.5*body.shape[0])
	offset_y = int(position[1] - 0.5*body.shape[1])
	for i in range(scene.shape[1]):
		for j in range(scene.shape[0]):
			index_x = i - offset_x
			index_y = j - offset_y	
			if index_x>=0 and index_x<body.shape[0] and index_y>=0 and index_y<body.shape[1]:
				if body[index_x][index_y][0]>0:
					scene[j][i] = np.array([0, 255, 0]) * ALPHA + scene[j][i] * (1 - ALPHA)

	im=Image.fromarray(scene)
	return im


body = Image.open('test/object/' + str(image_index) + '.png')
background_image = Image.open(imname)
result = show_result(background_image, body, [location_x, location_y], distance, rotation, radius)
result.save('result.png')

