import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

temp = [10 for i in range(3)]
for i in range(3):
	temp.append(0)
# img_list = []
# for i in range(6):
# 	img_list.append(temp)
# img_arr = np.array(img_list)

# hori_img_arr = img_arr.T

# temp_filter = [1, 0, -1]
# filter_arr = []
# for i in range(3):
# 	filter_arr.append(temp_filter)
# filter_arr = np.array(filter_arr)

# hori_filter_arr = filter_arr.T

# img_arr = img_arr.reshape((1, 6, 6, 1))
# filter_arr = filter_arr.reshape((3, 3, 1, 1))
# hori_filter_arr = hori_filter_arr.reshape((3, 3, 1, 1))

# hori_img_arr_new = hori_img_arr.reshape((1, 6, 6, 1))

# # convert numpy arrays into tensors
# x = tf.constant(img_arr, dtype=tf.float32)
# x2 = tf.constant(hori_img_arr_new, dtype=tf.float32)
# kernel = tf.constant(filter_arr, dtype=tf.float32)
# kernel2 = tf.constant(hori_filter_arr, dtype=tf.float32)

# output_ver = np.array(tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')).reshape(4, 4)
# output_hor = np.array(tf.nn.conv2d(x, kernel2, strides=[1, 1, 1, 1], padding='VALID')).reshape(4, 4)

# op_hor = np.array(tf.nn.conv2d(x2, kernel2, strides=[1, 1, 1, 1], padding='VALID')).reshape(4, 4)

# plt.imshow(output_ver, cmap="gray")
# plt.savefig('ver_op.png')
# plt.clf()

# plt.imshow(output_hor, cmap="gray")
# plt.savefig('hor_op.png')
# plt.clf()


# plt.imshow(hori_filter_arr.reshape((3, 3)), cmap="gray")
# plt.savefig('hor_filter.png')
# plt.clf()

# print(op_hor)

# plt.imshow(hori_img_arr, cmap="gray")
# plt.savefig('hori_image_ip.png')
# plt.clf()

# plt.imshow(op_hor, cmap="gray")
# plt.savefig('hori_image_op.png')
# plt.clf()

img_list = []
for i in range(3):
	img_list.append(temp)

for i in range(3):
	img_list.append(temp[::-1])
img_arr = np.array(img_list)
plt.imshow(img_arr, cmap="gray")
plt.savefig('mixed.png')
plt.clf()

temp_filter = [1, 0, -1]
filter_arr = []
for i in range(3):
	filter_arr.append(temp_filter)
filter_arr = np.array(filter_arr)

hori_filter_arr = filter_arr.T

img_arr = img_arr.reshape((1, 6, 6, 1))
filter_arr = filter_arr.reshape((3, 3, 1, 1))
hori_filter_arr = hori_filter_arr.reshape((3, 3, 1, 1))

# convert numpy arrays into tensors
x = tf.constant(img_arr, dtype=tf.float32)
kernel = tf.constant(filter_arr, dtype=tf.float32)
kernel2 = tf.constant(hori_filter_arr, dtype=tf.float32)

output_ver = np.array(tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')).reshape(4, 4)
output_hor = np.array(tf.nn.conv2d(x, kernel2, strides=[1, 1, 1, 1], padding='VALID')).reshape(4, 4)

print(output_hor)

plt.imshow(output_ver, cmap="gray")
plt.savefig('ver_mixed_op.png')
plt.clf()

plt.imshow(output_hor, cmap="gray")
plt.savefig('hor_mixed_op.png')
plt.clf()