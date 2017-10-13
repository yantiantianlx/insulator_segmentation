import tensorflow as tf
from tfRecords import TFRecords_Reader
import numpy as np

sess = tf.InteractiveSession()#config = tf_config)

NUM_EXAMPLE = 9600 #number of samples
NUM_EPOCH = 1000
BATCH_SIZE = 10
#SAVER_STEP = (NUM_EXAMPLE * NUM_EPOCH) / BATCH_SIZE / 10
SAVER_STEP = 5000

img_dir = '../img_rotation/'
annot_image_dir = '../edge_rotation/'
tfrecords_name = '../tfRecords/train_imAndanot.tfrecords'
premodel_dir = '../tfModel/tfModel0'
tfmodel_name = '../tfModel/tfModel1/model.ckpt'

# alexnet_path = '../tfModel/bvlc_alexnet.npy'
# alexnet_data_dict = np.load(alexnet_path,encoding='latin1').item()

tfreader = TFRecords_Reader(NUM_EXAMPLE)
#tfreader.write_records(img_dir,annot_image_dir,tfrecords_name)
index, image, annot_image = tfreader.readbatch_by_queue(tfrecords_name,batch_size=BATCH_SIZE,num_epoch=NUM_EPOCH)

def computLoss(im_predic,annot_im):
    cross_loss_1 = annot_im * tf.log(tf.clip_by_value(im_predic, 1e-10 ,1.0))
    loss_1 = tf.reduce_mean( tf.reduce_sum( cross_loss_1, [1,2]) )
    cross_loss_0 = (1.0 - annot_im) * tf.log(tf.clip_by_value(1.0-im_predic, 1e-10, 1.0))
    loss_0 = tf.reduce_mean( tf.reduce_sum( cross_loss_0, [1,2]) )
    loss = -(loss_0 + loss_1 ) / 2
    return loss

im_origin = tf.placeholder(tf.float32, shape=[None, 300,400,3])
annot_im = tf.placeholder(tf.float32, shape=[None, 300,400,1])

im_norm = ( im_origin - 128.0 ) / 255.0
annot_im_norm = annot_im / 255.0
keep_prob = tf.placeholder(tf.float32)

## conv1 layer ##
#300*400*3->75*100*96->38*50*96
W_conv1 = tf.Variable(tf.truncated_normal([11,11, 3,96], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[96]))
conv1 = tf.nn.conv2d(im_norm, W_conv1, strides=[1, 4, 4, 1], padding='SAME')
h_conv1 = tf.nn.relu(conv1 + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
## conv2 layer ##
#38*50*96->38*50*256->19*25*256
W_conv2 = tf.Variable(tf.truncated_normal([5,5,96,256], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[256]))
conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
h_conv2 = tf.nn.relu(conv2 + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
## conv3 layer ##
#19*25*256->19*25*384
W_conv3 = tf.Variable(tf.truncated_normal([3,3,256,384], stddev=0.1))
b_conv3 = tf.Variable(tf.constant(0.1, shape=[384]))
conv3 = tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
h_conv3 = tf.nn.relu(conv3 + b_conv3)
## conv4 layer ##
#19*25*384->19*25*384
W_conv4 = tf.Variable(tf.truncated_normal([3,3,384,384], stddev=0.1))
b_conv4 = tf.Variable(tf.constant(0.1, shape=[384]))
conv4 = tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 1, 1, 1], padding='SAME')
h_conv4 = tf.nn.relu(conv4 + b_conv4)
## conv5 layer ##
#19*25*384->19*25*256->10*13*256
W_conv5 = tf.Variable(tf.truncated_normal([3,3,384,256], stddev=0.1))
b_conv5 = tf.Variable(tf.constant(0.1, shape=[256]))
conv5 = tf.nn.conv2d(h_conv4, W_conv5, strides=[1, 1, 1, 1], padding='SAME')
h_conv5 = tf.nn.relu(conv5 + b_conv5)
h_pool5 = tf.nn.max_pool(h_conv5, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
## c6 layer ##
#10*13*256->10*13*4096
W_c6 = tf.Variable(tf.truncated_normal([1,1,256,4096], stddev=0.1))
b_c6 = tf.Variable(tf.constant(0.1, shape=[4096]))
c6 = tf.nn.conv2d(h_pool5, W_c6, strides=[1, 1, 1, 1], padding='SAME')
c6_relu = tf.nn.relu(c6 + b_c6)
## c7 layer ##
#10*13*4096->10*13*4096
W_c7 = tf.Variable(tf.truncated_normal([1,1,4096,4096], stddev=0.1))
b_c7 = tf.Variable(tf.constant(0.1, shape=[4096]))
c7 = tf.nn.conv2d(c6, W_c7, strides=[1, 1, 1, 1], padding='SAME')
c7_relu = tf.nn.relu(c7 + b_c7)
## c8 layer ##
#10*13*4096->10*13*1
W_c8 = tf.Variable(tf.truncated_normal([1,1,4096,1], stddev=0.1))
b_c8 = tf.Variable(tf.constant(0.1, shape=[1]))
c8 = tf.nn.conv2d(c7, W_c8, strides=[1, 1, 1, 1], padding='SAME') + b_c8

######deconvolution1 : ... * 2 = x2############
#deconv1:c8 10*13*1->19*25*1
W_deconv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 1]))
c8_2x = tf.nn.conv2d_transpose(c8, W_deconv1,[BATCH_SIZE, 19, 25, 1], [1, 2, 2, 1], 'SAME')
#reduce dim: conv5 19*25*256->19*25*1
W_5 = tf.Variable(tf.truncated_normal([1,1,256,1], stddev=0.1))
h_5 = tf.nn.conv2d(h_conv5,W_5,strides=[1, 1, 1, 1], padding='SAME')
x2 = c8_2x + h_5

######deconvolution2 : x2 * 2 = x4############
#deconv1:x2 19*25*1->38*50*1
W_deconv2 = tf.Variable(tf.truncated_normal([3, 3, 1, 1]))
#h = (h_in - 1) * stride_h + kernel_h - 2 * pad_h
x2_2 = tf.nn.conv2d_transpose(x2, W_deconv2,[BATCH_SIZE, 38, 50, 1], [1, 2, 2, 1], 'SAME')
#reduce dim: conv2 38*50*256->38*50*1
W_2 = tf.Variable(tf.truncated_normal([1,1,256,1], stddev=0.1))
h_2 = tf.nn.conv2d(h_conv2,W_2, strides=[1, 1, 1, 1], padding='SAME')
x4 = x2_2 + h_2

######deconvolution3 : x4 * 8 = x32############
#deconv1:x2 38*50*1->300*400*1
W_deconv3 = tf.Variable(tf.truncated_normal([3, 3, 1, 1]))
x32 = tf.nn.conv2d_transpose(x4, W_deconv3,[BATCH_SIZE, 300, 400, 1], [1, 8, 8, 1], 'SAME')
x32_norm = tf.nn.sigmoid(x32)
loss_L2 = computLoss(x32_norm,annot_im_norm)

#train_step = tf.train.AdadeltaOptimizer(learning_rate=0.2, rho=0.95, epsilon=1e-08).minimize(loss_L2)
train_step = tf.train.AdadeltaOptimizer(0.5).minimize(loss_L2)

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


ckpt = tf.train.get_checkpoint_state(premodel_dir)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('model loading')
else:
   pass

j = 0
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
try:
    while not coord.should_stop():
        j = j + 1
        index_batch, image_batch, annot_image_batch = sess.run([index, image, annot_image])
        # if(j+1) % 10 == 0:
        print(j + 1, sess.run(loss_L2, feed_dict={im_origin: image_batch, annot_im: annot_image_batch}))
        sess.run(train_step, feed_dict={im_origin: image_batch, annot_im: annot_image_batch})
        # print 'j=', j, '\tloss_L2=', sess.run(loss_L2, feed_dict={im: img_batch, x: x_batch, y: y_batch, keep_prob: 1})
        # print(sess.run(computLoss(x_y, x, y)[1], feed_dict={im: img_batch, x: x_batch, y: y_batch, keep_prob: 1}))
        # print(sess.run(computLoss(x_y, x, y)[2], feed_dict={im: img_batch, x: x_batch, y: y_batch, keep_prob: 1}))
        if (j + 1) % SAVER_STEP == 0:
            saver.save(sess, tfmodel_name, global_step=j + 1)
except tf.errors.OutOfRangeError:
    print('Done training')
finally:
    coord.request_stop()

coord.join(threads)

sess.close()
