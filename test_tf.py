import tensorflow as tf

c = tf.Variable(tf.random_normal([5, 2, 2, 3], stddev=0.35),name="weights")
c1 = tf.layers.conv2d_transpose(
    inputs=c,
    strides=(2,2),
    filters=1,
    kernel_size=(3,3),
    padding="SAME",
    activation=tf.nn.relu,
    name="test"
)

print(c1)
with tf.Session() as sess:
    sess.run(c1)