import tensorflow as tf

x = tf.constant(5.0, shape=[5, 6])
print(x.get_shape())
w = tf.constant([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
print(w.get_shape())
xw = tf.multiply(x, w)
max_in_rows = tf.reduce_max(xw, 1)

sess = tf.Session()
print sess.run(xw)
# ==> [[0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
#      [0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
#      [0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
#      [0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
#      [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]]

print sess.run(max_in_rows)
# ==> [25.0, 25.0, 25.0, 25.0, 25.0]
