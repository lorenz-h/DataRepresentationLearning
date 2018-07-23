import tensorflow as tf
e = 12
e = tf.cast(e, dtype=tf.float32)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    e += 1
    print(e)
    print(sess.run(e))
