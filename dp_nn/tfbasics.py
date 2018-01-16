import tensorflow as tf

X1 = tf.constant(5)
X2 = tf.constant(6)
result = tf.multiply(X1, X2)  # defind model

print(result)

with tf.Session() as sess: # begin seesion
    output = sess.run(result)
    print(output)               #tf

print(output)       #python
