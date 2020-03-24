import unittest
import resnet
import tensorflow.compat.v1 as tf


from keras import backend as K


import numpy as np
class modelTest(unittest.TestCase):

    def test_convnet(self):
      spec = tf.placeholder(tf.float32, shape=(None,128, 256, 1))
      model = resnet.ResnetBuilder.build_resnet_18((128,256,1),1)
      sess = tf.Session()
      sess.run(tf.global_variables_initializer())
      before = sess.run(tf.trainable_variables())
      _ = sess.run(model.fit, feed_dict={
                   spec: np.ones((1, 128, 256, 1)),
                   })
      after = sess.run(tf.trainable_variables())
      for b, a, n in zip(before, after):
          # Make sure something changed.
          assert (b != a).any()

if __name__ == '__main__':
    unittest.main()
