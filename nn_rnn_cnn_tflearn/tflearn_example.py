import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

X, Y, test_x, test_y = mnist.load_data(one_hot=True)

X = X.reshape([-1,28,28,1])
test_x = test_x.reshape([-1,28,28,1])

convent = input_data([None, 28, 28, 1], name='input')

convent = conv_2d(convent, 32, 2, activation='relu')
convent = max_pool_2d(convent, 2)

convent = conv_2d(convent, 64, 2, activation='relu')
convent = max_pool_2d(convent, 2)

convent = fully_connected(convent, 1024, activation='relu')
convent = dropout(convent, 0.8)

convent = fully_connected(convent, 10, activation='softmax')
convent = regression(convent, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convent)

# train and save model
model.fit({'input': X}, {'targets':Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}),
                snapshot_step=500, show_metric=True, run_id='mnist')

model.save('tflearncnn.model')
#load model (load only weignhts), if different structure of nn we can`t use this model
#model.load('tflearncnn.model')
# test model
#print( np.round(model.predict([test_x[1]])[0]) )
#print(test_y[1])