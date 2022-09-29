import tensorflow as tf
import numpy as np

from Demo_func import weights_to_lines_and_dots
from Demo_func import plot_dense_NN
from Demo_func import first_weight
#load model
#model=tf.keras.models.load_model('./../Trained_model/SLiM_NN_omega.h5')
model=tf.keras.models.load_model('./../Trained_model/SLiM_NN.h5')

#get from: https://stackoverflow.com/questions/52702220/access-the-weight-matrix-in-tensorflow-in-order-to-make-apply-changes-to-it-in-n
weights = model.get_weights()
first_weight=first_weight(weights)
print(first_weight)
if 1==1:
	lines,dots=weights_to_lines_and_dots(weights)
	plot_dense_NN(lines,dots)

