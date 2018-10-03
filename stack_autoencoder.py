from __future__ import print_function
import tensorflow as tf
import properties
from NeuralNet import NeuralNetwork


class StackAutoEncoder(NeuralNetwork):
    
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(args, kwargs)


    