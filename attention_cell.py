import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import ops, constant_op, dtypes
from tensorflow.python.ops import init_ops, array_ops, math_ops, nn_ops
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
"""
    create new attention cell to predict pm25 with feeding future weather
"""
_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class AttentionCell(RNNCell):

    def __init__(self,
                num_units,
                forget_bias=1.0,
                activation=None,
                reuse=None,
                attention=None,
                max_val=None,
                min_val=None,
                name=None):
        super(AttentionCell, self).__init__(_reuse=reuse, name=name)
        self.attention = attention
        self.max_val = max_val
        self.min_val = min_val
        self.margin = self.max_val - self.min_val
        self.margin_rate = self.min_val / self.margin
        self.attention_shape = self.attention.get_shape()
        self.pm25_seed = tf.zeros(shape=[self.attention_shape[0], 1], dtype=tf.float32)
        self.input_spec = base_layer.InputSpec(ndim=2)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation or math_ops.tanh

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                            % inputs_shape)

        input_depth = inputs_shape[1].value
        h_depth = self._num_units
        att_depth = self.attention_shape[1].value
        self._kernel = self.add_variable(
            _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + h_depth + att_depth + 1, 4 * self._num_units])
        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[4 * self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):
        sigmoid = math_ops.sigmoid
        one = constant_op.constant(1, dtype=dtypes.int32)
        # Parameters of gates are concatenated into one multiply for efficiency.
        c, h = state

        gate_inputs = math_ops.matmul(
            array_ops.concat([self.attention, inputs, self.pm25_seed, h], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(
            value=gate_inputs, num_or_size_splits=4, axis=one)

        forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
        # Note that using `add` and `multiply` instead of `+` and `*` gives a
        # performance improvement. So using those at the cost of readability.
        add = math_ops.add
        multiply = math_ops.multiply
        new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                    multiply(sigmoid(i), self._activation(j)))
        new_h = multiply(self._activation(new_c), sigmoid(o))

        new_state = (new_c, new_h)

        with tf.variable_scope("global_attention", initializer=tf.contrib.layers.xavier_initializer()):
            output = tf.layers.dense(new_h, 
                                64,
                                activation=tf.nn.tanh,
                                name="hidden1")
            output = tf.layers.dense(output,
                                    1,
                                    name="hidden2")
        self.pm25_seed = output

        return new_h, new_state