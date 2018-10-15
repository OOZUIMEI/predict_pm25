"""
Reference site: https://github.com/philipperemy/tensorflow-fifo-queue-example
"""

import threading
import time

import tensorflow as tf

class DataGenerator(object):
    def __init__(self, coord, types, data_slots, shapes=None, max_queue_size=32, wait_time=0.01)
        self.wait_time = wait_time
        self.max_queue_size = max_queue_size
        self.queue = tf.FIFOQueue(max_queue_size, types, shapes)
        self.queue_size = self.queue.size()
        self.threads = []
        self.coord = coord
        self.enqueue = self.queue.enqueue(data_slots)
    
    def set_data(self, data):
        self.data = data

    def dequeue(self, num_elements):
        output = self.queue.dequeue()
        return output
    
    def thread_main(self, sess):
        stop = False
        end_step = global_step + preload_nums
        while not stop:
            for step in xrange(global_step, end_step):
                index = range(step * cons_b, (step + 1) * cons_b, stride)
                ct_t = ct[index]
                ct_t = np.asarray([range(int(x), int(x) + self.encoder_length) for x in ct_t])
                dec_t = ct_t + self.decoder_length
                q_dc = {
                    self.encoder_inputs: ct_t,
                    self.decoder_inputs: dec_t,
                    self.attention_inputs: ct_t
                }
                session.run(enqueue, feed_dict=q_dc)
