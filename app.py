from flask import Flask, make_response

from random import randint
import numpy as np
import tensorflow as tf

import data_utils
import s2s_model
from testchat import *

app = Flask(__name__)

buckets = data_utils.buckets
sess = tf.Session()

def create_model(session, forward_only):
    dtype = tf.float32
    model = s2s_model.S2SModel(
        data_utils.dim,
        data_utils.dim,
        buckets,
        512,
        1.0,
        1,
        5.0,
        64,
        0.0003,
        512,
        forward_only,
        dtype
    )
    return model


class TestBucket(object):
    def __init__(self, sentence):
        self.sentence = sentence

    def random(self):
        return self.sentence, ''


model = create_model(None, True)
model.batch_size = 1


def get_s2s_res(sentence):
    bucket_id = min([
        b for b in range(len(buckets))
        if buckets[b][0] > len(sentence)
    ])
    data, _ = model.get_batch_data(
        {bucket_id: TestBucket(sentence)},
        bucket_id
    )
    encoder_inputs, decoder_inputs, decoder_weights = model.get_batch(
        {bucket_id: TestBucket(sentence)},
        bucket_id,
        data
    )
    _, _, output_logits = model.step(
        sess,
        encoder_inputs,
        decoder_inputs,
        decoder_weights,
        bucket_id,
        True
    )
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    return data_utils.indice_sentence(outputs)


def get_res(msg):
    ret = get_key_res(msg)
    if ret == 0:
        if randint(0, 100) > 50:
            ret = get_s2s_res(msg)
        else:
            ret = get_hehe_res()
    return ret


@app.route('/getResponse/<sentence>', methods=['GET'])
def make_res(sentence):
    ret = get_res(sentence)
    response = make_response(ret)
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


@app.route('/')
def hello_world():
    return 'Hello World!'


sess.run(tf.initialize_all_variables())

ckpt = tf.train.get_checkpoint_state(r"./model/model3")
model.saver.restore(sess, ckpt.model_checkpoint_path)

if __name__ == '__main__':
    app.run(host = '0.0.0.0')
