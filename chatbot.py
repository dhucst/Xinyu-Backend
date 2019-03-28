import sys

import numpy as np
import tensorflow as tf

import data_utils
import s2s_model

buckets = data_utils.buckets


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


with tf.Session() as sess:
    model = create_model(sess, True)
    model.batch_size = 1
    sess.run(tf.initialize_all_variables())

    ckpt = tf.train.get_checkpoint_state(r"./model/model3")
    if not ckpt or not ckpt.model_checkpoint_path:
        print('restore model fail')

    print('restore model file %s' % ckpt.model_checkpoint_path)
    print(ckpt.model_checkpoint_path)

    model.saver.restore(sess, ckpt.model_checkpoint_path)
    print("Input 'exit()' to exit test mode!")
    sys.stdout.flush()


    def res(sentence):
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
        ret = data_utils.indice_sentence(outputs)
        return ret
    while True:
        sentence = sys.stdin.readline()
        print(res(sentence))