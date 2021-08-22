import sys
import os
import time

path = sys.path[0]
sys.path.append(path.replace('quora_question_pairs', ''))

import tensorflow as tf
import mogrifier
from mogrifier import MogrifierLSTM

file_tsv = 'quora_question_pairs/data/quora_duplicate_questions.tsv'
dataset = tf.data.experimental.CsvDataset(file_tsv, record_defaults = [tf.int32, tf.int32, tf.int32, tf.constant("NOTASENTENCE", tf.string), tf.constant("NOTASENTENCE", tf.string), tf.int32], header = True, field_delim= '\t')

strTime = time.time()
dataset = dataset.map(lambda id, qid1, qid2, question1, question2, is_duplicate: (question1, question2, is_duplicate))
totTime = time.time() - strTime

print(f"Time To Remove First Few Rows from Dataset: {int(totTime // 60)}:{int(totTime % 60)}")

vocab_size = 200000

textVec1 = tf.keras.layers.TextVectorization(
    max_tokens = vocab_size,
    output_sequence_length = 64,
    output_mode = 'int'
)

textVec2 = tf.keras.layers.TextVectorization(
    max_tokens = vocab_size,
    output_sequence_length = 64,
    output_mode = 'int'
)


strTime = time.time()
textVec1.adapt(dataset.map(lambda q1, q2, is_duplicate: q1))
totTime = time.time() - strTime
print(f"Time To Adapt TextVec1: {int(totTime // 60)}:{int(totTime % 60)}")

strTime = time.time()
textVec2.adapt(dataset.map(lambda q1, q2, is_duplicate: q2))
totTime = time.time() - strTime
print(f"Time To Adapt TextVec1: {int(totTime // 60)}:{int(totTime % 60)}")

batch_size = 1024
shuffle_buffer = 400000

dataset = dataset.shuffle(buffer_size = shuffle_buffer).batch(batch_size, drop_remainder = True).prefetch(tf.data.experimental.AUTOTUNE)

print("Built Dataset for execution")

class ParaphraseDetector(tf.keras.Model):
  def __init__(self, vocab_size = 100000, textVec1 = None, textVec2 = None, embed_dim = 300, dimWeightMatrix = 16, dimHiddenState = 16, dimQK = 16, numMogrifyRounds = 5, units=32, return_sequences = False, state_size = tf.TensorShape([None, None])):

    super(ParaphraseDetector, self).__init__()

    self.textVec1 = textVec1
    self.embedding1 = tf.keras.layers.Embedding(vocab_size, embed_dim)
    self.forward_layer1 = MogrifierLSTM(dimWeightMatrix = dimWeightMatrix, dimHiddenState = dimHiddenState, dimQK = dimQK, numMogrifyRounds = numMogrifyRounds, units = units, return_sequences = return_sequences, state_size = state_size)
    self.backward_layer1 = MogrifierLSTM(dimWeightMatrix = dimWeightMatrix, dimHiddenState = dimHiddenState, dimQK = dimQK, numMogrifyRounds = numMogrifyRounds, units = units, return_sequences = return_sequences, go_backwards = True, state_size = state_size)
    self.bidirectional1 = tf.keras.layers.Bidirectional(layer = self.forward_layer1, backward_layer = self.backward_layer1, merge_mode = 'sum')

    self.textVec2 = textVec2
    self.embedding2 = tf.keras.layers.Embedding(vocab_size, embed_dim)
    self.forward_layer2 = MogrifierLSTM(dimWeightMatrix = dimWeightMatrix, dimHiddenState = dimHiddenState, dimQK = dimQK, numMogrifyRounds = numMogrifyRounds, units = units, return_sequences = return_sequences, state_size = state_size)
    self.backward_layer2 = MogrifierLSTM(dimWeightMatrix = dimWeightMatrix, dimHiddenState = dimHiddenState, dimQK = dimQK, numMogrifyRounds = numMogrifyRounds, units = units, return_sequences = return_sequences, go_backwards = True, state_size = state_size)
    self.bidirectional2 = tf.keras.layers.Bidirectional(layer = self.forward_layer2, backward_layer = self.backward_layer2, merge_mode = 'sum')

    self.flat = tf.keras.layers.Flatten()
    self.concat = tf.keras.layers.Concatenate()
    self.dense1 = tf.keras.layers.Dense(100, activation = 'sigmoid', kernel_initializer= 'random_normal')

    self.dense_output = tf.keras.layers.Dense(1, activation = 'softmax', kernel_initializer= 'random_normal')

  def call(self, q1, q2):

    vec1 = self.textVec1(q1)
    vec2 = self.textVec1(q2)

    embed1 = self.embedding1(vec1)
    embed2 = self.embedding2(vec2)

    rnn1 = self.bidirectional1(embed1)
    rnn2 = self.bidirectional2(embed2)

    flat1 = self.flat(rnn1)
    flat2 = self.flat(rnn2)

    concat = self.concat([flat1, flat2])

    return self.dense_output(self.dense1(concat))

dimHiddenState = 16

detector = ParaphraseDetector(textVec1 = textVec1, textVec2 = textVec2, units = 8, vocab_size = vocab_size, embed_dim = 200, dimHiddenState = dimHiddenState, state_size = tf.TensorShape([batch_size, dimHiddenState]))
print("Built model")

epochs = 20

loss = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
acc = tf.keras.metrics.Accuracy()

with tf.device('/GPU:0'):
    strTime = time.time()
    for epoch in range(epochs):
      print(f"<----------------------------- STARTING EPOCH {epoch} ----------------------------->\n\n")

      for step, data in enumerate(dataset):
        q1 = data[0]
        q2 = data[1]
        val = data[2]

        with tf.GradientTape() as tape:
          op = detector(q1, q2)
          loss_val = loss(val, op)

        grads = tape.gradient(loss_val, detector.trainable_weights)
        optimizer.apply_gradients(zip(grads, detector.trainable_weights))
        acc_val = acc(val, op)

        if (step % 500 == 0):
            print(f"Step: {step}, loss: {loss_val}, acc: {acc_val}")

    totTime = time.time() - strTime

    print(f"Time To Finish Epoch {epoch}: {int(totTime // 60)}:{int(totTime % 60)}\n\n")

tf.save_model.save(detector, export_dir = '/quora_question_pairs/model')
