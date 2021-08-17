import tensorflow as tf

class MogrifierLayer(tf.keras.layers.Layer):

  def __init__(self, dimWeightMatrix = 32, dimHiddenState = 32, dimQK = 32, numMogrifyRounds = 5, units=32, state_size = tf.TensorShape([128, 128])):
      super(MogrifierLayer, self).__init__()
      self.units = units
      self.numRounds = numMogrifyRounds
      self.state_size = [state_size, state_size]
      self.dimWeight = dimWeightMatrix
      self.dimHidden = dimHiddenState
      self.dimQK = dimQK

  def build(self, input_shape):

      batch_size = input_shape[0]
      embedding = input_shape[1]

      print(f"batch_size: {batch_size}, embedding: {embedding}")

      self.list_wx = []
      self.list_wh = []
      self.list_b = []
      self.list_dense_x = []
      self.list_dense_h = []

      for i in range(4):
        self.list_wx.append(self.add_weight(shape=(batch_size, self.dimWeight),
                               initializer='glorot_uniform',
                               trainable=True))
        self.list_wh.append(self.add_weight(shape=(batch_size, self.dimWeight),
                               initializer='glorot_uniform',
                               trainable=True))
        self.list_b.append(self.add_weight(shape=(embedding,),
                               initializer='glorot_uniform',
                               trainable=True))
        self.list_dense_x.append(tf.keras.layers.Dense(batch_size))
        self.list_dense_h.append(tf.keras.layers.Dense(batch_size))


      self.h_state = self.add_weight(shape = (batch_size, self.dimHidden),
                               initializer = 'glorot_uniform',
                               trainable=True)
      self.hiddenDenseContract = tf.keras.layers.Dense(embedding)
      self.hiddenDenseExpand = tf.keras.layers.Dense(self.dimHidden)

      self.c_state = self.add_weight(shape=(batch_size, self.dimHidden),
                               initializer='glorot_uniform',
                               trainable=True)
      self.cellDenseContract = tf.keras.layers.Dense(embedding)
      self.cellDenseExpand = tf.keras.layers.Dense(self.dimHidden)

      self.qk_list = []
      self.qk_denseContract_list = []
      self.qk_denseExpand_list = []

      for i in range(self.numRounds):
          self.qk_list.append(
              self.add_weight(shape=(batch_size, self.dimQK),
                               initializer='glorot_uniform',
                               trainable=True)
          )

          self.qk_denseContract_list.append(
              tf.keras.layers.Dense(batch_size)
          )


  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return [self.h_state, self.c_state]

  def mogrify(self, x, h, rounds):
    for i in range(rounds):
      if (i%2 == 0):
        h = self.hiddenDenseExpand(tf.math.multiply(2*tf.math.sigmoid(tf.matmul(self.qk_denseContract_list[i](self.qk_list[i]), x)), self.hiddenDenseContract(h)))
      else:
        x = tf.math.multiply(2*tf.math.sigmoid(tf.matmul(self.qk_denseContract_list[i](self.qk_list[i]), self.hiddenDenseContract(h))), x)
    return x, h

  def call(self, inputs, states):

      x_val, h_val = self.mogrify(inputs, states[0], self.numRounds)

      f = tf.math.sigmoid(tf.matmul(self.list_dense_x[0](self.list_wx[0]), x_val)+ tf.matmul(self.list_dense_h[0](self.list_wh[0]), self.hiddenDenseContract(h_val))+ self.list_b[0])
      i = tf.math.sigmoid(tf.matmul(self.list_dense_x[1](self.list_wx[1]), x_val)+ tf.matmul(self.list_dense_h[1](self.list_wh[1]), self.hiddenDenseContract(h_val))+ self.list_b[1])
      j = tf.math.tanh(tf.matmul(self.list_dense_x[2](self.list_wx[2]), x_val)+ tf.matmul(self.list_dense_h[2](self.list_wh[2]), self.hiddenDenseContract(h_val))+ self.list_b[2])
      o = tf.math.sigmoid(tf.matmul(self.list_dense_x[3](self.list_wx[3]), x_val)+ tf.matmul(self.list_dense_h[3](self.list_wh[3]), self.hiddenDenseContract(h_val))+ self.list_b[3])

      c = tf.math.multiply(f, self.cellDenseContract(states[1])) + tf.math.multiply(i, j)
      h = tf.math.multiply(o, tf.math.tanh(c))

      self.h_state = self.hiddenDenseExpand(h)
      self.c_state = self.cellDenseExpand(c)


      return o, (self.h_state, self.c_state)

class MogrifierLSTM(tf.keras.layers.Layer):
  def __init__(self, dimWeightMatrix = 32, dimHiddenState = 32, dimQK = 32, numMogrifyRounds = 5, units=32, state_size = tf.TensorShape([128, 128]), return_sequences = False, return_state = False):
    super(MogrifierLSTM, self).__init__()
    self.lstmCell = MogrifierLayer(dimWeightMatrix = dimWeightMatrix, dimHiddenState = dimHiddenState, dimQK = dimQK, numMogrifyRounds = numMogrifyRounds, units=units, state_size = state_size)
    self.lstm = tf.keras.layers.RNN(self.lstmCell, return_sequences = return_sequences, return_state= return_state)

  def call(self, input):
    return self.lstm(input)
