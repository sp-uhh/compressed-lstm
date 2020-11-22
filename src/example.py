from compress_weights import compress_weights
from clstm import CLSTM
from keras.models import Model
from keras import layers, Input


# Definition of uncompressed example model
x = Input(shape=[100, 600], batch_shape=[32, 100, 600])
lstm_1 = layers.LSTM(units=600, return_sequences=True)(x)
lstm_2 = layers.LSTM(units=600, return_sequences=True)(lstm_1)
output = layers.TimeDistributed(layers.Dense(units=300), name='dense')(lstm_2)
model = Model(
    inputs=[x],
    outputs=[output],
    name='example_model'
)
model.summary()

# In this example we do not train the example_model.
# In general the compressed LSTMs are only used at inference or for curriculum training
# with the compressed weights of a previously trained model with LSTMs instead of CLSTMs.


# Getting weights from (trained) model
weights = model.get_weights()
# Reorganizing weights by layer
weights_by_layer = [weights[0:3], weights[3:6], weights[6:]]
# Compressing weights
compressed_weights = compress_weights(weights_by_layer, 0.7)


# Definition of compressed example model
clstm_1 = CLSTM(
    units=600,
    rank=compressed_weights[0][1].shape[1],
    return_sequences=True,
    name='clstm_1'
)(x)
clstm_2 = CLSTM(
    units=600,
    rank=compressed_weights[1][1].shape[1],
    return_sequences=True,
    name='clstm_2'
)(clstm_1)
output_2 = layers.TimeDistributed(layers.Dense(units=300), name='cdense')(clstm_2)
compressed_model = Model(
    inputs=[x],
    outputs=[output_2],
    name='compressed_example_model'
)

# Loading compressed weights into compressed example model
clstm_1_layer = compressed_model.get_layer(name='clstm_1')
clstm_1_layer.set_weights(compressed_weights[0])
clstm_2_layer = compressed_model.get_layer(name='clstm_2')
clstm_2_layer.set_weights(compressed_weights[1])
cdense_layer = compressed_model.get_layer(name='cdense')
cdense_layer.set_weights(compressed_weights[2])

compressed_model.summary()


