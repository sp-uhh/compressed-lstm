[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

# Compressed Long Short-Term Memory (CLSTM) Keras Layer

The compressed long short-term memory (CLSTM) Keras layer presented in the repository is  based on LSTMCell and LSTM layer
from Keras version 2.2.4. [[1](#1)] which are compressed using the compression method presented by Prabhavalkar
et al. in [[2](#2)].
The CLSTMs can be used at inference or for curriculum training with compressed weights of a previously trained network
with LSTMs instead of CLSTMs.
The file [src/compress_weights.py](./src/compress_weights.py) contains all required functions to compress the weights of trained LSTM layers, which can then be passed to the CLSTM layers.

In our master thesis [[3](#3)] we explain the CLSTM in detail. Furthermore, we study the effect of replacing the LSTMs with CLSTMs on the model size, run time, and speech separation quality of the Online Deep Attractor Network [[4](#4)] for monaural speech separation.
Our experiments show that the proposed compression method for LSTMs is superior to hyper-parameter tuning in the task of reducing the run time by shrinking a neural network while trying to maintaining the speech separation quality.

## Requirements
```
pip install numpy Keras==2.2.4 tensorflow==1.13.1
```

## Sources
The folder [src](./src) contains three source files:
- [clstm.py](./src/clstm.py) contains the CLSTM layer class which uses the CLSTMCell class.
- [compress_weights.py](./src/compress_weights.py) contains functions to compress the weights of LSTM layers, which then can be passed to the CLSTM layers. The `compress_weights(weights_to_be_compressed, compression_threshold)` function expects the `weights_to_be_compressed` as list of weights by LSTM layer providing a list containing `[kernel, recurrent_kernel, bias]`. The last element of the list should be the weights of the layer that follows the last LSTM layer starting with the kernel weights. Furthermore, the `compression_threshold` (equivalent to the threshold tau in [[2](#2)]) has to be given to the `compress_weights()` function.
- [example.py](./src/example.py) provides an example model with two LSTM layers followed by one dense layer. The weights of this model are saved, compressed, and loaded into the compressed version of the example model in which the LSTM layers are replaced by CLSTM layers.

## Documentation
The [documentation](./docs/README.md) provides some theoretical background on the compressed long short-term memory. For more details see [[3](#3)] and [[2](#2)].


## References

<a id="1">[1]</a>  [https://github.com/keras-team/keras/blob/2.2.4/keras/layers/recurrent.py](https://github.com/keras-team/keras/blob/2.2.4/keras/layers/recurrent.py)

<a id="2">[2]</a> R. Prabhavalkar, O. Alsharif, A. Bruguier, and L. McGraw, “**On the compression of recurrent neural networks with an application to LVCSR acoustic modeling for embedded speech recognition**,” in 2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2016, pp. 5970–5974. [https://ieeexplore.ieee.org/abstract/document/7472823/](https://ieeexplore.ieee.org/abstract/document/7472823/)

<a id="3">[3]</a> M. Siemering, “**Real-time speech separation with deep attractor networks on an embedded system**,” Nov. 2020.

<a id="4">[4]</a> C. Han, Y. Luo, and N. Mesgarani, “**Online Deep Attractor Network for Real-time Single-channel Speech Separation**,” in ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), May 2019, pp. 361–365, doi: 10.1109/ICASSP.2019.8682884. [https://ieeexplore.ieee.org/abstract/document/8682884](https://ieeexplore.ieee.org/abstract/document/8682884)


