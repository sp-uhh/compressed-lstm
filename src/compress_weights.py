import numpy as np

def get_projection_rank(sigma, threshold):
    """
    :param sigma: Singular values (SV) from singular value decomposition (SVD) ordered from large SV to small SV
    :param threshold: Percentage of SV taken into account based on energy of SV
    :return: Rank of projection matrix
    """
    k = 0
    while (np.sum(sigma[:k] ** 2) / np.sum(sigma ** 2) < threshold):
        k += 1
    return k


def get_projection_matrices(weight_matrix, threshold):
    """
    Applies SVD to the `weight_matrix` and returns truncated `projection_matrix` and `back_projection_matrix` based on
    the provided `threshold`.
    :param weight_matrix: weight_matrix to be compressed
    :param threshold: compression threshold
    :return: [projection_matrix, back_projection_matrix] for provided weight_matrix
    """
    u, s, vh = np.linalg.svd(weight_matrix, full_matrices=True)
    rank = get_projection_rank(s, threshold)
    ut = u[:, :rank]
    st = np.diag(s[:rank])
    vht = vh[:rank, :]
    projection_matrix = ut @ st
    back_projection_matrix = vht
    return [projection_matrix, back_projection_matrix]


def compress_weights(weights_to_be_compressed, compression_threshold):
    """
    Compresses the weights of LSTM layers, which then can be passed to the CLSTM Layers.
    :param weights_to_be_compressed: List of weights for each LSTM layer to be compressed provide as
    [kernel, recurrent_kernel, bias]. The last element of the list should be the weights of the layer that follows the
    last LSTM Layer starting with the kernel weights.
    :param compression_threshold: compression level
    :return: Returns list of compressed weights kernel_back_projection, projection_matrix, recurrent_kernel, bias
    """
    number_of_layers = len(weights_to_be_compressed)
    bias_list = []
    kernel_list = []
    recurrent_kernel_list = []

    kernel_back_projection_list = []
    projection_matrix_list = []
    recurrent_kernel_back_projection_list = []

    compressed_weight_list = []

    # Reading weights from weights_to_be_compressed by layer
    for layer in range(number_of_layers-1):
        kernel_list.append(weights_to_be_compressed[layer][0])
        recurrent_kernel_list.append(weights_to_be_compressed[layer][1])
        bias_list.append(weights_to_be_compressed[layer][2])
    # Append kernel of following layer
    kernel_list.append(weights_to_be_compressed[number_of_layers-1][0])

    # Compressing weights in LSTM layers
    for layer in range(number_of_layers-1):
        if layer == 0:
            kernel_back_projection_list.append(kernel_list[0])
        else:
            kernel_back_projection, _, _, _ = \
                np.linalg.lstsq(projection_matrix_list[layer - 1], kernel_list[layer], rcond=None)
            kernel_back_projection_list.append(kernel_back_projection)
        projection_matrix, recurrent_kernel_back_projection_matrix = \
            get_projection_matrices(recurrent_kernel_list[layer], compression_threshold)

        projection_matrix_list.append(projection_matrix)
        recurrent_kernel_back_projection_list.append(recurrent_kernel_back_projection_matrix)

        compressed_weight_list.append(
            [kernel_back_projection_list[layer],
             projection_matrix_list[layer],
             recurrent_kernel_back_projection_list[layer],
             bias_list[layer]]
        )

    # Compressing kernel of the layer following the LSTM to be compressed
    following_layer_kernel_back_projection, _, _, _ = \
        np.linalg.lstsq(projection_matrix_list[number_of_layers-2], kernel_list[number_of_layers-1], rcond=None)
    following_layer_weights = [following_layer_kernel_back_projection]
    following_layer_weights.extend(weights_to_be_compressed[number_of_layers-1][1:])
    compressed_weight_list.append(following_layer_weights)

    return compressed_weight_list


