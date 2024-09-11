import numpy as np


def quantize(weights, num_bits, lowest=0, highest=0):
    """ 
    Quantizes the complex-valued weights as a whole by separating their real and imaginary parts, 
    quantizing them, and rejoining them.

    Parameters
    ----------
    weights : np.ndarray
        1D array of complex-valued weights.
    num_bits : int
        number of bits in which the full weights will be quantized.
    lowest : float
        smallest value in the quantized spectrum.
    highest : float
        biggest value in the quantized spectrum.
    
    Returns
    -------
    np.ndarray 
        1D array of complex-valued weights.
    """
    if lowest == highest == 0:
        highest = max(weights)
        lowest = min(weights)

    #accounting for the real and imaginary parts
    num_bits /= 2
    scale = (highest - lowest) / (2**num_bits - 1)

    real = np.round(np.clip(np.real(weights), lowest, highest) / scale) * scale
    imaginary = np.round(np.clip(np.imag(weights), lowest, highest) / scale) * scale

    quantized_weights = real + 1j * imaginary
    return quantized_weights


def quantizeShell(network, num_bits, lowest=0, highest=0):
    """ 
    Quantizes the complex-valued weights of a network as a whole by separating 
    their real and imaginary parts, quantizing them, and rejoining them.

    Parameters
    ----------
    network : NeuralNetwork 
        Complex-valued neural network.
    num_bits : int
        number of bits in which the full weights will be quantized.
    lowest : float
        smallest value in the quantized spectrum.
    highest : float
        biggest value in the quantized spectrum.
    
    Returns
    -------
    None.
    """
    weights = getWeightsFlat(network)
    weights = quantize(weights, num_bits, lowest, highest)
    changeWeightsFlat(network, weights)


def getWeightsFlat(network):
    """ 
    Returns all the weights of the network aside from input and output, flattened for proccessing.

    Parameters
    ----------
    network : NeuralNetwork 
        Complex-valued neural network.
    
    Returns
    -------
    np.ndarray 
        1D array of complex-valued weights.
    """

    weights = np.array([])
    for layer in network.layers[1:-1]:
        weights = np.concatenate((weights, layer.weights.flatten()))

    return weights


def changeWeightsFlat(network, weights):
    """ 
    Changes the weights of the network for new ones, the structure of the new and
    old weights should be the same.
    
    Parameters
    ----------
    network : NeuralNetwork 
        Complex-valued neural network.
    weights : np.ndarray
        1D array of complex-valued weights.
    
    Returns
    -------
    None.
    """

    weight_indx = 0
    for i in range(1, np.shape(network.layers[:-1])[0]):
        for j in range(np.shape(network.layers[i].weights)[0]):
            sz = network.layers[i].weights[j].size
            network.layers[i].weights[j] = np.copy(weights[weight_indx:weight_indx+sz])
            weight_indx += sz


def magnitudeBasedPruning(weights, prune_fraction):
    """
    Prune weights based on their magnitude and the pruning fraction.

    Parameters
    ----------
    weights : np.ndarray
        1D array of complex-valued weights.
    prune_fraction : float
        fraction of weights to be pruned, range = [0,1].
    
    Returns
    -------
    np.ndarray 
        1D array of complex-valued weights.
    """

    total = len(weights)
    ammount = prune_fraction * total

    ab_weights = sorted(np.abs(weights))[int(np.floor(ammount)):]
    cutoff = ab_weights[0]

    weights[np.abs(weights) < cutoff] = 0

    return weights


def magnitudeBasedPruningShell(network, prune_fraction):
    """
    Prune network based on their weights' magnitude and the pruning fraction.

    Parameters
    ----------
    network : NeuralNetwork 
        Complex-valued neural network.
    prune_fraction : float
        fraction of weights to be pruned, range = [0,1].
    
    Returns
    -------
    None.
    """
    weights = getWeightsFlat(network)
    weights = magnitudeBasedPruning(weights, prune_fraction)
    changeWeightsFlat(network, weights)


def mixedPruning(weights, prune_fraction, magnitude_factor):
    """
    Prunes a fraction of weights, considering both their magntitude 
    and phase by a factor of importance.
    
    Parameters
    ----------
    weights : np.ndarray
        1D array of complex-valued weights.
    prune_fraction : float
        fraction of weights to be pruned, range = [0,1].
    magnitude_factor : float
        Weight of the magnitude compared to the phase, range = [0,1].
    
    Returns
    -------
    np.ndarray 
        1D array of complex-valued weights.
    """

    magnitude = np.abs(weights)
    phase = np.angle(weights)

    normalized_magnitude = magnitude / np.max(magnitude)
    normalized_phase = np.abs(phase) / np.pi

    weights_score = magnitude_factor * normalized_magnitude + (1 - magnitude_factor) * normalized_phase
    priority_indx = np.argsort(weights_score)

    num_weights_to_prune = int(prune_fraction * weights_score.size)

    prune_mask = np.ones(weights_score.size)
    prune_mask[priority_indx[:num_weights_to_prune]] = 0
    pruned_weights = weights * prune_mask
    
    return pruned_weights


def mixedPruningShell(network, prune_fraction, magnitude_factor):
    """
    Prunes a fraction of a network's weights, considering both their magntitude 
    and phase by a factor of importance.
    
    Parameters
    ----------
    network : NeuralNetwork 
        Complex-valued neural network.
    prune_fraction : float
        fraction of weights to be pruned, range = [0,1].
    magnitude_factor : float
        Weight of the magnitude compared to the phase, range = [0,1].
    
    Returns
    -------
    None.
    """
    weights = getWeightsFlat(network)
    weights = mixedPruning(weights, prune_fraction, magnitude_factor)
    changeWeightsFlat(network, weights)


def thresholdMagntudeBasedPruning(network, prune_fraction, layer_threshold):
    """
    Prune network based on their weights' magnitude, the pruning fraction and a
    threshold for the minimum number of weights in each layer.
    
    Parameters
    ----------
    network : NeuralNetwork 
        Complex-valued neural network.
    prune_fraction : float
        fraction of weights to be pruned, range = [0,1].
    layer_threshold : float
        maximum fraction of weights to be pruned by layer, range = [0,1].
    
    Returns
    -------
    None.
    """

    ammount_weights = sum([np.shape(layer.weights.reshape(-1))[0] for layer in network.layers[1:-1]])

    ammount_pruned = prune_fraction * ammount_weights
    flat_weights = np.array([])
    weights = []

    for layer in network.layers[1:-1]:
        flat_weights = np.concatenate((flat_weights, layer.weights.ravel()), axis=0)
        weights.append(layer.weights.ravel())

    ab_weights = sorted(np.abs(flat_weights))[int(np.floor(ammount_pruned)):]
    cutoff = ab_weights[0]

    pruned_weights = np.array([])
    for i, layer in enumerate(network.layers[1:-1]):
        priority_indx = np.argsort(np.abs(weights[i]))
        ammount_cut = layer.weights[np.abs(layer.weights) < cutoff].size

        max_ammount_cut = int(np.ceil(layer.weights.size * layer_threshold))
        ammount_cut = min(max_ammount_cut, ammount_cut)

        pruned_layer = np.asarray(weights[i])
        pruned_layer[priority_indx[:ammount_cut]] = 0
        pruned_weights = np.concatenate((pruned_weights, pruned_layer), axis = 0)

    changeWeightsFlat(network, pruned_weights)
        