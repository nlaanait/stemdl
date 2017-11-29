"""
Created on 10/8/17.
@author: Numan Laanait.
email: laanaitn@ornl.gov
"""
from collections import OrderedDict
import json

# JSON utility functions


def write_json_network_config(file, layer_keys, layer_params):
    """
    Constructs and OrderedDict object and writes it .json file
    :param file: string, path to file
    :param layer_keys: list(string), names of layers
    :param layer_params: list(dict), dictionary of layer parameters
    :return: None
    """
    assert len(layer_keys) == len(layer_params), '# of layer names and # of layer parameter dictionaries do not match!'
    network_config = OrderedDict(zip(layer_keys, layer_params))
    with open(file, mode='w') as f:
        json.dump(network_config, f, indent="\t")
    print('Wrote %d NN layers to %s' % (len(network_config.keys()), file))


def load_json_network_config(file):
    """
    Reads a .json file and returns and OrderedDict object to be used to construct neural nets.
    :param file: .json file.
    :return: network_config OrderedDict
    """
    with open(file, mode='r') as f:

        def _as_ordered_dict(val):
            return OrderedDict(val)

        def _as_list(val):
            return list(val)

        output = json.load(f, object_hook=_as_ordered_dict, object_pairs_hook=_as_ordered_dict)
        network_config = OrderedDict(output)

    print('Read %d NN layers from %s' % (len(network_config.keys()), file))
    return network_config


def write_json_hyper_params(file, hyper_params):
    """
    Write hyper_parameters to .json file
    :param file: string, path to .json file
    :param hyper_params: dict, hyper-paramters.
    :return: None
    """
    with open(file, mode='w') as f:
        json.dump(hyper_params, f, indent="\t")
    print('Wrote %d hyperparameters to %s' % (len(hyper_params.keys()), file))


def load_json_hyper_params(file):
    """
    Loads hyper_parameters dictionary from .json file
    :param file: string, path to .json file
    :return: dict, hyper-paramters.
    """
    with open(file, mode='r') as f:
        hyper_params = json.load(f)
    print('Read %d hyperparameters from %s' % (len(hyper_params.keys()), file))
    return hyper_params


