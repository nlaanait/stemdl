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
        json.dump(network_config, f, indent=4)
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
        json.dump(hyper_params, f, indent=4)
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


def load_flags_from_simple_json(file_path, flags, verbose=False):
    image_parms = load_json_hyper_params(file_path)
    for parm_name in image_parms.keys():
        val = image_parms[parm_name]
        if verbose:
            print('\t{}: {}'.format(parm_name, val))
        if isinstance(val, bool):
            dtype = 'boolean'
            func = flags.DEFINE_boolean
        elif isinstance(val, int):
            dtype ='integer'
            func = flags.DEFINE_integer
        elif isinstance(val, float):
            dtype = 'float'
            func = flags.DEFINE_float
        elif type(val) in [str, unicode]:
            dtype = 'string'
            func = flags.DEFINE_string
        else:
            raise NotImplemented('{} : {} of type that we cannot handle now'.format(parm_name, val))
        if verbose:
            print('{} : {} saved as {}'.format(parm_name, val, dtype))
        func(parm_name, val, """""")


def get_dict_from_json(file_path):
    json_dict = load_json_hyper_params(file_path)
    new_dict = dict()
    for key, val in json_dict.items():
        assert isinstance(val, dict)
        new_dict[key] = val['value']
    return new_dict


def load_flags_from_json(file_path, flags, verbose=False):
    image_parms = load_json_hyper_params(file_path)
    for parm_name, parm_values in list(image_parms.items()):
        if verbose:
            print('\t{}: {}'.format(parm_name, parm_values))
        if parm_values['type'] == 'bool':
            func = flags.DEFINE_boolean
        elif parm_values['type'] == 'int':
            func = flags.DEFINE_integer
        elif parm_values['type'] == 'float':
            func = flags.DEFINE_float
        elif parm_values['type'] == 'str':
            func = flags.DEFINE_string
        else:
            raise NotImplemented('Cannot handle type: {} for parameter: {}'.format(parm_values['type'], parm_name))
        if verbose:
            print('{} : {} saved as {} with description: {}'.format(parm_name, parm_values['value'],
                                                                    parm_values['type'], parm_values['desc']))
        func(parm_name, parm_values['value'], parm_values['desc'])
