#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reading utilities for learnable scattering transform."""

import pathlib

import h5py
import logging
import numpy as np
import os
import yaml

from termcolor import colored
from tqdm import trange
from yaml import Loader

# Make tensorflow a little bit quiet.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def parse_arguments(yaml_file):
    """Extract paramters from YAML statements.

    Arguments
    ---------
    yaml_file: str
        Path to the YAML file where the paramters are declared.

    Returns
    -------
    dict
        The YAML file instructions in a dictionnary.
    """
    # Load arguments
    *_, yaml_base = yaml_file.split(os.sep)
    tag, _ = os.path.splitext(yaml_base)
    args = yaml.load(open(yaml_file).read(), Loader=Loader)
    args['summary']['tag'] = os.path.join(tag)
    args['summary']['yaml_file'] = yaml_file

    # Logging level
    set_logging(**args['logging'])
    logging.info('{} (done)'.format(args['summary']['yaml_file']))
    return args


def set_logging(**kwargs):
    """Define logging level.

    Keyword arguments
    -----------------
    kwargs: dict
        Optional arguments passed to :func:`~logging.basicConfig():.
    """
    kwargs.setdefault('level', 'INFO')
    kwargs.setdefault('format', '{function_name:s} {message:s}'.format(
        function_name=colored('%(funcName)s:', 'blue'),
        message='%(message)s'))
    logging.basicConfig(**kwargs)
    pass


class Summary():
    """Summary output manager."""

    def __init__(self, args):
        """Create the summary directory.

        Arguments
        ---------
        args: dict
            Summary-specific directives. The ``path`` gives the directory
            where the summary files should be stored. Other arguments
            such as ``tag`` have automatically been generated by the
            :func:`~scatnet.io.parse()` function.
        """
        self.__dict__ = args
        self.path = os.path.join(self.path, self.tag)
        self.mkdir()
        self.epoch = 0
        self.save_scat = None if self.save_scat == 0 else self.save_scat
        pass

    def mkdir(self, path=None):
        """Make directory, and clean it if already exsit."""
        if path is None:
            path = self.path
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info('{} (created)'.format(path))
        else:
            for file in os.listdir(path):
                os.remove(os.path.join(path, file))
            logging.info('{} (cleaned)'.format(path))
        pass

    def save_args(self, file_name='args.yaml'):
        """Duplicate yaml arguments in the summary directory."""
        file_args = os.path.join(self.path, file_name)
        os.popen('cp {} {}'.format(self.yaml_file, file_args))
        logging.info('{} (done)'.format(file_args))
        pass

    def save_times(self, times, files=['scatterings', 'clusters']):
        """Save time vector for scattering coefficients."""
        for base_name in files:
            file_name = os.path.join(self.path, base_name + '.h5')
            with h5py.File(file_name, 'a') as file:
                file.create_dataset('time', data=times)

    def save_scatterings(self, s):
        """Save full scattering coefficients."""
        if self.save_scat is not None:
            if (self.epoch % self.save_scat) == 0:
                base_file = 'scat_{:05d}.npy'.format(self.epoch)
                file_name = os.path.join(self.path, base_file)
                np.save(file_name, s)
                logging.info('{} (done)'.format(file_name))
        pass

    def save_clusters(self, features, gmm, pca):
        """Save clustering results."""
        base_file = 'clusters_{:05d}'.format(self.epoch)
        file_name = os.path.join(self.path, base_file)
        np.savez(file_name,
                 indexes=gmm.predict(features),
                 projections=features,
                 centroids=gmm.means_.astype(np.float32),
                 covariances=gmm.covariances_.astype(np.float32),
                 eigenvalues=pca.explained_variance_)
        logging.debug('{} (done)'.format(file_name))
        pass

    def save_hot(self, features, gmm, pca, dtype=np.float32):
        """Save clustering results."""
        file_name = os.path.join(self.path, 'clusters.h5')
        with h5py.File(file_name, 'a') as file:
            g = file.create_group('epoch_{:05d}'.format(self.epoch))
            g.create_dataset('hot', data=gmm.predict(features))
            g.create_dataset('features', data=features)
            g.create_dataset('means', data=gmm.means_.astype(dtype))
            g.create_dataset('covariance', data=gmm.covariances_.astype(dtype))
            g.create_dataset('eigenvalues', data=pca.explained_variance_)
        logging.debug('{} (done)'.format(file_name))
        pass

    def save_full(self, s):
        """Save full scattering coefficients."""
        file_name = os.path.join(self.path, 'scatterings.h5')
        if self.save_scat is not None:
            if (self.epoch % self.save_scat) == 0:
                with h5py.File(file_name, 'a') as file:
                    dataset = 'epoch_{:05d}'.format(self.epoch)
                    file.create_dataset(dataset, data=s)
                    logging.info('{} (done)'.format(file_name))
        pass

    def save_scalar(self, base_name, value):
        """Save clustering results."""
        base_file = '{}.txt'.format(base_name)
        file_name = os.path.join(self.path, base_file)
        with open(file_name, 'a') as file:
            file.write('{}\n'.format(value))
        pass

    def save_wavelets(self, parameters, hilbert=True):
        """Save the parameters that control the wavelet shape."""
        file_name = os.path.join(self.path, 'wavelets.h5')
        with h5py.File(file_name, 'a') as file:
            g = file.create_group('epoch_{:05d}'.format(self.epoch))
            for i, p in enumerate(parameters):
                gi = g.create_group('layer_{}'.format(i))
                gi.create_dataset('real_values', data=p[0])
                gi.create_dataset('imag_values', data=p[1])
                gi.create_dataset('scale', data=p[2])
                gi.create_dataset('knots', data=p[3])
            pass

    def save_wavelet(self, parameters, hilbert=True):
        """Save the parameters that control the wavelet shape."""
        # Parameters
        for i, p in enumerate(parameters):

            # Wavelet files
            base_file = 'wavelet_{}.txt'.format(i)
            file_name = os.path.join(self.path, base_file)
            if hilbert is False:
                with open(file_name, 'a') as f:
                    [f.write('{},'.format(m)) for m in p[0][0]]
                    [f.write('{},'.format(m)) for m in p[0][1]]
                    [f.write('{},'.format(m)) for m in p[1][0]]
                    [f.write('{},'.format(m)) for m in p[1][1]]
                    f.write('\n')
            else:
                with open(file_name, 'a') as f:
                    [f.write('{},'.format(m)) for m in p[0]]
                    [f.write('{},'.format(m)) for m in p[1]]
                    f.write('\n')

            # Scale file
            base_file = 'scale_{}.txt'.format(i)
            file_name = os.path.join(self.path, base_file)
            with open(file_name, 'a') as f:
                [f.write('{},'.format(m)) for m in p[2]]
                f.write('\n')

            # Knot file
            base_file = 'knots_{}.txt'.format(i)
            file_name = os.path.join(self.path, base_file)
            with open(file_name, 'a') as f:
                [f.write('{},'.format(m)) for m in p[3]]
                f.write('\n')

        pass

    def trange(self, n_batches, desc='Epoch {}/{}'):
        """Batch progress bar."""
        tqdm_kw = dict(
            desc=colored(desc.format(self.epoch, self.epochs - 1), 'blue'),
            ascii=True
        )
        return trange(n_batches, **tqdm_kw)

    def watch(self, epoch, epochs=None):
        """Set current epoch."""
        self.epoch = epoch
        self.epochs = 1 if epochs is None else epochs
        pass

    # @changefilepath
    def save_graph(self, layers, sampling_rate=1, file_name='arch.yaml'):
        """Write time and frequency properties of the graph.

        Keyword arguments
        -----------------
        sampling_rate: float, optional
            Default to 1. Used to calculate the time scales of each layer.
        file_name: str, optional
            Name of the YAML file where to store the graph properties.
        """
        file_name = os.path.join(self.path, file_name)
        # change current path to upper folder twice for the corret relative path to file_name
        os.chdir(str(pathlib.Path(__file__).parent.parent.absolute()))

        # Headers
        header = '# Time and frequency properties of the graph.\n'
        header += '# Authors: Randall Balestriero and Leonard Seydoux\n'
        header += '# Email: leonard.seydoux@uiv-grenoble-alpes.fr\n\n'
        with open(file_name, 'w') as file:
            file.write(header)

        # Layer 0 (input data)
        out = dict(modulus_0=dict())
        out['modulus_0']['batch_size'] = layers[0].shape_input[0]
        out['modulus_0']['channels'] = layers[0].shape_input[1]
        out['modulus_0']['patch_shape'] = layers[0].shape_input[2]
        out['modulus_0']['sampling_rate'] = round(sampling_rate, 2)
        out['modulus_0']['sampling_period'] = round(1 / sampling_rate, 2)
        to_write = yaml.dump(out, default_flow_style=False) + '\n'

        # Layers 1+
        parent_rate = sampling_rate
        for l, layer in enumerate(layers):
            key = 'modulus_{}'.format(l + 1)
            out = {key: dict()}
            u_shape = layer.u.get_shape().as_list()
            sampling = sampling_rate * u_shape[-1] / layers[0].shape_input[-1]
            out[key] = dict()
            out[key]['batch_size'] = u_shape[0]
            out[key]['channels'] = int(np.prod(u_shape[1:-1]))
            out[key]['patch_shape'] = u_shape[-1]
            out[key]['sampling_rate'] = round(sampling, 2)
            out[key]['sampling_period'] = round(1 / sampling, 2)
            out[key]['largest_period'] = round(2 ** layer.j / parent_rate, 2)
            to_write += yaml.dump(out, default_flow_style=False) + '\n'
            parent_rate = sampling

        # Write the file.
        with open(file_name, 'a') as file:
            file.write('# Modulus layers properties\n')
            file.write('# {}\n'.format(77 * '-'))
            file.write('# modulus_0 stands for input data\n')
            file.write('# Durations are in seconds, and frequencies in Hz\n\n')
            file.write(to_write)

        # Scattering layers (all same properties)
        key = 'scattering_layers'
        out = {key: dict()}
        s_shape = layer.s.get_shape().as_list()
        sampling = sampling_rate * s_shape[-1] / layers[0].shape_input[-1]
        out[key] = dict()
        out[key]['patch_shape'] = s_shape[-1]
        out[key]['sampling_rate'] = round(sampling, 2)
        out[key]['sampling_period'] = round(1 / sampling, 2)
        to_write = yaml.dump(out, default_flow_style=False) + '\n'

        # Write
        with open(file_name, 'a') as file:
            file.write('# Scattering layers properties\n')
            file.write('# {}\n'.format(77 * '-'))
            file.write('# Because of the concatenation, the scattering ')
            file.write('layers have an equal sampling.\n')
            file.write('# The number of coefficients is given for ')
            file.write('the corresponding modulus layers.\n\n')
            file.write(to_write)

        logging.info('{} (done)'.format(file_name))
        pass
