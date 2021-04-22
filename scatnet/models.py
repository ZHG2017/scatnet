#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Learnable scattering wavelets and scales with kmeans loss minimization.

This version does not learn k-means first, but takes external centroids.
"""
# import sys, os
# import pathlib
# # append path of the script's parent folder to system while launching the script
# parent = os.path.dirname(pathlib.Path(sys.argv[0]).parent.absolute())
# sys.path.append(str(parent))

import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf

# from tensorflow.contrib.distributions import MultivariateNormalFullCovariance
# MVNFC = MultivariateNormalFullCovariance
import tensorflow_probability as tfp
MVNFC = tfp.distributions.MultivariateNormalFullCovariance


def gmm(mu, cov, tau, sx_proj, n_clusters=None, gmm_type='natural',
        trainable=True, cov_diag=None):
    """Gaussian mixture clustering."""
    if trainable is True:
        log_tau = tf.log(tf.nn.softmax(tau))
        eps = tf.eye(2)
        d = cov_diag
        gm = [MVNFC(mu[c], tf.nn.elu(d[c]) * np.eye(2) +
                    tf.matmul(cov[c], tf.transpose(cov[c])) + eps)
              for c in range(n_clusters)]
    else:
        n_clusters = cov.get_shape().as_list()[0]
        log_tau = tf.log(tau)
        gm = [MVNFC(mu[c], cov[c]) for c in range(n_clusters)]
    log_p = [gm[c].log_prob(sx_proj) for c in range(n_clusters)]
    cat = tf.stack([log_tau[c] + log_p[c] for c in range(n_clusters)], 1)

    # Choose you loss
    if gmm_type == 'natural':
        # reduce_logsumexp() Computes log(sum(exp(elements across dimensions of a tensor))).
        # C.f. https://www.tensorflow.org/api_docs/python/tf/math/reduce_logsumexp
        q = tf.reduce_logsumexp(cat, axis=1)
        loss = - tf.reduce_mean(q)
    else:
        # argmax() Returns the index with the largest value across axes of a tensor.
        # C.f. https://www.tensorflow.org/api_docs/python/tf/math/argmax
        y = tf.argmax(cat, axis=1)
        y_un, _ = tf.unique(y)
        q = tf.reduce_sum(tf.stop_gradient(tf.nn.softmax(cat)) * cat, 1)
        loss = tf.map_fn(
            lambda c: tf.reduce_sum(
                q * tf.cast(tf.equal(y, tf.ones_like(y) * c),
                            tf.float32)) /
            tf.reduce_sum(
                tf.cast(tf.equal(y, tf.ones_like(y) * c), tf.float32)),
            y_un, dtype=tf.float32) * 1e3

    return loss, cat
