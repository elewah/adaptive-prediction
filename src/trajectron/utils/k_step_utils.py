"""Utilities for k-step adaptation experiments.

This module provides helper functions to collect sufficient statistics
for multiple prediction steps and to apply sequential updates to the
model's Bayesian last layer.

The functions follow the pseudocode suggested in user instructions and
are intentionally light-weight so they can be imported directly inside
notebooks without modifying the core library.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


@torch.no_grad()
def compute_kstep_sufficient_stats(model, batch, k_steps: int, mode: str):
    """Collect features and targets for ``k_steps`` ahead.

    Parameters
    ----------
    model: object
        Model with ``encoder`` and ``decoder`` members that expose the
        methods used below.  The exact API depends on the model used in
        the notebooks.
    batch: dict
        Mini-batch containing the current state ``s_t`` and a tensor of
        future ground-truth states ``s_future`` with shape ``[T, ...]``.
    k_steps: int
        Number of future steps to unroll when gathering statistics.
    mode: str
        One of ``"single"``, ``"teacher_forcing"`` or ``"free_rollout"``.

    Returns
    -------
    Tuple[List[Tensor], List[Tensor], List[Tensor]]
        Lists of design matrices ``Phi``, targets ``y`` and predicted
        noise covariances ``Sigma`` for each step.
    """

    state = model.encoder(batch)  # encode once to initialise decoder
    Phi_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []
    Sigma_list: List[torch.Tensor] = []

    s_in = batch["s_t"]
    gt_future = batch["s_future"]
    T = gt_future.shape[0]
    k = min(k_steps, T)

    h = model.decoder.init_hidden(state)

    for tau in range(k):
        # Decoder should provide features and predictive noise estimate
        Phi_tau, Sigma_tau, h_next = model.decoder.features_and_noise(h, s_in)
        y_mean_tau = Phi_tau @ model.last_layer.weight_mean
        y_target_tau = gt_future[tau]

        Phi_list.append(Phi_tau.detach())
        Sigma_list.append(Sigma_tau.detach())
        y_list.append(y_target_tau.detach())

        if mode == "teacher_forcing":
            s_in = y_target_tau
            h = h_next
        elif mode == "free_rollout":
            s_in = y_mean_tau
            h = h_next
        else:  # "single"
            break

    return Phi_list, y_list, Sigma_list


def kstep_lastlayer_update(
    kf,
    Phi_list: List[torch.Tensor],
    y_list: List[torch.Tensor],
    Sigma_list: List[torch.Tensor],
):
    """Sequentially apply Kalman/BLR updates for multiple observations.

    Parameters
    ----------
    kf: object
        Last-layer filter exposing a ``correct`` method taking ``Phi``,
        ``y`` and ``Sigma``.
    Phi_list, y_list, Sigma_list: list of Tensor
        Collected sufficient statistics for each step.
    """

    for Phi, y, Sigma in zip(Phi_list, y_list, Sigma_list):
        kf.correct(Phi, y, Sigma)
