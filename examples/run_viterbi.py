"""
Standalone demo: run the PyBERT Viterbi (ISI) decoder on a synthetic link.

Builds a ``ViterbiDecoder_ISI`` for a 2-level (NRZ) channel with strong
post-cursor ISI, generates a noisy received waveform, decodes it via maximum
likelihood sequence estimation (MLSE), and compares the error count against a
naive symbol-by-symbol slicer.

Run (from the repository root)::

    PYTHONPATH=src python examples/run_viterbi.py

This example depends only on ``numpy``/``scipy`` -- not on the optional
``pyibisami`` dependency.
"""

import numpy as np

from pybert.models.viterbi import ViterbiDecoder_ISI

rng = np.random.default_rng(42)

# --- Link / decoder parameters ---
L = 2                                # NRZ (2 levels: -1, +1)
N = 2                                # symbols of memory per state
sigma = 0.22                         # Gaussian noise std-dev (V)
pulse_resp = np.array([1.0, 0.8])    # cursor + strong post-cursor ISI tap (V/UI)

dec = ViterbiDecoder_ISI(L=L, N=N, sigma=sigma, pulse_resp_samps=pulse_resp)
expecteds = [dec.expectation(i) for i in range(len(dec.states))]
print(f"States ({len(dec.states)}): {dec.states}")
print(f"Expected observations:     {np.round(expecteds, 3)}")
print(f"State Transition Probability matrix:\n{dec.trans}\n")

# --- Generate a random NRZ symbol stream (-1/+1) ---
n_syms = 5000
levels = np.array([-1.0, 1.0])
tx = rng.choice(levels, size=n_syms)

# --- Pass through the ISI channel + add noise to make the received waveform ---
clean = np.convolve(tx, pulse_resp)[:n_syms]
rx = clean + rng.normal(0.0, sigma, size=n_syms)

# --- Decode ---
state_idxs = dec.decode(list(rx))
# Each decoded state is an N-symbol vector; the current symbol is its last element.
decoded = np.array([dec.states[i][-1] for i in state_idxs])

# Align (decoder output corresponds to the symbol stream) and count errors.
m = min(len(tx), len(decoded))
errs = int(np.sum(tx[:m] != decoded[:m]))

# --- Compare against a naive symbol-by-symbol slicer (threshold at 0) ---
slicer = np.where(rx > 0, 1.0, -1.0)
slicer_errs = int(np.sum(tx[:m] != slicer[:m]))

idx_erros = np.where(tx[:m] != slicer[:m])[0]

print(f"Symbols decoded     : {m}")
print(f"Viterbi MLSE errors : {errs}  (BER ~ {errs / m:.3g})")
print(f"Naive slicer errors : {slicer_errs}  (BER ~ {slicer_errs / m:.3g})")

print("\nFirst 10 errors (index, original symbol, decoded symbol, slicer symbol):")
print("index: ", idx_erros[:10])
print("original symbols: ", tx[idx_erros[:10]])
print("decoded symbols: ", decoded[idx_erros[:10]])
print("slicer symbols:  ", slicer[idx_erros[:10]])


#  [0.  0.  0.5 0.5]
#  [0.5 0.5 0.  0. ]
#  [0.  0.  0.5 0.5]]

# Decoding 5000 samples with trellis depth 2 and 4 states...
# Symbols decoded     : 5000
# Viterbi MLSE errors : 0  (BER ~ 0)
# Naive slicer errors : 452  (BER ~ 0.0904)

# First 10 errors (index, original symbol, decoded symbol, slicer symbol):
# index:  [ 21  71  75  91 101 102 104 113 116 126]
# original symbols:  [-1. -1. -1.  1. -1.  1.  1. -1. -1. -1.]
# decoded symbols:  [-1. -1. -1.  1. -1.  1.  1. -1. -1. -1.]
# slicer symbols:   [ 1.  1.  1. -1.  1. -1. -1.  1.  1.  1.]