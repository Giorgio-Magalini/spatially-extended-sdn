# Differentiable Scattering Delay Networks

Fork of Alessandro Ilic Mezza's differentiable Scattering Delay Network repository.

## Goal

Push the model toward a **generalized** formulation that solves **acoustic navigability**: synthesizing physically consistent Room Impulse Responses at arbitrary, unseen source/listener positions from a sparse set of measurements, while keeping the SDN's geometric prior, low cost and real-time viability.

**Most recent step:** an MLP variant that conditions the SDN parameters on geometry (node distances) instead of learning a single globally shared parameter set — adding positional freedom without giving up the physical prior.

## Thesis sub-project — resonance analysis

Master's thesis, Politecnico di Milano (Advisor: Prof. Fabio Antonacci, Co-advisor: Dr. Alessandro Ilic Mezza): *Spatially Distributed Training of Differentiable SDNs for Improved Acoustic Navigability*.

A single-position DSDN fit overfits the resonant behaviour of that point. The thesis introduces **batched multi-microphone training**: one shared parameter set is optimized against many receiver positions at once, minimizing a joint averaged acoustic loss built from three perceptual descriptors — EDC (Energy Decay Curve), EDR (Energy Decay Relief, per mel band) and EDP (Echo Density Profile).

Setup: simulated 4×7×3 m shoebox (pyroomacoustics, ISM order 250), frequency-dependent wall materials, 8×8 grid of 64 mics at 5 cm spacing, checkerboard split of 16 train / 48 held-out test positions.

Result: batched training suppresses spatial resonance variability across the 48 unseen positions while keeping energy decay accurate everywhere; single-mic training degrades substantially on the EDR loss at test positions. Main limitations: parameters are shared globally (a fit to the average geometry), and real-room validation is blocked by source directivity violating the omnidirectional assumption — which is exactly what the geometry-conditioned MLP addresses.
