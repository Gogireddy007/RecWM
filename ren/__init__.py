"""
REN -- Reflexive Equilibrium Networks.

A real, runnable implementation of the ten inventions described in the
RecWM thesis (https://bhavith-chandra.github.io/RecWM-Thesis/), built to
produce genuinely measured metrics rather than restating the paper's
design targets as results.

Modules
-------
belief_fields        Invention 2 -- B(theta, t, s)
influence_kernel      Invention 1 -- K*(s) = (I - Gamma A(s))^-1 A(s)
equilibrium_engine    Invention 3 -- ATLAS + HYDRA + SEISMOGRAPH
self_awareness        Invention 4
regime_transition     Invention 5
scenario_branching    Invention 6
composition_algebra   Invention 7
counterfactual        Invention 8
adversarial_defense   Invention 9
live_system           Invention 10 -- fusion / integration
"""
