# Concept directions

Controlled pairs estimate `v = mean_i(h_i^+ - h_i^-)`. Evaluate per-pair cosine with `v`, held-out prompt templates, and cross-layer stability. Context-dependent directions can cancel under a mean, while a direction estimated from lexical substitutions can encode the words rather than the concept.

Adding `αv` and observing a systematic output change is stronger than decoding. It still does not establish that unmodified inference naturally travels along `v` or that `v` is the only representation of the concept.

