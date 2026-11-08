# Phase 3C: frozen E1 official-count decoding

No model, feature, candidate set, probability, MMR coefficient, or learned
parameter changes. The completed E1 word 1–2 gram LR is loaded as-is. The only
repair is replacing the project regex word counter with the public Gist
`count_word` implementation: spaCy `en_core_web_sm` tokens consisting solely
of punctuation or whitespace are excluded.

For each of the frozen 703 development documents, the reference is counted
with that function, candidate sentences are counted with the same function,
and complete-sentence decoding must use no more than the resulting budget.
The run fails on any budget violation. Existing predictions are then evaluated
with isolated `py-rouge==1.1`, `max_n=2`, internal length limiting disabled,
and all other parameters at defaults. Official-test contents remain sealed.
