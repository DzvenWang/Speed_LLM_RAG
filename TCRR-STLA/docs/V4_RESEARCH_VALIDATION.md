# TRC-focused validation matrix for v4

**Protocol**: Net-06 speed+flow first, H=P=12, seed10, s40 masked training,
s20/s40/s60/s90 held-out tests, K=12, up to 50k train/5k validation queries.

## Matched controls

1. Original STLAformer checkpoint, without trainable adapter.
2. S0 adapted STLAformer with full frozen forward and no retrieval.
3. Legacy TCRR v3 trained against a *matching* S0 base (do not interpret a
   mismatched v3 checkpoint as an identical-backbone experiment).
4. TCRR v4 full plus separately trained no-pair-interaction, shuffled-CF,
   no-support, no-utility-rank, no-GainHarm and no-risk ablations.
5. v4 inference interventions on exactly one checkpoint (supportive only).
6. PCFT and non-retrieval baselines with the *same observed inputs, target,
   mask protocol, query indices, and backbone if comparing independent gains*.

## Claims and falsification tests

**C1**: Paired normal/failure responses must outperform equal-capacity unpaired
or shuffled pair controls on MAE/RMSE, particularly at s60/s90. Mere pair
classification cannot substantiate the claimed forecast utility.

**C2**: Predicted gain/harm must materially change Ranking. Evaluate correlation
with realized candidate gain, oracle-headroom ratio, Support OFF, and
identical-capacity utility OFF. C2 must not use test outcomes to choose rank.

**C3**: Report MAE/RMSE, harm rate versus coverage, tail error and calibration
curves. A model whose only improvement arises from rejecting all corrections
has not demonstrated useful retrieval. Threshold is determined on the final
chronological fifth of validation only.

**Efficiency**: Record walltime per epoch, net train samples per second,
max GPU memory, bank cache build time and s0 time. Batch 64/96/128 should be
compared on equal validation MAE and time-to-accuracy, not optimizer steps.

### Suggested performance ambitions (NOT observed results)

- Speed avg MAE: 1.6969 → <= 1.65.
- Flow avg MAE: 18.9606 → <= 18.60.
- RMSE not higher than v3, particularly Flow s90 (v3 MAE 24.2944).
- Stable across seeds 10/20/30, Net-01/02/03/04/06 and missing mechanisms.

The major success criterion is independent, statistically supported
reduction in MAE/RMSE attributable to C1/C2/C3, not the presence of new
neural modules alone. Formal results remain pending.
