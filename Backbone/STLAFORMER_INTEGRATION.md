# STLAformer integration contract

- Venue: Transportation Research Part C, 2026, Volume 182, 105423.
- Official repository: https://github.com/laibajiu/STLAformer
- Pinned commit: `f3eba25e2c63023a4834fd87d88da1c3347f16a9`.
- Vendor model class: `model/STLformer.py::STLformer`.
- TraffiDent adapter does not modify vendor source.
- History/horizon stay H=P=12.
- Input channel 0: canonical normalized speed.
- Input channel 1: canonical fractional time-of-week `(DOW*288 + TOD)/2016`.
- No future time features, adjacency, labels, validation outcomes, or test outcomes enter the model input.
- Native constructor defaults retained for representation dimensions: input embedding 32, TOW embedding 32, FFN 256, 4 heads, 6 layers, dropout 0.3. The paper reports batch size 16, Adam, 200 epochs, initial LR 0.001, and ReduceLROnPlateau with factor 0.1 / patience 20.
- PatchSTG remains the downstream frozen reference by explicit project policy. STLAformer is an additional recent TRC numerical comparison.

## Training-loss note

The accessible STLAformer paper settings specify the optimizer, initial learning rate, epoch budget, plateau decay, batch size, embedding dimension, attention heads, and layer depth, but do not explicitly specify the training loss in the inspected settings section. The unified Backbone wrapper therefore keeps the project-common Huber training loss; this is a training-wrapper choice and does not alter the pinned STLAformer architecture.
