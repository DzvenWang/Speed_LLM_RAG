# v9.0.0 gradient reset hotfix

This patch fixes `tcrr_stla/v9/training.py::one_epoch`: `optimizer.zero_grad(set_to_none=True)` is now executed at the beginning of each training minibatch, before forward/backward/optimizer.step. Previously gradients silently accumulated across minibatches while parameters stepped every minibatch. The optional cross-failure phase already cleared gradients per minibatch and remains unchanged.

Added a regression test asserting two SGD minibatches both produce the expected independent gradient. No other model architecture, data protocol or checkpoint serialization code changed.

**Action for formal experiments**: stop training jobs running the original affected code; deploy this patch; restart Normal/Transfer/Utility training with a fresh run directory, not an affected checkpoint. Prepared replay/evidence can be reused if their identity hashes still match. Resume checkpoints produced under the incorrect accumulation should not be used for paper-quality results.
