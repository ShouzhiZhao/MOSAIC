# ToupleGDD Reference Implementations and Checkpoints

`s2vdqn.ckpt` and `touplegdd.ckpt` are the pretrained checkpoints published
in the official ToupleGDD repository:

- Source: https://github.com/Dtrycode/ToupleGDD
- Reference commit: `8ee2620c7c938ce7740698ef3247bcc43428e4fb`
- License: MIT; see `LICENSE.ToupleGDD`

The PyTorch implementations in `algorithms/_official_neural.py` follow these
reference architectures. Checkpoint binaries are distributed separately and
loaded without altering their tensors. `touplegdd.ckpt` refers to the source
repository's `tripling.ckpt`, renamed to make its role explicit in this project.
