# Source and license notices

MOSAIC-specific software is distributed under the MIT license in `LICENSE`.
This grant does not replace the terms of third-party software, model weights,
or input datasets. In particular, it does not assign a new license to the CSV data.

## PromoSim / RecAgent

The simulator, agents, memory organization, recommender interfaces, and utility
code in `mosaic/promosim/` derive from RecAgent (now YuLan-Rec):
https://github.com/RUC-GSAI/YuLan-Rec

Upstream license: MIT, copyright 2023 Lei Wang. The full notice is included in
`mosaic/promosim/third_party/LICENSE.RecAgent` and in built distributions.
The license was verified at upstream commit `fff935570079502d5a17509b0baf590becd1eaab` on 2026-09-08.
This is the license verification revision, not a claim that this repository
was originally forked at that revision; the original fork revision is not recorded.
Local changes include behavioral acceptance measurement, structured watch events,
paired initialization, bounded requests, content caching, and dataset generation.

## Neural baselines

Native PyTorch ports of the ToupleGDD and S2V-DQN reference architectures are
in `baselines/algorithms/_official_neural.py`. Their source revision, checkpoint
names and MIT notice are recorded in `baselines/third_party/NOTICE.md` and
`baselines/third_party/LICENSE.ToupleGDD`. Checkpoint binaries are not included.

## Research methods and external models

CASO uses flow matching and graph prediction; relevant method references are
recorded in the source. Model packages such as BGE-M3 and Qwen are obtained
separately under their own terms. They are not redistributed in this release.
The input CSV provenance and limitations are recorded in `data/SOURCES.md`.
