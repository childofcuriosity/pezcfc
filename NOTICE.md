# Attribution and provenance

This repository is a research modification of
[Hard Prompts Made Easy (PEZ)](https://github.com/YuxinWenRick/hard-prompts-made-easy),
originally developed by Yuxin Wen, Neel Jain, John Kirchenbauer, Micah
Goldblum, Jonas Geiping, and Tom Goldstein.

The vendored `open_clip/`, `optim_utils.py`, and execution path are derived
from the PEZ reference implementation under its MIT license. The original
copyright notice is preserved in `LICENSE`.

The codebook-std candidate filter, threshold-sweep experiment, anonymized
result records, analysis scripts, documentation, and tests are modifications
made for this repository. The experiment snapshot is retained in
`reference/optim_utils_experiment.py` so that the published refactor can be
audited against the code used during exploration. Its SHA-256 digest is
`f4f0bf9db1f92d1aee0287226cc0eb70859e32428985e47977606ac89394beb6`.
