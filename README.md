# Source-Specific Backdoor Attack (Demon SCAn)
# PAPER
Demon in the Variant Statistical Analysis of DNNs for Robust Backdoor, Usenix Security, 2021.
# DEPENDENCIES
Our code is tested on Python 3.6.8 based on pytorch.
# How to use
## TaCT
We implemented our TaCT on one datasets: CIFAR10. The logic is as simple as inject mislabeled trigger-carrying images (poisoned images) together with correctly labeled trigger-carrying images (cover images) into the training set.
## SCAn
SCAn was implemented in SCAn.py, needed to replace global and local features, it can get abnormal scores of all classes.
