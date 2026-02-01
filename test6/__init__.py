# -*- coding: utf-8 -*-
"""
Experiment 6: In-Domain Training on SMD and Buoy Datasets

This experiment trains models on SMD and Buoy datasets separately,
then evaluates on the same dataset (in-domain performance).

Scripts in this folder:
  - prepare_smd_trainset.py      : Prepare SMD train/val/test splits
  - prepare_buoy_trainset.py     : Prepare Buoy train/val/test splits
  - make_fusion_cache_smd_train.py   : Generate FusionCache for SMD training
  - make_fusion_cache_buoy_train.py  : Generate FusionCache for Buoy training
  - train_fusion_cnn_smd.py      : Train on SMD
  - train_fusion_cnn_buoy.py     : Train on Buoy
  - evaluate_smd_indomain.py     : Evaluate on SMD test set
  - evaluate_buoy_indomain.py    : Evaluate on Buoy test set
  - run_experiment6.py           : Master script to run all steps

Degradation: Only rain and fog (simple, proven effective)
"""
