#!/usr/bin/env python

import json
import os
import re
import numpy as np
from google.protobuf import json_format
from itertools import chain
import collections
import argparse
import time
import pickle

import tprism.expl_pb2 as expl_pb2
import tprism.expl_graph as expl_graph


def to_string_goal(goal):
    """
    s=goal.name
    s+="("
    s+=",".join([str(arg) for arg in goal.args])
    s+=")"
    """
    s = goal.name
    s += ","
    s += ",".join([str(arg) for arg in goal.args])
    return s


class Flags(object):
    def __init__(self, args, options):
        self.internal_config = dict()
        self.args = args
        self.flags = {f.key: f.value for f in options.flags}

    def __getattr__(self, k):
        return (
            dict.get(self.internal_config, k)
            or getattr(self.args, k, None)
            or dict.get(self.flags, k)
        )

    def add(self, k, v):
        self.internal_config[k] = v

    def update(self):
        ##
        batch_size = 10
        if self.sgd_minibatch_size == "default":
            pass
        else:
            batch_size = int(self.sgd_minibatch_size)
        self.sgd_minibatch_size = batch_size
        self.add("sgd_minibatch_size", batch_size)
        ##
        if self.max_iterate == "default":
            self.max_iterate = 100
        else:
            self.max_iterate = int(self.max_iterate)
        ##
        self.sgd_learning_rate = float(self.sgd_learning_rate)


def get_goal_dataset(goal_dataset):
    out_idx = []
    for j, goal in enumerate(goal_dataset):
        all_num = goal["dataset"].shape[1]
        all_idx = np.array(list(range(all_num)))
        out_idx.append(all_idx)
    return out_idx


def split_goal_dataset(goal_dataset, valid_ratio=0.1):
    train_idx = []
    valid_idx = []
    for j, goal in enumerate(goal_dataset):
        ph_vars = goal["placeholders"]
        all_num = goal["dataset"].shape[1]
        all_idx = np.array(list(range(all_num)))
        np.random.shuffle(all_idx)
        train_num = int(all_num - valid_ratio * all_num)
        train_idx.append(all_idx[:train_num])
        valid_idx.append(all_idx[train_num:])
    return train_idx, valid_idx


#
# goal_dataset["placeholders"] => ph_vars
# goal_dataset["dataset"]: dataset
# dataset contains indeces: values in the given dataset is coverted into index
def build_goal_dataset(input_data, tensor_provider):
    goal_dataset = []

    def to_index(value, ph_name):
        return tensor_provider.convert_value_to_index(value, ph_name)

    to_index_func = np.vectorize(to_index)
    for d in input_data:
        ph_names = d["placeholders"]
        # TODO: multiple with different placeholders
        ph_vars = [tensor_provider.ph_var[ph_name] for ph_name in ph_names]
        dataset = [None for _ in ph_names]
        goal_data = {"placeholders": ph_vars, "dataset": dataset}
        goal_dataset.append(goal_data)
        for i, ph_name in enumerate(ph_names):
            rec = d["records"]
            if tensor_provider.is_convertable_value(ph_name):
                print("[INFO]", ph_name, "converted!!")
                dataset[i] = to_index_func(rec[:, i], ph_name)
            else:  # goal placeholder
                dataset[i] = rec[:, i]
                print("[WARN] no conversion from values to indices:", ph_name)
                print("goal_placeholder?")
                print(rec.shape)
                print(ph_name)
            print("*")
    for obj in goal_dataset:
        obj["dataset"] = np.array(obj["dataset"])
    return goal_dataset
