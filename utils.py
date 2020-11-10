import json
import logging
import sys
import numpy as np
import random
from pathlib2 import Path
from shutil import copy
import torch
import math

config = {}


def read_config_file(path='config.json'):
    global config

    with open(path, 'r') as f:
        config.update(json.load(f))


def maybe_cuda(x, is_cuda=None):
    global config

    if is_cuda is None and 'cuda' in config:
        is_cuda = config['cuda']

    if is_cuda:
        return x.cuda()
    return x


def setup_logger(logger_name, filename, delete_old = False):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    stderr_handler = logging.StreamHandler(sys.stderr)
    file_handler   = logging.FileHandler(filename, mode='w') if delete_old else logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    stderr_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stderr_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)
    logger.addHandler(file_handler)
    return logger


def unsort(sort_order):
    result = [-1] * len(sort_order)

    for i, index in enumerate(sort_order):
        result[index] = i

    return result

def get_random_files(count, input_folder, output_folder, specific_section = True):
    files = Path(input_folder).glob('*/*/*/*') if specific_section else Path(input_folder).glob('*/*/*/*/*')
    file_paths = []
    for f in files:
        file_paths.append(f)

    random_paths = random.sample(file_paths, count)

    for random_path in random_paths:
        output_path = Path(output_folder).joinpath(random_path.name)
        copy(str(random_path), str (output_path))


logger = setup_logger(__name__, 'train.log')
def collate_fn(batch):
    batched_data = []
    batched_targets = []
    paths = []

    window_size = 1
    before_sentence_count = int(math.ceil(float(window_size - 1) /2))
    after_sentence_count = window_size - before_sentence_count - 1

    for data, targets, path in batch:
        try:
            max_index = len(data)
            tensored_data = []
            for curr_sentence_index in range(0, len(data)):
                from_index = max([0, curr_sentence_index - before_sentence_count])
                to_index = min([curr_sentence_index + after_sentence_count + 1, max_index])
                sentences_window = [word for sentence in data[from_index:to_index] for word in sentence]
                tensored_data.append(torch.FloatTensor(np.concatenate(sentences_window)))
            tensored_targets = torch.zeros(len(data)).long()
            tensored_targets[torch.LongTensor(targets)] = 1
            tensored_targets = tensored_targets[:-1]
            batched_data.append(tensored_data)
            batched_targets.append(tensored_targets)
            paths.append(path)
        except Exception as e:
            logger.info('Exception "%s" in file: "%s"', e, path)
            logger.debug('Exception!', exc_info=True)
            continue

    return batched_data, batched_targets, paths