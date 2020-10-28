from pathlib2 import Path

import torch
from torch.utils.data import Dataset
import numpy as np
import random
from text_manipulation import split_sentences, word_model, extract_sentence_words
import utils
import math

logger = utils.setup_logger(__name__, 'train.log')


def get_choi_files(path):
    all_objects = Path(path).glob('**/*.ref')
    files = [str(p) for p in all_objects if p.is_file()]
    return files

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