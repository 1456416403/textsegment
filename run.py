import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import json
from utils import collate_fn
from argparse import ArgumentParser
from paramters import create_parser
from utils import maybe_cuda
import gensim
import utils
from tensorboard_logger import configure
import os
import sys
from pathlib2 import Path
from wiki_loader import WikipediaDataSet
import numpy as np
from termcolor import colored
import pickle
from train import train
from test import test
from validate import validate

torch.multiprocessing.set_sharing_strategy('file_system')

def softmax(x):
    max_each_row = np.max(x, axis=1, keepdims=True)
    exps = np.exp(x - max_each_row)
    sums = np.sum(exps, axis=1, keepdims=True)
    return exps / sums

def import_model(model_name):
    module = __import__('models.' + model_name, fromlist=['models'])
    return module.create()

def main(args):
    sys.path.append(str(Path(__file__).parent))
    checkpoint_path = Path(args.checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)
    logger = utils.setup_logger(__name__, os.path.join(args.checkpoint_dir, 'train.log'))
    utils.read_config_file(args.config)
    utils.config.update(args.__dict__)
    logger.debug('Running with config %s', utils.config)
    configure(os.path.join('runs', args.expname))
    if not args.test:
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(utils.config['word2vecfile'], binary=True)
    else:
        word2vec = None
    dataset_path = Path(utils.config['wikidataset'])
    train_dataset = WikipediaDataSet(dataset_path / 'true_train', word2vec=word2vec, train=True,
                                     high_granularity=args.high_granularity)
    dev_dataset = WikipediaDataSet(dataset_path / 'true_dev', word2vec=word2vec, train=False, high_granularity=args.high_granularity)
    test_dataset = WikipediaDataSet(dataset_path / 'true_test', word2vec=word2vec, train=False,
                                    high_granularity=args.high_granularity)
    train_dl = DataLoader(train_dataset, batch_size=args.bs, collate_fn=collate_fn, shuffle=True,
                          num_workers=args.num_workers)
    dev_dl = DataLoader(dev_dataset, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False,
                        num_workers=args.num_workers)
    test_dl = DataLoader(test_dataset, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False,
                         num_workers=args.num_workers)
    assert bool(args.model) ^ bool(args.load_from)  # exactly one of them must be set
    if args.model:
        model = import_model(args.model)
    elif args.load_from:
        with open(args.load_from, 'rb') as f:
            model = torch.load(f)
    model.train()
    model = maybe_cuda(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if not args.infer:
        best_val_pk = 1.0
        for j in range(args.epochs):
            train(model, args, j, train_dl, logger, optimizer)
            with (checkpoint_path / 'train_model'/ 'model{:03d}.t7'.format(j)).open('wb') as f:
                torch.save(model, f)
            val_pk, threshold = validate(model, args, j, dev_dl, logger)
            if val_pk < best_val_pk:
                test_pk, test_result = test(model, args, j, test_dl, logger, threshold)
                logger.debug(
                    colored(
                        'Current best model from epoch {} with p_k {} and threshold {}'.format(j, test_pk, threshold),
                        'green'))
                best_val_pk = val_pk
                if not os.path.exists(checkpoint_path/"result_{}".format(j)):
                    os.mkdir(checkpoint_path/"result_{}".format(j))
                with (checkpoint_path / "result_{}".format(j) /'best_model.t7').open('wb') as f:
                    torch.save(model, f)
                with (checkpoint_path / "result_{}".format(j) / "best_result.txt").open("w") as f:
                    print('saved in this path:',checkpoint_path / "result_{}".format(j) / "best_result.txt")
                    f.write("\n".join(map(lambda x: str(x), test_result.items())))
                with open(checkpoint_path / "result_{}".format(j)/ "best_result.json", "w") as f:
                    json.dump(test_result, f)
                with (checkpoint_path / "result_{}".format(j) / "best_result").open("wb") as f:
                    pickle.dump(test_result, f)
    else:
        if not os.path.exists(checkpoint_path / "infer_result"):
            os.mkdir(checkpoint_path / "infer_result")
        test_dataset = WikipediaDataSet(dataset_path / 'true_test', word2vec=word2vec, train=False,
                                        high_granularity=args.high_granularity)
        test_dl = DataLoader(test_dataset, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False,
                             num_workers=args.num_workers)
        test_pk, test_result= test(model, args, 0, test_dl, logger, 0.4)
        with (checkpoint_path / "infer_result"/ "infer_result.txt").open("w") as f:
            print('saved in this path:', checkpoint_path / "infer_result"/ "infer_result.txt")
            f.write("\n".join(map(lambda x: str(x), test_result.items())))
        with open(checkpoint_path / "infer_result"/ "infer_result.json", "w") as f:
            json.dump(test_result, f)
        with (checkpoint_path / "infer_result" / "infer_result").open("wb") as f:
            pickle.dump(test_result, f)

if __name__ == '__main__':
    parser = create_parser()
    main(parser)