import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

from choiloader import collate_fn
from tqdm import tqdm
from argparse import ArgumentParser
from utils import maybe_cuda
import gensim
import utils
from tensorboard_logger import configure, log_value
import os
import sys
from pathlib2 import Path
from wiki_loader import WikipediaDataSet
import accuracy
import numpy as np
from termcolor import colored
from evaluation_utils import *
import pickle

torch.multiprocessing.set_sharing_strategy('file_system')

preds_stats = utils.predictions_analysis()

# change two: remove the class named by Accuracies to evaluation_utils.py
def softmax(x):
    max_each_row = np.max(x, axis=1, keepdims=True)
    exps = np.exp(x - max_each_row)
    sums = np.sum(exps, axis=1, keepdims=True)
    return exps / sums


def import_model(model_name):
    module = __import__('models.' + model_name, fromlist=['models'])
    return module.create()
def train(model, args, epoch, dataset, logger, optimizer):
    model.train()
    total_loss = float(0)
    with tqdm(desc='Training', total=len(dataset)) as pbar:
        for i, (data, target, paths) in enumerate(dataset):
            if True:
                if i == args.stop_after:
                    break

                pbar.update()
                model.zero_grad()
                output = model(data)


                target_var = torch.tensor(maybe_cuda(torch.cat(target, 0), args.cuda), requires_grad=False)
                loss = model.criterion(output, target_var)
                loss.backward()

                optimizer.step()
                total_loss += loss.item()
                # logger.debug('Batch %s - Train error %7.4f', i, loss.data[0])
                pbar.set_description('Training, loss={:.4}'.format(loss.item()))
            # except Exception as e:
                # logger.info('Exception "%s" in batch %s', e, i)
                # logger.debug('Exception while handling batch with file paths: %s', paths, exc_info=True)
                # pass

    total_loss = total_loss / len(dataset)
    logger.debug('Training Epoch: {}, Loss: {:.4}.'.format(epoch + 1, total_loss))
    log_value('Training Loss', total_loss, epoch + 1)


def validate(model, args, epoch, dataset, logger):
    model.eval()
    with tqdm(desc='Validatinging', total=len(dataset)) as pbar:
        acc = Accuracies()
        for i, (data, target, paths) in enumerate(dataset):
            if True:
                if i == args.stop_after:
                    break
                pbar.update()
                output = model(data)
                output_softmax = F.softmax(output, 1)
                targets_var = torch.tensor(maybe_cuda(torch.cat(target, 0), args.cuda), requires_grad=False)

                output_seg = output.data.cpu().numpy().argmax(axis=1)
                target_seg = targets_var.data.cpu().numpy()
                preds_stats.add(output_seg, target_seg)

                acc.update(output_softmax.data.cpu().numpy(), target, paths)

        # epoch_pk, epoch_windiff, threshold = acc.calc_accuracy()
        # change three: more metrices
        epoch_pk, epoch_windiff, epoch_b, epoch_s, threshold = acc.calc_accuracy()
        logger.info('Validating Epoch: {}, accuracy: {:.4}, Pk: {:.4}, Windiff: {:.4}, B: {:.4}, S: {:.4} ,F1: {:.4} . '.format(epoch + 1,
                                                                                                            preds_stats.get_accuracy(),
                                                                                                            epoch_pk,
                                                                                                            epoch_windiff,
                                                                                                            epoch_b,
                                                                                                            epoch_s,
                                                                                                            preds_stats.get_f1()))
        preds_stats.reset()

        return epoch_pk, threshold


def test(model, args, epoch, dataset, logger, threshold):
    model.eval()
    with tqdm(desc='Testing', total=len(dataset)) as pbar:
        acc = Accuracy() # accuracies class is not needed
        for i, (data, target, paths) in enumerate(dataset):
            if True:
                if i == args.stop_after:
                    break
                pbar.update()
                output = model(data)
                output_softmax = F.softmax(output, 1)
                targets_var = torch.tensor(maybe_cuda(torch.cat(target, 0), args.cuda), requires_grad=False)
                output_seg = output.data.cpu().numpy().argmax(axis=1)
                target_seg = targets_var.data.cpu().numpy()
                preds_stats.add(output_seg, target_seg)

                current_idx = 0

                for k, t in enumerate(target):
                    # change four: record the path of test documents
                    path = paths[k]
                    document_sentence_count = len(t)
                    to_idx = int(current_idx + document_sentence_count)

                    output = ((output_softmax.data.cpu().numpy()[current_idx: to_idx, :])[:, 1] > threshold)
                    h = np.append(output, [1])
                    tt = np.append(t, [1])

                    acc.update(h, tt, path)

                    current_idx = to_idx
                    # acc.update(output_softmax.data.cpu().numpy(), target)

        # epoch_pk, epoch_windiff = acc.calc_accuracy()
        epoch_pk, epoch_windiff, epoch_b, epoch_s = acc.calc_accuracy()
        logger.info('Testing Epoch: {}, accuracy: {:.4}, Pk: {:.4}, Windiff: {:.4}, B: {:.4}, S: {:.4} ,F1: {:.4} . '.format(
                                                                                                    epoch + 1,
                                                                                                    preds_stats.get_accuracy(),
                                                                                                    epoch_pk,
                                                                                                    epoch_windiff,
                                                                                                    epoch_b,
                                                                                                    epoch_s,
                                                                                                    preds_stats.get_f1()))
        preds_stats.reset()
        preds_stats.reset()
        epoch_result = acc.all_test_result
        return epoch_pk, epoch_result


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

    if not args.infer:
        if args.wiki:
            dataset_path = Path(utils.config['wikidataset'])
            # change: set the train parameter
            train_dataset = WikipediaDataSet(dataset_path / 'true_train', word2vec=word2vec, train=True,
                                             high_granularity=args.high_granularity)
            dev_dataset = WikipediaDataSet(dataset_path / 'true_dev', word2vec=word2vec, train=False, high_granularity=args.high_granularity)
            test_dataset = WikipediaDataSet(dataset_path / 'true_test', word2vec=word2vec, train=False,
                                            high_granularity=args.high_granularity)

        else:
            dataset_path = utils.config['choidataset']
            train_dataset = ChoiDataset(dataset_path, word2vec)
            dev_dataset = ChoiDataset(dataset_path, word2vec)
            test_dataset = ChoiDataset(dataset_path, word2vec)

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
            with (checkpoint_path / 'model{:03d}.t7'.format(j)).open('wb') as f:
                torch.save(model, f)

            val_pk, threshold = validate(model, args, j, dev_dl, logger)
            if val_pk < best_val_pk: # if val_pk < best_val_pk we store the result of every document on test!
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
                with (checkpoint_path / "result_{}".format(j) / "best_result").open("wb") as f:
                    pickle.dump(test_result, f)
    else:
        test_dataset = WikipediaDataSet(args.infer, word2vec=word2vec,
                                        high_granularity=args.high_granularity)
        test_dl = DataLoader(test_dataset, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False,
                             num_workers=args.num_workers)
        print(test(model, args, 0, test_dl, logger, 0.4))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', help='Use cuda?', action='store_false')
    parser.add_argument('--test', help='Test mode? (e.g fake word2vec)', action='store_true')
    parser.add_argument('--bs', help='Batch size', type=int, default=32)
    parser.add_argument('--test_bs', help='Batch size', type=int, default=32)
    parser.add_argument('--epochs', help='Number of epochs to run', type=int, default=10)
    parser.add_argument('--model', help='Model to run - will import and run', default="max_sentence_embedding")
    parser.add_argument('--load_from', help='Location of a .t7 model file to load. Training will continue')
    parser.add_argument('--expname', help='Experiment name to appear on tensorboard', default='exp1')
    parser.add_argument('--checkpoint_dir', help='Checkpoint directory', default='checkpoints')
    parser.add_argument('--stop_after', help='Number of batches to stop after', default=None, type=int)
    parser.add_argument('--config', help='Path to config.json', default='config.json')
    parser.add_argument('--wiki', help='Use wikipedia as dataset?', action='store_true')
    parser.add_argument('--num_workers', help='How many workers to use for data loading', type=int, default=0)
    parser.add_argument('--high_granularity', help='Use high granularity for wikipedia dataset segmentation', action='store_true')
    parser.add_argument('--infer', help='inference_dir', type=str)

    main(parser.parse_args())