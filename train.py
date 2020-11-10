from tqdm import tqdm
import torch
from tensorboard_logger import log_value
from evaluation_utils import *
from utils import maybe_cuda
preds_stats = predictions_analysis()
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
                pbar.set_description('Training, loss={:.4}'.format(loss.item()))
    total_loss = total_loss / len(dataset)
    logger.debug('Training Epoch: {}, Loss: {:.4}.'.format(epoch + 1, total_loss))
    log_value('Training Loss', total_loss, epoch + 1)