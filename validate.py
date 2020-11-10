from tqdm import tqdm
import torch
from tensorboard_logger import log_value
import torch.nn.functional as F
from utils import maybe_cuda
from evaluation_utils import *
preds_stats = predictions_analysis()
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