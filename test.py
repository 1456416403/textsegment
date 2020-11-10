from tqdm import tqdm
import torch
from tensorboard_logger import log_value
from evaluation_utils import *
import torch.nn.functional as F
from utils import maybe_cuda
preds_stats = predictions_analysis()
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
                    path = paths[k]
                    document_sentence_count = len(t)
                    to_idx = int(current_idx + document_sentence_count)
                    output = ((output_softmax.data.cpu().numpy()[current_idx: to_idx, :])[:, 1] > threshold)
                    h = np.append(output, [1])
                    tt = np.append(t, [1])
                    acc.update(h, tt, path)
                    current_idx = to_idx
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
        epoch_result = acc.all_test_result
        return epoch_pk, epoch_result