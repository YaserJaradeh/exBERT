from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import torch


def tc_compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    simple_accuracy = (preds == labels).mean()
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'simple_accuracy': simple_accuracy
    }


def rp_compute_metrics(pred):
    global test_triples
    global all_triples_str_set
    global label_list

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    simple_accuracy = (preds == labels).mean()

    ranks = []
    filter_ranks = []
    hits = []
    hits_filter = []
    for i in range(10):
        hits.append([])
        hits_filter.append([])

    for i, pred in enumerate(preds):
        rel_values = torch.tensor(pred)
        _, argsort1 = torch.sort(rel_values, descending=True)
        argsort1 = argsort1.cpu().numpy()

        rank = np.where(argsort1 == labels[i])[0][0]
        # print(argsort1, all_label_ids[i], rank)
        ranks.append(rank + 1)
        test_triple = test_triples[i]
        filter_rank = rank
        for tmp_label_id in argsort1[:rank]:
            tmp_label = label_list[tmp_label_id]
            tmp_triple = [test_triple[0], tmp_label, test_triple[2]]
            # print(tmp_triple)
            tmp_triple_str = '\t'.join(tmp_triple)
            if tmp_triple_str in all_triples_str_set:
                filter_rank -= 1
        filter_ranks.append(filter_rank + 1)

        for hits_level in range(10):
            if rank <= hits_level:
                hits[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)

            if filter_rank <= hits_level:
                hits_filter[hits_level].append(1.0)
            else:
                hits_filter[hits_level].append(0.0)

    metrics_with_values = {
        'raw_mean_rank': np.mean(ranks),
        'filtered_mean_rank': np.mean(filter_ranks),
        'simple_accuracy': simple_accuracy
    }

    for i in [0, 2, 9]:
        metrics_with_values[f'raw_hits @{i + 1}'] = np.mean(hits[i])
        metrics_with_values[f'hits_filter Hits @{i + 1}'] = np.mean(hits_filter[i])

    return metrics_with_values


def htp_compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    simple_accuracy = (preds == labels).mean()
    ranks = []
    ranks_left = []
    ranks_right = []
    hits_left = []
    hits_right = []
    hits = []
    top_ten_hit_count = 0

    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    for triple_id in range(0, len(labels), 41):
        preds = pred.predictions[triple_id:triple_id+41, 1]
        rel_values = torch.tensor(preds)
        _, argsort1 = torch.sort(rel_values, descending=True)
        argsort1 = argsort1.cpu().numpy()
        rank1 = np.where(argsort1 == 0)[0][0]
        ranks.append(rank1 + 1)
        ranks_left.append(rank1 + 1)
        if rank1 < 10:
            top_ten_hit_count += 1
        rel_values = torch.tensor(preds)
        _, argsort1 = torch.sort(rel_values, descending=True)
        argsort1 = argsort1.cpu().numpy()
        rank2 = np.where(argsort1 == 0)[0][0]
        ranks.append(rank2 + 1)
        ranks_right.append(rank2 + 1)
        if rank2 < 10:
            top_ten_hit_count += 1
        for hits_level in range(10):
            if rank1 <= hits_level:
                hits[hits_level].append(1.0)
                hits_left[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_left[hits_level].append(0.0)

            if rank2 <= hits_level:
                hits[hits_level].append(1.0)
                hits_right[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_right[hits_level].append(0.0)
    metrics_with_values = {
        'simple_accuracy': simple_accuracy,
    }
    for i in [0, 2, 9]:
        metrics_with_values[f'hits_left_@{i+1}'] = np.mean(hits_left[i])
        metrics_with_values[f'hits_right_@{i + 1}'] = np.mean(hits_right[i])
        metrics_with_values[f'hits_@{i + 1}'] = np.mean(hits[i])
    metrics_with_values[f'mean_rank_left'] = np.mean(ranks_left)
    metrics_with_values[f'mean_rank_right'] = np.mean(ranks_right)
    metrics_with_values[f'mean_rank'] = np.mean(ranks)
    metrics_with_values['mean_reciprocal_rank_left'] = np.mean(1. / np.array(ranks_left))
    metrics_with_values['mean_reciprocal_rank_right'] = np.mean(1. / np.array(ranks_right))
    metrics_with_values['mean_reciprocal_rank'] = np.mean(1. / np.array(ranks))

    return metrics_with_values

