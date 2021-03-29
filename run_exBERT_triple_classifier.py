from typing import Tuple, List
import logging
import argparse
import torch
import csv
import sys
import os
import random
from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification, Trainer, TrainingArguments


logger = logging.getLogger(__name__)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(cell.unicode('utf-8') for cell in line)
                lines.append(line)
            return lines


class KGProcessor(DataProcessor):

    def __init__(self, tokenizer: str):
        self.labels = set()
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer)

    def get_relations(self, data_dir):
        """Gets all labels (relations) in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "relations.txt"), 'r') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip())
        return relations

    def get_labels(self, data_dir):
        """Gets all labels (0, 1) for triples in the knowledge graph."""
        return ["0", "1"]

    def get_entities(self, data_dir):
        """Gets all entities in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "entities.txt"), 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip())
        return entities

    def get_train_triples(self, data_dir):
        """Gets training triples."""
        return self._read_tsv(os.path.join(data_dir, "train.tsv"))

    def get_dev_triples(self, data_dir):
        """Gets validation triples."""
        return self._read_tsv(os.path.join(data_dir, "dev.tsv"))

    def get_test_triples(self, data_dir):
        """Gets test triples."""
        return self._read_tsv(os.path.join(data_dir, "test.tsv"))

    @staticmethod
    def corrupt_head_tail(ent2text, entities, line, lines_str_set, text_a, text_b, text_c):
        rnd = random.random()
        texts = []
        labels = []
        if rnd <= 0.5:
            # corrupting head
            tmp_head = ''
            while True:
                tmp_ent_list = set(entities)
                tmp_ent_list.remove(line[0])
                tmp_ent_list = list(tmp_ent_list)
                tmp_head = random.choice(tmp_ent_list)
                tmp_triple_str = tmp_head + '\t' + line[1] + '\t' + line[2]
                if tmp_triple_str not in lines_str_set:
                    break
            tmp_head_text = ent2text[tmp_head]
            texts.append(f"[CLS] {tmp_head_text} [SEP] {text_b} [SEP] {text_c} [SEP]")
            labels.append(0)
        else:
            # corrupting tail
            tmp_tail = ''
            while True:
                tmp_ent_list = set(entities)
                tmp_ent_list.remove(line[2])
                tmp_ent_list = list(tmp_ent_list)
                tmp_tail = random.choice(tmp_ent_list)
                tmp_triple_str = line[0] + '\t' + line[1] + '\t' + tmp_tail
                if tmp_triple_str not in lines_str_set:
                    break
            tmp_tail_text = ent2text[tmp_tail]
            texts.append(f"[CLS] {text_a} [SEP] {text_b} [SEP] {tmp_tail_text} [SEP]")
            labels.append(0)
        return texts, labels

    def create_datasets(self, data_dir: str) -> Tuple[CustomDataset, CustomDataset, CustomDataset]:
        train_lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        dev_lines = self._read_tsv(os.path.join(data_dir, "dev.tsv"))
        test_lines = self._read_tsv(os.path.join(data_dir, "test.tsv"))

        train_dataset = self.transform_portion_to_dataset(data_dir, train_lines)
        dev_dataset = self.transform_portion_to_dataset(data_dir, dev_lines)
        test_dataset = self.transform_portion_to_dataset(data_dir, test_lines)

        return train_dataset, dev_dataset, test_dataset

    def transform_portion_to_dataset(self, data_dir: str, lines: List) -> CustomDataset:
        ent2text = {}
        with open(os.path.join(data_dir, "entity2text.txt"), 'r', encoding='utf8') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    ent2text[temp[0]] = temp[1]
        entities = list(ent2text.keys())
        rel2text = {}
        with open(os.path.join(data_dir, "relation2text.txt"), 'r', encoding='utf8') as f:
            rel_lines = f.readlines()
            for line in rel_lines:
                temp = line.strip().split('\t')
                rel2text[temp[0]] = temp[1]
        lines_str_set = set(['\t'.join(line) for line in lines])
        texts = []
        labels = []
        for (i, line) in enumerate(lines):
            with_labels = False
            if len(line) > 3:
                triple_label = line[3]
                with_labels = True
                if triple_label == "1":
                    label = 1
                else:
                    label = 0
            head_ent_text = ent2text[line[0]]
            tail_ent_text = ent2text[line[2]]
            relation_text = rel2text[line[1]]
            texts.append(f"[CLS] {head_ent_text} [SEP] {relation_text} [SEP] {tail_ent_text} [SEP]")
            if with_labels:
                labels.append(label)
            else:
                labels.append(1)
                corrupt_texts, corrupt_labels = self.corrupt_head_tail(ent2text, entities, line, lines_str_set,
                                                                       head_ent_text, relation_text, tail_ent_text)
                texts += corrupt_texts
                labels += corrupt_labels
        encodings = self.tokenizer(texts, truncation=True, padding=True)
        ds = CustomDataset(encodings, labels)
        # ds.set_format("torch", column=["input_ids"])
        return ds


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model (check HF for details)")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    kg = KGProcessor(args.bert_model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,  # output directory
        num_train_epochs=args.num_train_epochs,  # total number of training epochs
        per_device_train_batch_size=args.train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.eval_batch_size,  # batch size for evaluation
        warmup_ratio=args.warmup_proportion,  # ratio of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=1000,
        learning_rate=args.learning_rate,
        local_rank=args.local_rank,
        seed=args.seed,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_predict,
        no_cuda=args.no_cuda
    )

    model = BertForSequenceClassification.from_pretrained(args.bert_model)
    train_ds, eval_ds, test_ds = kg.create_datasets(args.data_dir)

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_ds,  # training dataset
        eval_dataset=eval_ds  # evaluation dataset
    )

    trainer.train()
    os.makedirs(args.cache_dir)
    model.save_pretrained(args.cache_dir)
    trainer.predict(test_ds)


if __name__ == "__main__":
    main()
