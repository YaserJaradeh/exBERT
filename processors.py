from typing import Tuple, List, Any
from transformers import AutoTokenizer
import os
import sys
import csv
import random
from datasets import CustomDataset
import logging
import pickle

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def serialize_data(data: Any, path: str):
    pickle.dump(data, open(path, "wb"))


def deserialize_data(path: str):
    return pickle.load(open(path, "rb"))


class DataProcessor:

    def get_relations(self, data_dir) -> List[str]:
        pass

    def get_labels(self) -> List[str]:
        pass

    def get_entities(self, data_dir) -> List[str]:
        pass

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None) -> List:
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

    def __init__(self, tokenizer: str, data_dir: str, caching_dir: str, max_seq_length: int = None):
        self.labels = set()
        self.max_seq_length = max_seq_length
        self.caching_dir = caching_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
        self._setup_internal_fields(data_dir)

    def _setup_internal_fields(self, data_dir: str):
        self.ent2text = {}
        with open(os.path.join(data_dir, "entity2text.txt"), 'r', encoding='utf8') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    self.ent2text[temp[0]] = temp[1]
        self.entities = list(self.ent2text.keys())
        self.rel2text = {}
        with open(os.path.join(data_dir, "relation2text.txt"), 'r', encoding='utf8') as f:
            rel_lines = f.readlines()
            for line in rel_lines:
                temp = line.strip().split('\t')
                self.rel2text[temp[0]] = temp[1]

    def get_relations(self, data_dir) -> List[str]:
        """Gets all labels (relations) in the knowledge graph."""
        with open(os.path.join(data_dir, "relations.txt"), 'r') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip())
        return relations

    def get_labels(self) -> List[str]:
        """Gets all labels (0, 1) for triples in the knowledge graph."""
        return ["0", "1"]

    def get_entities(self, data_dir) -> List[str]:
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

        dev_dataset = self._transform_portion_to_dataset(dev_lines, 'dev')
        test_dataset = self._transform_portion_to_dataset(test_lines, 'test')
        train_dataset = self._transform_portion_to_dataset(train_lines, 'train')

        return train_dataset, dev_dataset, test_dataset

    def _transform_portion_to_dataset(self, lines: List, ds_type: str, load_from_pkl: bool = True) -> CustomDataset:
        texts_path = os.path.join(self.caching_dir, f'texts-{ds_type}.pkl')
        labels_path = os.path.join(self.caching_dir, f'labels-{ds_type}.pkl')
        if load_from_pkl and os.path.exists(texts_path) and os.path.exists(labels_path):
            logger.info("Loading pickle files rather than creating them")
            texts = deserialize_data(texts_path)
            labels = deserialize_data(labels_path)
        else:
            lines_str_set = set(['\t'.join(line) for line in lines])
            texts = []
            labels = []
            logger.info(f"Processing now #{len(lines)} lines")
            for (i, line) in enumerate(lines):
                with_labels = False
                if len(line) > 3:
                    triple_label = line[3]
                    with_labels = True
                    if triple_label == "1":
                        label = 1
                    else:
                        label = 0
                head_ent_text = self.ent2text[line[0]]
                tail_ent_text = self.ent2text[line[2]]
                relation_text = self.rel2text[line[1]]
                texts.append(f"[CLS] {head_ent_text} [SEP] {relation_text} [SEP] {tail_ent_text} [SEP]")
                if with_labels:
                    labels.append(label)
                else:
                    labels.append(1)
                    corrupt_texts, corrupt_labels = self.corrupt_head_tail(self.ent2text, self.entities, line,
                                                                           lines_str_set,
                                                                           head_ent_text, relation_text, tail_ent_text)
                    texts += corrupt_texts
                    labels += corrupt_labels
            serialize_data(texts, texts_path)
            serialize_data(labels, labels_path)
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_seq_length)
        return CustomDataset(encodings, labels)

    def convert_triple_to_text(self, line, lines_str_set):
        labels = []
        texts = []
        with_labels = False
        if len(line) > 3:
            triple_label = line[3]
            with_labels = True
            if triple_label == "1":
                label = 1
            else:
                label = 0
        head_ent_text = self.ent2text[line[0]]
        tail_ent_text = self.ent2text[line[2]]
        relation_text = self.rel2text[line[1]]
        texts.append(f"[CLS] {head_ent_text} [SEP] {relation_text} [SEP] {tail_ent_text} [SEP]")
        if with_labels:
            labels.append(label)
        else:
            labels.append(1)
            corrupt_texts, corrupt_labels = self.corrupt_head_tail(self.ent2text, self.entities, line,
                                                                   lines_str_set,
                                                                   head_ent_text, relation_text, tail_ent_text)
            texts += corrupt_texts
            labels += corrupt_labels
        return labels, texts
