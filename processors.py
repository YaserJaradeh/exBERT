from typing import Tuple, List
from transformers import BertTokenizerFast
import os
import sys
import csv
import random
from datasets import CustomDataset


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

    def __init__(self, tokenizer: str):
        self.labels = set()
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer)

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