from typing import Tuple, List, Any, Union
from transformers import AutoTokenizer
import os
import sys
import csv
import random
from datasets import CustomDataset, EfficientDataset
import logging
from file_utils import serialize_data, deserialize_data
import metrics
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


class DataProcessor:

    def get_relations(self) -> List[str]:
        pass

    def get_labels(self) -> List[str]:
        pass

    def get_entities(self) -> List[str]:
        pass

    def which_metrics(self):
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

    def __init__(self, tokenizer: str, data_dir: str, caching_dir: str, max_seq_length):
        self.labels = set()
        self.max_seq_length = max_seq_length
        self.caching_dir = caching_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
        self._setup_internal_fields(data_dir)
        self.data_dir = data_dir

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

    def _formulate_string_from_triple(self, text_a: str, text_b: str, text_c: Union[str, None]) -> str:
        pass

    def get_relations(self) -> List[str]:
        """Gets all labels (relations) in the knowledge graph."""
        with open(os.path.join(self.data_dir, "relations.txt"), 'r') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip())
        return relations

    def get_labels(self) -> List[str]:
        """Gets all labels (0, 1) for triples in the knowledge graph."""
        return ["0", "1"]

    def get_entities(self) -> List[str]:
        """Gets all entities in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(self.data_dir, "entities.txt"), 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip())
        return entities

    def get_train_triples(self):
        """Gets training triples."""
        return self._read_tsv(os.path.join(self.data_dir, "train.tsv"))

    def get_dev_triples(self):
        """Gets validation triples."""
        return self._read_tsv(os.path.join(self.data_dir, "dev.tsv"))

    def get_test_triples(self):
        """Gets test triples."""
        return self._read_tsv(os.path.join(self.data_dir, "test.tsv"))

    def corrupt_head_tail(self, ent2text, entities, line, lines_str_set, text_a, text_b, text_c):
        rnd = random.random()
        texts = []
        labels = []
        if rnd <= 0.5:
            # corrupting head
            while True:
                tmp_ent_list = set(entities)
                tmp_ent_list.remove(line[0])
                tmp_ent_list = list(tmp_ent_list)
                tmp_head = random.choice(tmp_ent_list)
                tmp_triple_str = tmp_head + '\t' + line[1] + '\t' + line[2]
                if tmp_triple_str not in lines_str_set:
                    break
            tmp_head_text = ent2text[tmp_head]
            texts.append(self._formulate_string_from_triple(tmp_head_text, text_b, text_c))
            labels.append(0)
        else:
            # corrupting tail
            while True:
                tmp_ent_list = set(entities)
                tmp_ent_list.remove(line[2])
                tmp_ent_list = list(tmp_ent_list)
                tmp_tail = random.choice(tmp_ent_list)
                tmp_triple_str = line[0] + '\t' + line[1] + '\t' + tmp_tail
                if tmp_triple_str not in lines_str_set:
                    break
            tmp_tail_text = ent2text[tmp_tail]
            texts.append(self._formulate_string_from_triple(text_a, text_b, tmp_tail_text))
            labels.append(0)
        return texts, labels

    def corrupt_all_head_tail(self, ent2text, entities, line, text_a, text_b, text_c):
        texts = []
        # corrupting heads
        head_entities = set(entities)
        head_entities.remove(line[0])
        head_entities = list(head_entities)
        randomizer = random.Random(12)
        for corrupt_head in randomizer.sample(head_entities, 20):
            tmp_head_text = ent2text[corrupt_head]
            texts.append(self._formulate_string_from_triple(tmp_head_text, text_b, text_c))
        # corrupting tails
        tail_entities = set(entities)
        tail_entities.remove(line[2])
        tail_entities = list(tail_entities)
        for corrupt_tail in randomizer.sample(tail_entities, 20):
            tmp_tail_text = ent2text[corrupt_tail]
            texts.append(self._formulate_string_from_triple(text_a, text_b, tmp_tail_text))
        return texts
        # path = os.path.join(self.caching_dir, f'texts-test-{i}.pkl')
        # serialize_data(texts, path)

    def create_datasets(self, data_dir: str) -> Tuple[CustomDataset, CustomDataset, CustomDataset]:
        train_lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        dev_lines = self._read_tsv(os.path.join(data_dir, "dev.tsv"))
        test_lines = self._read_tsv(os.path.join(data_dir, "test.tsv"))

        dev_dataset = self.transform_portion_to_dataset(dev_lines, 'dev')
        test_dataset = self.transform_portion_to_dataset(test_lines, 'test')
        train_dataset = self.transform_portion_to_dataset(train_lines, 'train')

        return train_dataset, dev_dataset, test_dataset

    def transform_portion_to_dataset(self, lines: List, ds_type: str, load_from_pkl: bool = True) -> CustomDataset:
        return self._transform_portion_to_dataset(lines, ds_type, load_from_pkl)

    def _transform_portion_to_dataset(self, lines: List, ds_type: str, load_from_pkl: bool = True, efficient: bool = False) -> CustomDataset:
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
            labels, texts = self.process_lines_into_strings(labels, lines, lines_str_set, texts)
            serialize_data(texts, texts_path)
            serialize_data(labels, labels_path)
        if efficient:
            new_texts, labels, mapper = EfficientDataset.create_short_lists(zip(texts, labels))
            encodings = self.tokenizer(new_texts, truncation=True, padding=True, max_length=self.max_seq_length)
            return EfficientDataset(encodings, labels, texts, mapper)
        else:
            encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_seq_length)
            return CustomDataset(encodings, labels)

    def process_lines_into_strings(self, labels, lines, lines_str_set, texts) -> Tuple[List[str], List[Any]]:
        # See child classes for concert definitions
        pass

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
        texts.append(self._formulate_string_from_triple(head_ent_text, relation_text, tail_ent_text))
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


class TripleClassificationProcessor(KGProcessor):

    def __init__(self, tokenizer: str, data_dir: str, caching_dir: str, max_seq_length: int = None):
        super().__init__(tokenizer, data_dir, caching_dir, max_seq_length)

    def _formulate_string_from_triple(self, text_a: str, text_b: str, text_c: Union[str, None]) -> str:
        return f"[CLS] {text_a} [SEP] {text_b} [SEP] {text_c} [SEP]"

    def process_lines_into_strings(self, labels, lines, lines_str_set, texts) -> Tuple[List[str], List[Any]]:
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
            texts.append(self._formulate_string_from_triple(head_ent_text, relation_text, tail_ent_text))
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

    def which_metrics(self):
        return metrics.tc_compute_metrics


class RelationPredictionProcessor(KGProcessor):

    def __init__(self, tokenizer: str, data_dir: str, caching_dir: str, max_seq_length: int = None):
        super().__init__(tokenizer, data_dir, caching_dir, max_seq_length)

    def _formulate_string_from_triple(self, text_a: str, text_b: str, text_c: Union[str, None]) -> str:
        return f"[CLS] {text_a} [SEP] {text_b} [SEP]"

    def process_lines_into_strings(self, labels, lines, lines_str_set, texts) -> Tuple[List[str], List[Any]]:

        label_map = {label: i for i, label in enumerate(self.get_relations())}

        for (i, line) in enumerate(lines):
            head_ent_text = self.ent2text[line[0]]
            tail_ent_text = self.ent2text[line[2]]
            label = line[1]
            label_id = label_map[label]
            labels.append(label_id)

            texts.append(self._formulate_string_from_triple(head_ent_text, tail_ent_text, None))

        return labels, texts

    def which_metrics(self):
        global all_triples_str_set
        global test_triples
        global label_list
        train_triples = self.get_train_triples()
        dev_triples = self.get_dev_triples()
        test_triples = self.get_test_triples()
        all_triples = train_triples + dev_triples + test_triples

        all_triples_str_set = set()
        for triple in all_triples:
            triple_str = '\t'.join(triple)
            all_triples_str_set.add(triple_str)
        label_list = self.get_relations()

        return metrics.rp_compute_metrics


class HeadTailPredictionProcessor(KGProcessor):

    def __init__(self, tokenizer: str, data_dir: str, caching_dir: str, max_seq_length: int = None):
        super().__init__(tokenizer, data_dir, caching_dir, max_seq_length)
        self.is_training = False
        self.is_testing = False

    def _formulate_string_from_triple(self, text_a: str, text_b: str, text_c: Union[str, None]) -> str:
        return f"[CLS] {text_a} [SEP] {text_b} [SEP] {text_c} [SEP]"

    def transform_portion_to_dataset(self, lines: List, ds_type: str, load_from_pkl: bool = True) -> CustomDataset:
        if ds_type == 'train':
            self.is_training = True
        if ds_type == 'test':
            self.is_testing = True
        return self._transform_portion_to_dataset(lines, ds_type, load_from_pkl)

    def process_lines_into_strings(self, labels, lines, lines_str_set, texts) -> Tuple[List[str], List[Any]]:
        for i, line in tqdm(enumerate(lines)):
            head_ent_text = self.ent2text[line[0]]
            tail_ent_text = self.ent2text[line[2]]
            relation_text = self.rel2text[line[1]]

            label = 1
            labels.append(label)

            texts.append(self._formulate_string_from_triple(head_ent_text, relation_text, tail_ent_text))

            if self.is_training:
                corrupt_texts, corrupt_labels = self.corrupt_head_tail(self.ent2text, self.entities, line,
                                                                       lines_str_set,
                                                                       head_ent_text, relation_text, tail_ent_text)
                texts += corrupt_texts
                labels += corrupt_labels
            if self.is_testing:
                corrupt_texts = self.corrupt_all_head_tail(self.ent2text, self.entities, line,
                                                           head_ent_text, relation_text, tail_ent_text)
                texts += corrupt_texts
                labels += [0] * len(corrupt_texts)
        self.is_training = False
        self.is_testing = False
        return labels, texts

    def which_metrics(self):
        return metrics.htp_compute_metrics

