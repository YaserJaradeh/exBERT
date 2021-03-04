import os
from typing import List, Tuple

newline = '\n'  # os.linesep


def write_stats(data_path: str, entities, relations, triples):
    with open(os.path.join(data_path, "stats.txt"), 'w', encoding='utf-8') as stats_f:
        stats_f.write(f'entities\t{len(entities)}{newline}')
        stats_f.write(f'relations\t{len(relations)}{newline}')
        stats_f.write(f'triples\t{len(triples)}{newline}')


def convert_ids_to_dict(data_dir: str, file_type: str):
    result = {}
    with open(os.path.join(data_dir, f"{file_type}2id.txt"), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            if index == 0 or len(line.strip()) == 0:
                continue
            parts = line.strip().split('\t')
            result[parts[0]] = parts[1]
    return result


def create_ids_dict(data_dir):
    relations = convert_ids_to_dict(data_dir, 'relation')
    entities = convert_ids_to_dict(data_dir, 'entity')
    return relations, entities


def write_datasets_file(data_dir: str, file_type: str, content: List[Tuple[str, str, str]]):
    relations, entities = create_ids_dict(data_dir)
    with open(os.path.join(data_dir, f"{file_type}.tsv"), 'w', encoding='utf-8') as f:
        for subj, pred, obj in content:
            if subj in entities and obj in entities and pred in relations:
                f.write(f'{subj}\t{pred}\t{obj}{newline}')
    with open(os.path.join(data_dir, f"{file_type}2id.txt"), 'w', encoding='utf-8') as f:
        f.write(f'{len(content)}{newline}')
        for subj, pred, obj in content:
            if subj in entities and obj in entities and pred in relations:
                f.write(f'{entities[subj]}\t{relations[pred]}\t{entities[obj]}{newline}')
