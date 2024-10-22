import logging
import os
import sys
import math
from typing import List, Tuple
from forbiddenfruit import curse
from orkg import ORKG
from sklearn.model_selection import train_test_split
from utils.functions import write_datasets_file, write_stats, create_ids_dict

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
sh = logging.StreamHandler(sys.stdout)
log.addHandler(sh)
newline = '\n'  # os.linesep
orkg = ORKG(host="https://www.orkg.org/orkg")
path = "../datasets/ORKG21/"
batch_size = 500


def _clean_string(string: str) -> str:
    return string.strip().replace('\n', ' ').replace('\t', ' ').replace('  ', ' ')


curse(str, "clean", _clean_string)


def _write_relations_file(data_dir: str, relations: List[str]):
    log.info(f"Writing relations file {data_dir}")
    with open(os.path.join(data_dir, "relations.txt"), 'w', encoding='utf-8') as f:
        for relation in relations:
            f.write(f'{relation}{newline}')


def _write_entities_file(data_dir: str, entities: List[str]):
    log.info(f"Writing entities file {data_dir}")
    with open(os.path.join(data_dir, "entities.txt"), 'w', encoding='utf-8') as f:
        for entity in entities:
            f.write(f'{entity}{newline}')


def _write_relation2id_file(data_dir: str, relations: List[Tuple[int, str]]):
    log.info(f"Writing relation2id file {data_dir}")
    with open(os.path.join(data_dir, "relation2id.txt"), 'w', encoding='utf-8') as f:
        f.write(f'{len(relations)}{newline}')
        for index, relation in relations:
            f.write(f'{relation}\t{index}{newline}')


def _write_entity2id_file(data_dir: str, entities: List[Tuple[int, str]]):
    log.info(f"Writing entity2id file {data_dir}")
    with open(os.path.join(data_dir, "entity2id.txt"), 'w', encoding='utf-8') as f:
        f.write(f'{len(entities)}{newline}')
        for index, entity in entities:
            f.write(f'{entity}\t{index}{newline}')


def _write_relation2text_file(data_dir: str, relations: List[Tuple[str, str]]):
    log.info(f"Writing relation2text file {data_dir}")
    with open(os.path.join(data_dir, "relation2text.txt"), 'w', encoding='utf-8') as f:
        for relation_id, relation_label in relations:
            f.write(f'{relation_id}\t{relation_label.clean()}{newline}')


def _write_entity2text_file(data_dir: str, entities: List[Tuple[str, str]]):
    log.info(f"Writing entity2text file {data_dir}")
    with open(os.path.join(data_dir, "entity2text.txt"), 'w', encoding='utf-8') as f:
        for entity_id, entity_label in entities:
            f.write(f'{entity_id}\t{entity_label.clean()}{newline}')


def write_orkg_relations():
    log.info("Fetching predicates from ORKG")
    orkg_predicates = orkg.predicates.get(size=9999999).content
    relations = []
    for orkg_predicate in orkg_predicates:
        relations.append(orkg_predicate['id'])
    _write_relations_file(path, relations)
    _write_relation2id_file(path, list(enumerate(relations)))
    _write_relation2text_file(path, [(p['id'], p['label'].strip().replace('\n', '')) for p in orkg_predicates])


def write_orkg_entities():
    log.info("Fetching resources from ORKG & filtering empty labels")
    orkg_resources = orkg.resources.get(size=9999999).content
    orkg_resources = [r for r in filter(lambda r: len(r['label']) > 0, orkg_resources)]
    log.info("Fetching literals from ORKG & filtering empty labels")
    literals = [l for l in filter(lambda l: len(l['label']) > 0, orkg.literals.get_all().content)]
    orkg_resources += literals
    entities = []
    for orkg_resource in orkg_resources:
        entities.append(orkg_resource['id'])
    _write_entities_file(path, entities)
    _write_entity2id_file(path, list(enumerate(entities)))
    _write_entity2text_file(path, [(r['id'], r['label']) for r in orkg_resources])


def write_orkg_statements():
    log.info("Fetching statements from ORKG")
    orkg_statements = orkg.statements.get(size=100).content
    # ignore literal statements
    log.info("Removing literal statements")
    orkg_statements = filter_out_unwanted_statements(orkg_statements)
    # split into train/test/validation
    log.info("Splitting into train, dev, test")
    X_train, X_test = train_test_split(orkg_statements, test_size=0.2, random_state=1)
    X_train, X_val = train_test_split(X_train, test_size=0.25, random_state=1)
    log.info("Writing datasets")
    write_datasets_file(path, 'train',
                        [(st['subject']['id'], st['predicate']['id'], st['object']['id']) for st in X_train])
    write_datasets_file(path, 'dev',
                        [(st['subject']['id'], st['predicate']['id'], st['object']['id']) for st in X_val])
    write_datasets_file(path, 'test',
                        [(st['subject']['id'], st['predicate']['id'], st['object']['id']) for st in X_test])


def filter_out_unwanted_statements(orkg_statements):
    log.info("Filtering-out statements for classes and predicates in the object/subject position")
    log.info(f"Unfiltered: {len(orkg_statements)}")
    filtered = [x for x in filter(
        lambda st: st["object"]["_class"] != 'class' and st["object"]["_class"] != 'predicate' and st["subject"][
            "_class"] != 'class' and st["subject"]["_class"] != 'predicate', orkg_statements)]
    log.info(f"Filtered: {len(filtered)}")
    return filtered


def _get_n_statements_starting_from_x(n_statements: int, x_page: int = 1):
    result = []
    pages = math.ceil(n_statements / batch_size)
    for page in range(x_page, x_page + pages):
        log.info(f"Fetching statements {page - x_page + 1}/{pages}")
        result += orkg.statements.get(size=min(n_statements - len(result), batch_size), page=page).content
    return result, x_page + pages + 1


def write_orkg_statements_new():
    log.info("Fetching statements from ORKG")
    orkg_statements, new_page = _get_n_statements_starting_from_x(1000 * 1000 * 2)
    orkg_statements = filter_out_unwanted_statements(orkg_statements)
    X_train, X_test = train_test_split(orkg_statements, test_size=0.2, random_state=1)
    X_train, X_val = train_test_split(X_train, test_size=0.25, random_state=1)
    log.info("Writing datasets")
    write_datasets_file(path, 'train',
                        [(st['subject']['id'], st['predicate']['id'], st['object']['id']) for st in X_train])
    write_datasets_file(path, 'test',
                        [(st['subject']['id'], st['predicate']['id'], st['object']['id']) for st in X_test])
    write_datasets_file(path, 'dev',
                        [(st['subject']['id'], st['predicate']['id'], st['object']['id']) for st in X_val])
    relations, entities = create_ids_dict(path)
    write_stats(path, entities, relations, orkg_statements)


def write_orkg_statements_backup():
    log.info("Fetching statements from ORKG")
    orkg_statements, new_page = _get_n_statements_starting_from_x(600, 1)
    orkg_statements = filter_out_unwanted_statements(orkg_statements)
    log.info("Writing datasets")
    write_datasets_file(path, 'train',
                        [(st['subject']['id'], st['predicate']['id'], st['object']['id']) for st in orkg_statements])
    orkg_statements, new_page2 = _get_n_statements_starting_from_x(200, new_page)
    orkg_statements = filter_out_unwanted_statements(orkg_statements)
    write_datasets_file(path, 'test',
                        [(st['subject']['id'], st['predicate']['id'], st['object']['id']) for st in orkg_statements])
    orkg_statements, _ = _get_n_statements_starting_from_x(200, new_page2)
    orkg_statements = filter_out_unwanted_statements(orkg_statements)
    write_datasets_file(path, 'dev',
                        [(st['subject']['id'], st['predicate']['id'], st['object']['id']) for st in orkg_statements])


if __name__ == '__main__':
    write_orkg_relations()
    write_orkg_entities()
    write_orkg_statements_new()
