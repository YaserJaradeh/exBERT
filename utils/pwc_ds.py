from paperswithcode import PapersWithCodeClient
from typing import Union, List, Tuple
import datetime
import os
from sklearn.model_selection import train_test_split
from utils.functions import write_datasets_file, write_stats
import json

client = PapersWithCodeClient()
papers = {}
areas = {}
methods = {}
tasks = {}
datasets = {}
evaluations = {}
conferences = {}
proceedings = {}
literals = {}
relations = {}
evaluation_results = {}
results = {}
metrics = {}
literal_counter = 1
eval_results_counter = 1
triples = []

newline = '\n'  # os.linesep
path = "../datasets/PWC21/"


def method_factory(method_type: str):
    """
    create an injector method into the appropriate dictionary based on method type
    :param method_type: the method type to create
    """
    mapper = {'conference': conferences,
              'proceeding': proceedings,
              'paper': papers,
              'area': areas,
              'method': methods,
              'task': tasks,
              'evaluation': evaluations,
              'dataset': datasets,
              'relation': relations,
              'evaluation_result': evaluation_results,
              'metrics': results,
              'metric': metrics
              }
    if method_type not in mapper:
        def insert(value: Union[int, str, List[str], datetime.date]) -> Union[str, List[str]]:
            global literal_counter
            if isinstance(value, str):
                if value in literals:
                    return literals[value]
                literals[value] = f'/literal_{literal_counter}'
                literal_counter += 1
                item_id = literals[value]
            elif isinstance(value, datetime.date):
                if str(value) in literals:
                    return literals[str(value)]
                literals[str(value)] = f'/literal_{literal_counter}'
                literal_counter += 1
                item_id = literals[str(value)]
            elif isinstance(value, int):
                if str(value) in literals:
                    return literals[str(value)]
                literals[str(value)] = f'/literal_{literal_counter}'
                literal_counter += 1
                item_id = literals[str(value)]
            else:
                item_id = []
                for list_value in value:
                    if list_value in literals:
                        item_id.append(literals[list_value])
                        continue
                    literals[list_value] = f'/literal_{literal_counter}'
                    literal_counter += 1
                    item_id.append(literals[list_value])
            return item_id
    else:
        if method_type == 'metrics':
            def insert(result: dict) -> str:
                global eval_results_counter
                holder = mapper[method_type]
                if json.dumps(result) in holder:
                    return holder[json.dumps(result)]
                for key, value in result.items():
                    holder[json.dumps(result)] = f'/{method_type}/result_{eval_results_counter}'
                    eval_results_counter += 1
                    key_id = method_factory('metric')(key)
                    property_name = "has metric"
                    relation_id = method_factory('relation')(property_name)
                    triples.append((holder[json.dumps(result)], relation_id, key_id))
                    value_id = method_factory('value')(value)
                    property_name = "has value"
                    relation_id = method_factory('relation')(property_name)
                    triples.append((holder[json.dumps(result)], relation_id, value_id))
                return holder[json.dumps(result)]

        else:
            def insert(value: str) -> str:
                holder = mapper[method_type]
                if value in holder:
                    return holder[value]
                value_id = value.strip().replace("\t", "").replace("\n", "").replace(" ", "-")
                holder[value] = f'/{method_type}/{value_id}'
                return holder[value]
    return insert


def fetch_component(component_type: str, max_pages: int, params: str = None):
    page = 1
    while page is not None:
        print(f'Ganna process {component_type}\'s page {page}. Yippee!')
        try:
            if params is None:
                code = f'client.{component_type}_list(page=page)'
            else:
                code = f'client.{component_type}_list({params})'
            component_list = eval(code)
            if 'next_page' in dir(component_list):
                page = component_list.next_page
                iterator = component_list.results
            else:
                page = None
                iterator = component_list
            for component in iterator:
                component_id = method_factory(component_type)(component.id)
                for field, value in component:
                    if value is None or (isinstance(value, str) and len(value) == 0):
                        continue
                    if field == 'id':
                        continue
                    value_id = method_factory(field)(value)
                    property_name = f"has {field}"
                    relation_id = method_factory('relation')(property_name)
                    triples.append((component_id, relation_id, value_id))
        except:
            print(f'Oh-oh {component_type}\'s page <{page}> is crappy! Skipping!')
            if page is not None:
                page += 1
        if page is None or page >= max_pages:
            break


def fetch_papers(max_pages: int = 100):
    fetch_component('paper', max_pages)


def fetch_areas(max_pages: int = 10):
    fetch_component('area', max_pages)


def fetch_conferences(max_pages: int = 100):
    fetch_component('conference', max_pages)


def fetch_methods(max_pages: int = 100):
    fetch_component('method', max_pages)


def fetch_tasks(max_pages: int = 100):
    fetch_component('task', max_pages)


def fetch_datasets(max_pages: int = 100):
    fetch_component('dataset', max_pages)


def fetch_evaluations(max_pages: int = 100):
    fetch_component('evaluation', max_pages)
    for eval_id, _ in evaluations.items():
        fetch_component('evaluation_result', max_pages, f"evaluation_id='{eval_id}'")


def create_dataset_files():
    # Write relations
    with open(os.path.join(path, "relations.txt"), 'w', encoding='utf-8') as relations_f, open(
            os.path.join(path, "relation2id.txt"), 'w', encoding='utf-8') as relations_ids_f, open(
            os.path.join(path, "relation2text.txt"), 'w', encoding='utf-8') as relations_text_f:
        relations_ids_f.write(f'{len(relations)}{newline}')
        for index, (relation_label, relation_id) in enumerate(relations.items()):
            relations_ids_f.write(f'{relation_id}\t{index}{newline}')
            relations_text_f.write(f'{relation_id}\t{relation_label}{newline}')
            relations_f.write(f'{relation_id}{newline}')
    # Write entities
    entities = papers.copy()
    entities.update(areas)
    entities.update(methods)
    entities.update(tasks)
    entities.update(datasets)
    entities.update(evaluations)
    entities.update(conferences)
    entities.update(proceedings)
    entities.update(evaluation_results)
    entities.update(results)
    entities.update(metrics)
    entities.update(literals)
    with open(os.path.join(path, "entities.txt"), 'w', encoding='utf-8') as entities_f, open(
            os.path.join(path, "entity2id.txt"), 'w', encoding='utf-8') as entities_ids_f, open(
            os.path.join(path, "entity2text.txt"), 'w', encoding='utf-8') as entities_text_f:
        entities_ids_f.write(f'{len(entities)}{newline}')
        for index, (entity_label, entity_id) in enumerate(entities.items()):
            clean_entity_label = entity_label.strip().replace('\r\n', '').replace("\t", "").replace("\n", "")
            if len(clean_entity_label) > 0:
                entities_ids_f.write(f'{entity_id}\t{index}{newline}')
                entities_text_f.write(f'{entity_id}\t{clean_entity_label}{newline}')
                entities_f.write(f'{entity_id}{newline}')
    # Pre-process triples
    new_triples = []
    for t in triples:
        if isinstance(t[2], list):
            for item in t[2]:
                new_triples.append((t[0], t[1], item))
        else:
            new_triples.append(t)
    # Split Into train/test/dev
    X_train, X_test = train_test_split(new_triples, test_size=0.2, random_state=1)
    X_train, X_val = train_test_split(X_train, test_size=0.25, random_state=1)
    # Write train/test/dev to files
    write_datasets_file(path, 'train', X_train)
    write_datasets_file(path, 'test', X_test)
    write_datasets_file(path, 'dev', X_val)
    write_stats(path, entities, relations, new_triples)


def create_pwc_dataset(pages: int = 50):
    fetch_evaluations(pages)
    fetch_datasets(pages)
    fetch_tasks(pages)
    fetch_methods(pages)
    fetch_conferences(pages)
    fetch_areas(pages)
    fetch_papers(pages * 10)
    create_dataset_files()


if __name__ == '__main__':
    create_pwc_dataset()
