from paperswithcode import PapersWithCodeClient
from typing import Union, List
import datetime
import os

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
literal_counter = 1
triples = []

newline = '\n'  # os.linesep
path = "../datasets/PWC/"


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
              'relation': relations
              }
    if method_type not in mapper:
        def insert(value: Union[str, List[str], datetime.date]) -> Union[str, List[str]]:
            global literal_counter
            item_id = ""
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
        def insert(value: str) -> str:
            holder = mapper[method_type]
            if value in holder:
                return holder[value]
            value_id = value.strip().replace("\t", "").replace("\n", "").replace(" ", "-")
            holder[value] = f'/{method_type}/{value_id}'
            return holder[value]
    return insert


def fetch_component(component_type: str, max_pages: int):
    page = 1
    while page is not None:
        try:
            component_list = eval(f'client.{component_type}_list(page=page)')
            page = component_list.next_page
            for component in component_list.results:
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
            page += 1
        if page is None or page >= max_pages:
            break
        print(f'Ganna process {component_type}\'s page {page}. Yippee!')


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


def create_dataset_files():
    # Write relations
    with open(os.path.join(path, "relations.txt"), 'w', encoding='utf-8') as relations_f, open(os.path.join(path, "relation2id.txt"), 'w', encoding='utf-8') as relations_ids_f, open(os.path.join(path, "relation2text.txt"), 'w', encoding='utf-8') as relations_text_f:
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
    entities.update(literals)
    with open(os.path.join(path, "entities.txt"), 'w', encoding='utf-8') as entities_f, open(os.path.join(path, "entity2id.txt"), 'w', encoding='utf-8') as entities_ids_f, open(os.path.join(path, "entity2text.txt"), 'w', encoding='utf-8') as entities_text_f:
        entities_ids_f.write(f'{len(entities)}{newline}')
        for index, (entity_label, entity_id) in enumerate(entities.items()):
            entities_ids_f.write(f'{entity_id}\t{index}{newline}')
            clean_entity_label = entity_label.strip().replace('\r\n', '').replace("\t", "").replace("\n", "")
            entities_text_f.write(f'{entity_id}\t{clean_entity_label}{newline}')
            entities_f.write(f'{entity_id}{newline}')


if __name__ == '__main__':
    fetch_evaluations(3)
    fetch_datasets(3)
    fetch_tasks(3)
    fetch_methods(3)
    fetch_conferences(3)
    fetch_areas(3)
    fetch_papers(3)
    create_dataset_files()

# Deal with list in the triples (object of has_authors)
