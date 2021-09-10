from typing import List, Tuple, Dict
from enum import Enum
from tqdm import tqdm


class RelationType(Enum):
    ONE_2_ONE = 1
    ONE_2_N = 2
    N_2_ONE = 3
    N_2_N = 4


def read_aggregated_file(path: str) -> List[Tuple[str]]:
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        return [tuple(line.strip().split('\t')) for line in lines]


def convert_to_enum(N_to: bool, One_to: bool, to_N: bool, to_One: bool) -> RelationType:
    if One_to and to_One:
        return RelationType(1)
    if One_to and to_N:
        return RelationType(2)
    if N_to and to_One:
        return RelationType(3)
    if N_to and to_N:
        return RelationType(4)


def assign_relation_type(relation_type: RelationType, relation_types: Dict, relation: str):
    if relation not in relation_types:
        relation_types[relation] = relation_type
    else:
        if relation_types == RelationType.N_2_N:
            relation_types[relation] = relation_type
        elif relation_types == RelationType.N_2_ONE and relation_types[relation] != RelationType.N_2_N:
            if relation_types[relation] != RelationType.One_2_N:
                relation_types[relation] = RelationType.N_2_N
            else:
                relation_types[relation] = relation_type
        elif relation_types == RelationType.ONE_2_N and relation_types[relation] != RelationType.N_2_N:
            if relation_types[relation] != RelationType.N_2_ONE:
                relation_types[relation] = RelationType.N_2_N
            else:
                relation_types[relation] = relation_type


def print_stats(relation_types):
    print()
    print(relation_types)
    print()
    print('Percentages:')
    print(f'N-2-N: {len(list(filter(lambda v: v == RelationType.N_2_N, relation_types.values()))) / len(relation_types)}')
    print(f'1-2-N: {len(list(filter(lambda v: v == RelationType.ONE_2_N, relation_types.values()))) / len(relation_types)}')
    print(f'N-2-1: {len(list(filter(lambda v: v == RelationType.N_2_ONE, relation_types.values()))) / len(relation_types)}')
    print(f'1-2-1: {len(list(filter(lambda v: v == RelationType.ONE_2_ONE, relation_types.values()))) / len(relation_types)}')


if __name__ == '__main__':
    path = './orkg.tsv'
    triples = read_aggregated_file(path)
    sources = dict()
    destinations = dict()
    relations = dict()
    relation_types = {}

    for source, relation, destination in tqdm(triples, desc='Collection objects'):
        # fill sources
        if source in sources:
            if relation in sources[source]:
                sources[source][relation].append(destination)
            else:
                sources[source][relation] = [destination]
        else:
            sources[source] = {relation: [destination]}
        # fill destinations
        if destination in destinations:
            if relation in destinations[destination]:
                destinations[destination][relation].append(source)
            else:
                destinations[destination][relation] = [source]
        else:
            destinations[destination] = {relation: [source]}
        # fill relations
        if relation not in relations:
            relations[relation] = {'s': [source], 'd': [destination]}
        else:
            relations[relation]['s'].append(source)
            relations[relation]['d'].append(destination)

    N_to = False
    One_to = False
    to_N = False
    to_One = False
    for relation, connections in tqdm(relations.items(), desc='Iterating relations'):
        # for s in tqdm(set(connections['s']), desc=f'Iterating over sources of relation: {relation}'):
        for s in set(connections['s']):
            if len(sources[s][relation]) > 1:
                to_N = True
            elif len(sources[s][relation]) == 1:
                to_One = True
            for d in set(connections['d']):
                if len(destinations[d][relation]) > 1:
                    N_to = True
                elif len(destinations[d][relation]) == 1:
                    One_to = True
                assign_relation_type(convert_to_enum(N_to, One_to, to_N, to_One), relation_types, relation)
                # Stop working stop and continue to next relation
                if relation_types[relation] == RelationType.N_2_N:
                    print(f'Early stopping on relation: {relation}')
                    break
        # print_stats(relation_types)

    print_stats(relation_types)



