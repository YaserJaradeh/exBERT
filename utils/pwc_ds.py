from paperswithcode import PapersWithCodeClient
from typing import Union, List
import datetime

client = PapersWithCodeClient()
papers = {}
conferences = {}
proceedings = {}
literals = {}
literal_counter = 1


def method_factory(method_type: str):
    """
    create an injector method into the appropriate dictionary based on method type
    :param method_type: the method type to create
    """
    mapper = {'conference': conferences,
              'proceeding': proceedings,
              'paper': papers
              }
    if method_type not in mapper:
        def insert(value: Union[str, List[str], datetime.date]):
            global literal_counter
            if isinstance(value, str):
                if value in literals:
                    return
                literals[value] = f'/literal_{literal_counter}'
                literal_counter += 1
            elif isinstance(value, datetime.date):
                if str(value) in literals:
                    return
                literals[str(value)] = f'/literal_{literal_counter}'
                literal_counter += 1
            else:
                for list_value in value:
                    if list_value in literals:
                        continue
                    literals[list_value] = f'/literal_{literal_counter}'
                    literal_counter += 1
    else:
        def insert(value: str):
            holder = mapper[method_type]
            if value in holder:
                return
            holder[value] = f'/{method_type}/{value}'
    return insert


if __name__ == '__main__':
    page = 1
    while page is not None:
        paper_list = client.paper_list(page=page, items_per_page=200)
        page = paper_list.next_page
        for paper in paper_list.results:
            method_factory('paper')(paper.id)
            for field, value in paper:
                if value is None:
                    continue
                method_factory(field)(value)
        if page == 3:
            break
