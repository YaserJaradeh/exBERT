from paperswithcode import PapersWithCodeClient
from typing import Union, List
import datetime

client = PapersWithCodeClient()
papers = {}
conferences = {}
proceedings = {}
literals = {}
relations = {}
literal_counter = 1
triples = []


def method_factory(method_type: str):
    """
    create an injector method into the appropriate dictionary based on method type
    :param method_type: the method type to create
    """
    mapper = {'conference': conferences,
              'proceeding': proceedings,
              'paper': papers,
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


if __name__ == '__main__':
    # method_factory('relation')("is_a")
    page = 1
    while page is not None:
        try:
            paper_list = client.paper_list(page=page)
            page = paper_list.next_page
            for paper in paper_list.results:
                paper_id = method_factory('paper')(paper.id)
                # triples.append((paper.id, "is_a", ""))
                for field, value in paper:
                    if value is None:
                        continue
                    if field == 'id':
                        continue
                    value_id = method_factory(field)(value)
                    property_name = f"has_{field}"
                    relation_id = method_factory('relation')(property_name)
                    triples.append((paper_id, relation_id, value_id))
            print(f'Ganna process page {page} yippee!')
        except:
            print(f'oh-oh {page} is crappy! Skipping!')
            page += 1
        # if page == 3:
        #     break
# Deal with list in the triples (object of has_authors)
