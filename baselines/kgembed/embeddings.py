from pykeen.triples import TriplesFactory
from pykeen.models.predict import get_head_prediction_df, get_relation_prediction_df
from pykeen.models.predict import get_tail_prediction_df, get_all_prediction_df
from pykeen.pipeline import pipeline
import torch

PATH = '../../data/ORKG21'
entity_label_to_id = {}
entity_id_to_label = {}
relation_label_to_id = {}
relation_id_to_label = {}

with open(f'{PATH}/entity2text.txt', 'r') as f:
    f.readline()  # discard count
    for i, line in enumerate(f.readlines()):
        parts = line.strip().split('\t')
        entity_label_to_id[str(parts[1])] = parts[0]
        entity_id_to_label[parts[0]] = str(parts[1])
with open(f'{PATH}/relation2text.txt', 'r') as f:
    f.readline()  # discard count
    for i, line in enumerate(f.readlines()):
        parts = line.strip().split('\t')
        relation_label_to_id[str(parts[1])] = parts[0]
        relation_id_to_label[parts[0]] = str(parts[1])

tf = TriplesFactory.from_path('./data/orkg.tsv')
training, testing = tf.split(ratios=[0.7, 0.3], method='cleanup')


def load_model(path: str):
    return torch.load(path)  # , map_location=torch.device('cpu')


def train_model():
    pipeline_result = pipeline(
        training=training,
        testing=testing,
        # validation=validating,
        model='TransE',
        # optimizer='SGD',
        # optimizer_kwargs=dict(lr=0.01),
        training_loop='slcwa',
        training_kwargs=dict(num_epochs=200, batch_size=128),
        # model_kwargs=dict(embedding_dim=120),
        # negative_sampler='basic',
        # negative_sampler_kwargs=dict(num_negs_per_pos=1),
        # evaluator='RankBasedEvaluator',
        # evaluator_kwargs=dict(ks=[1, 5, 10])
    )
    # save the model
    pipeline_result.save_to_directory('test/orkg_model')
    return pipeline_result.model


def predict_tail(model, head, relation):
    if head not in entity_label_to_id:
        raise ValueError("Head is not known! should be from the graph")
    if relation not in relation_label_to_id:
        raise ValueError("relation is not known! should be from the graph")
    df = get_tail_prediction_df(model, entity_label_to_id[head], relation_label_to_id[relation])
    return [entity_id_to_label[entity_id] for entity_id in df.tail_label[:10]]  # .sort_values('score', ascending=False)


def predict_relation(model, head, tail):
    if head not in entity_label_to_id and tail not in entity_label_to_id:
        raise ValueError("Head & tail should be from the graph")
    df = get_relation_prediction_df(model, entity_label_to_id[head], entity_label_to_id[tail])
    return [relation_id_to_label[relation_id] for relation_id in df.relation_label[:10]]


def predict_head(model, relation, tail):
    if tail not in entity_label_to_id:
        raise ValueError("tail is not known! should be from the graph")
    if relation not in relation_label_to_id:
        raise ValueError("relation is not known! should be from the graph")
    df = get_head_prediction_df(model, relation_label_to_id[relation], entity_label_to_id[tail])
    return [entity_id_to_label[entity_id] for entity_id in df.head_label[:10]]


def run_simulation(model):
    intro = """
    Welcome to the one and only exBERT
    
    Please choose the task that you would like to do:
    t => tail prediction
    r => relation prediction
    h => head prediction
    :q => to quit the program
    """
    print(intro)
    while True:
        in_str = input()
        if in_str == ':q':
            print('Bye!!')
            break
        if in_str in ['t', 'r', 'h']:
            try:
                print('Please enter the following!')
                if in_str == 't':
                    head = input('Head: ')
                    rel = input('Relation: ')
                    print(predict_tail(model, head, rel))
                    print('=' * 10)
                if in_str == 'r':
                    head = input('Head: ')
                    tail = input('Tail: ')
                    print(predict_relation(model, head, tail))
                    print('=' * 10)
                if in_str == 'h':
                    rel = input('Relation: ')
                    tail = input('Tail: ')
                    print(predict_head(model, rel, tail))
                    print('='*10)
            except ValueError as ex:
                print(f'Uh-oh!! this is bad. Got this error: <<{ex}>>')
        else:
            print('option not known!')
            print('choose again wisely')


if __name__ == '__main__':
    # model = train_model()
    model = load_model('test/orkg_model/trained_model.pkl')
    run_simulation(model)

    # predict_tail(model, 'Contribution Similarity', 'utilizes')
    # predict_relation(model, 'Contribution Similarity', 'TF/iDF')
    # Predict tails
    # predicted_tails_df = get_tail_prediction_df(model, 'Contribution Similarity', 'Utilizes')
    # Predict relations
    # predicted_relations_df = get_relation_prediction_df(model, 'brazil', 'uk')
    # Predict heads
    # predicted_heads_df = get_head_prediction_df(model, 'conferences', 'brazil')
    # Score all triples (memory intensive)
    # predictions_df = get_all_prediction_df(model)
    # Score top K triples
    # top_k_predictions_df = get_all_prediction_df(model, k=150)

