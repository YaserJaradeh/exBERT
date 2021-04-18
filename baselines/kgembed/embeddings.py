from pykeen.triples import TriplesFactory
from pykeen.models.predict import get_head_prediction_df, get_relation_prediction_df
from pykeen.models.predict import get_tail_prediction_df, get_all_prediction_df
from pykeen.pipeline import pipeline
import torch

PATH = '../../data/ORKG21'
entity_to_id = {}
relation_to_id = {}
entity_label_to_id = {}
entity_id_to_label = {}
relation_label_to_id = {}
relation_id_to_label = {}
with open(f'{PATH}/entity2id.txt', 'r') as f:
    f.readline()  # discard count
    for _, line in enumerate(f.readlines()):
        parts = line.strip().split('\t')
        entity_to_id[str(parts[0])] = int(parts[1])
with open(f'{PATH}/relation2id.txt', 'r') as f:
    f.readline()  # discard count
    for _, line in enumerate(f.readlines()):
        parts = line.strip().split('\t')
        relation_to_id[str(parts[0])] = int(parts[1])
training = TriplesFactory.from_path(
    f'{PATH}/train.tsv',
    entity_to_id=entity_to_id,
    relation_to_id=relation_to_id,
    create_inverse_triples=True
)
testing = TriplesFactory.from_path(
    f'{PATH}/test.tsv',
    entity_to_id=entity_to_id,
    relation_to_id=relation_to_id,
    create_inverse_triples=True,
)
validating = TriplesFactory.from_path(
    f'{PATH}/dev.tsv',
    entity_to_id=entity_to_id,
    relation_to_id=relation_to_id,
    create_inverse_triples=True,
)

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


def load_model(path: str):
    return torch.load(path)


def train_model():
    pipeline_result = pipeline(
        training=training,
        testing=testing,
        validation=validating,
        model='ConvE',
        training_kwargs=dict(
            num_epochs=500,
        )
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
    return [entity_id_to_label[entity_id] for entity_id in df.sort_values('score').tail_label[:10]]


if __name__ == '__main__':
    model = train_model()
    # model = load_model('test/orkg_model/trained_model.pkl')
    x = predict_tail(model, 'Contribution Similarity', 'utilizes')
    # Predict tails
    # predicted_tails_df = get_tail_prediction_df(model, 'Contribution Similarity', 'Utilizes')
    # Predict relations
    #predicted_relations_df = get_relation_prediction_df(model, 'brazil', 'uk')
    # Predict heads
    #predicted_heads_df = get_head_prediction_df(model, 'conferences', 'brazil')
    # Score all triples (memory intensive)
    #predictions_df = get_all_prediction_df(model)
    # Score top K triples
    #top_k_predictions_df = get_all_prediction_df(model, k=150)

