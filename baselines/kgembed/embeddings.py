from pykeen.models.predict import get_head_prediction_df, get_relation_prediction_df
from pykeen.models.predict import get_tail_prediction_df, get_all_prediction_df
from pykeen.pipeline import pipeline

pipeline_result = pipeline(dataset='Nations', model='RotatE')
model = pipeline_result.model
# Predict tails
predicted_tails_df = get_tail_prediction_df(model, 'brazil', 'intergovorgs')
# Predict relations
predicted_relations_df = get_relation_prediction_df(model, 'brazil', 'uk')
# Predict heads
predicted_heads_df = get_head_prediction_df(model, 'conferences', 'brazil')
# Score all triples (memory intensive)
predictions_df = get_all_prediction_df(model)
# Score top K triples
top_k_predictions_df = get_all_prediction_df(model, k=150)

# save the model
pipeline_result.save_to_directory('doctests/nations_rotate')
