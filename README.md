# exBERT

A transformer-based model that performs multiple tasks on scholarly knowledge graphs.

## How to use exBERT

In order to use exBERT you have two options.

Via docker

```bash
$ docker run <some fancy image> # coming soon
```

Or alternatively you can setup the python code locally and run it with the shown commands.

First you need to download all the required packages

```bash
$ pip install -r requirements.txt
```

To run the **KG-BERT** scripts, you need to run the following command

Changing the parameters determines what task to perform on which datasets and what hyperparameters.

An example how to run UMLS with Sci-BERT on link prediction task is:
```bash
python3 exBERT.py --task htp \
     --do_train \
     --do_eval \
     --do_predict \
     --data_dir ./data/UMLS \
     --bert_model allenai/scibert_scivocab_uncased \
     --max_seq_length 15 \
     --train_batch_size 32 \
     --learning_rate 5e-5 \
     --num_train_epochs 8.0 \
     --output_dir ./output_UMLS/ \
     --gradient_accumulation_steps 1 \
     --eval_batch_size 135 \
     --fp16
```


