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

To run the **KG-BERT** scripts, you need to run the following commands

* To run on the PWC21 dataset
```bash
$ python3 run_exBERT_triple_classifier.py \
      --do_train \
      --do_eval  \
      --do_predict  \
      --data_dir ./datasets/PWC21  \
      --bert_model bert-base-uncased  \
      --max_seq_length 300  \
      --train_batch_size 32  \
      --learning_rate 5e-5  \
      --num_train_epochs 3.0  \
      --output_dir ./output_PWC21/   \
      --gradient_accumulation_steps 1  \
      --eval_batch_size 512 \
      --fp16
```

## Datasets
- ORKG ✔
- PWC ✔
- MAG 😕
- UMLS ❓

## Baselines
- KG-embeddings 😕
  - TransE
  - TransH
  - ...
- KG-BERT ✔
