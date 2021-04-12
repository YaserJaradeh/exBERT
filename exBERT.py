import logging
import torch
import os
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from processors import HeadTailPredictionProcessor, RelationPredictionProcessor, TripleClassificationProcessor
from transformers.trainer_utils import IntervalStrategy
import cli

logger = logging.getLogger(__name__)


def write_metrics(metrics: dict, output_dir: str):
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        for key, value in metrics.items():
            f.write(f'{key}:\t\t{value}\n')


def pre_flight_checks(args):
    if args.task.lower() not in ['tc', 'rp', 'htp']:
        raise ValueError('task should be on the defined values. Check help for more details')
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))


def main():
    args = cli.init_cli()
    pre_flight_checks(args)

    if args.task.lower().strip() == 'tc':
        processor_class = TripleClassificationProcessor
    elif args.task.lower().strip() == 'rp':
        processor_class = RelationPredictionProcessor
    else:
        processor_class = HeadTailPredictionProcessor

    kg = processor_class(
        args.custom_model if args.custom_model is not None and os.path.exists(args.custom_model) else args.bert_model,
        args.data_dir,
        args.dataset_cache,
        args.max_seq_length
    )

    logger.info("Initialized KG-Processor")

    training_args = TrainingArguments(
        output_dir=args.output_dir,  # output directory
        num_train_epochs=args.num_train_epochs,  # total number of training epochs
        per_device_train_batch_size=args.train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.eval_batch_size,  # batch size for evaluation
        warmup_ratio=args.warmup_proportion,  # ratio of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=1000,
        learning_rate=args.learning_rate,
        local_rank=args.local_rank,
        seed=args.seed,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        do_train=args.do_train,
        do_eval=args.do_eval,
        no_cuda=args.no_cuda,
        save_strategy=IntervalStrategy.NO,
    )

    logger.info("Created training args")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.custom_model if args.custom_model is not None and os.path.exists(args.custom_model) else args.bert_model)
    logger.info("Loaded model from disk or downloaded it")

    logger.info("Creating dataset objects")
    train_ds, eval_ds, test_ds = kg.create_datasets(args.data_dir)

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_ds,  # training dataset
        eval_dataset=eval_ds,  # evaluation dataset
        compute_metrics=kg.which_metrics()
    )

    logger.info("Initialized trainer object")

    if args.do_train or args.do_eval:
        logger.info("Training")
        trainer.train()
    if not (args.custom_model is not None and os.path.exists(args.custom_model)):
        logger.info("Saving model to disk")
        trainer.save_model(args.output_dir)
    if args.do_predict:
        logger.info("Predicting")
        results = trainer.predict(test_ds)
        print(results.metrics)
        logger.info("Writing metrics to disk")
        write_metrics(results.metrics, args.output_dir)


if __name__ == "__main__":
    main()
