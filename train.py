import argparse
from helper import *
from utils_glue import *
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,TensorDataset
from models import CliModel, Gen1Model, Gen2Model, FinModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils_trainer import Trainer
from datasets import load_dataset
from transformers import TrainingArguments
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", default='tmp', type=str, help='jobid to save training logs')
    parser.add_argument("--data_dir", default=None, type=str,help="The root dir of glue data")
    parser.add_argument("--task_name", default=None, type=str, help="The name of the glue task to train.")
    parser.add_argument("--output_dir", default='output', type=str,help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length", default=None, type=int, help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")

    parser.add_argument("--batch_size", default=None, type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=None, type=float, help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', '--wd', default=0.01, type=float, metavar='W', help='weight decay')
    parser.add_argument("--num_train_epochs", default=None, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument('--eval_step', type=int, default=100)
    parser.add_argument('--aug_train', action='store_true',
                        help="Whether using data augmentation or not")

    args = parser.parse_args()
    args.do_lower_case = True
    output_mode = output_modes[task_name]

    log_dir = os.path.join(args.output_dir, 'record_%s.log' % args.job_id)
    init_logging(log_dir)

    # Prepare devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logging.info("device: {} n_gpu: {}".format(device, n_gpu))

    # Prepare seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Prepare task settings
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dataset = load_dataset("ag_news")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    # tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    # tokenizer = AutoTokenizer.from_pretrained("ahmedrachid/FinancialBERT")
    # tokenizer = AutoTokenizer.from_pretrained("climatebert/distilroberta-base-climate-f")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text", "label"])
    tokenized_datasets.set_format("torch")
    train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42)
    # training_args = TrainingArguments(output_dir=args.output_dir,evaluation_strategy="epoch")
    # metric = evaluate.load("accuracy")

    num_train_optimization_steps = 0
    if not args.do_eval:
        # if args.aug_train:
            # train_examples = processor.get_aug_examples(args.data_dir)
        # else:
            # train_examples = processor.get_train_examples(args.data_dir)
        # if args.gradient_accumulation_steps < 1:
            # raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                # args.gradient_accumulation_steps))

        args.batch_size = args.batch_size // args.gradient_accumulation_steps

        # train_features = convert_examples_to_features(train_examples, label_list,
        #                                               args.max_seq_length, tokenizer, output_mode)
        # train_data, _ = get_tensor_data(output_mode, train_features)
        # train_sampler = RandomSampler(train_data)
        # train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

        num_train_optimization_steps = int(
            len(tokenized_datasets) / args.batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    # eval_examples = processor.get_dev_examples(args.data_dir)
    # eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
    # eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
    # eval_sampler = SequentialSampler(eval_data)
    # eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
    # model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base",num_labels=2)
    model = FinModel(output_dim=2, dropout_rate=0.3)
    model.to(device)
    # if n_gpu > 1:
    #    model = torch.nn.DataParallel(model)

    trainer = Trainer(args, device, model,num_train_optimization_steps)

    trainer.build(lr=args.learning_rate)
    trainer.train(train_dataset, args.task_name, output_mode, eval_labels,
                  num_labels, train_dataloader, eval_dataloader, eval_dataset, tokenizer,
                  mm_eval_dataloader=None, mm_eval_labels=None)

    del trainer
    return 0


if __name__ == "__main__":
    main()
