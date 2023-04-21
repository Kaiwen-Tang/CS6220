import torch
import logging
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, MSELoss
from optimization import BertAdam
from helper import *
from utils_glue import *
import numpy as np
import pickle
logging.basicConfig(level=logging.INFO)


class Trainer(object):
    def __init__(self, args, device, model,num_train_optimization_steps=None):
        self.args = args
        self.device = device
        self.n_gpu = torch.cuda.device_count()
        self.model = model
        self.num_train_optimization_steps = num_train_optimization_steps

    def build(self, lr=None):
        self.prev_global_step = 0
        self.output_dir = os.path.join(self.args.output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]

        schedule = 'warmup_linear'
        learning_rate = self.args.learning_rate if not lr else lr
        self.optimizer = BertAdam(optimizer_grouped_parameters,
                             schedule=schedule,
                             lr=learning_rate,
                             warmup=self.args.warmup_proportion,
                             t_total=self.num_train_optimization_steps)
        logging.info("Optimizer prepared.")
        # self._setup_grad_scale_stats()

    def _do_eval(self, model, task_name, eval_dataloader, output_mode, eval_labels, num_labels):
        eval_loss = 0
        nb_eval_steps = 0
        preds = []

        for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
            batch_ = tuple(t.to(self.device) for t in batch_)
            with torch.no_grad():
                input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_

                logits = model(input_ids, segment_ids, input_mask)

            # create eval loss and other metric required by the task
            if output_mode == "classification":
                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            elif output_mode == "regression":
                loss_fct = MSELoss()
                tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps

        preds = preds[0]
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(preds)
        # print(preds, eval_labels.numpy())
        result = compute_metrics(task_name, preds, eval_labels.numpy())
        result['eval_loss'] = eval_loss

        return result

    def train(self, train_examples, task_name, output_mode, eval_labels, num_labels,
                    train_dataloader, eval_dataloader, eval_examples, tokenizer, mm_eval_labels, mm_eval_dataloader):
        loss_mse = MSELoss()

        global_step = self.prev_global_step
        best_dev_acc = 0.0
        output_eval_file = os.path.join(self.args.output_dir, "eval_results.txt")

        logging.info("***** Running training, Task: %s, Job id: %s*****" % (self.args.task_name, self.args.job_id))
        logging.info("  Num examples = %d", len(train_examples))
        logging.info("  Batch size = %d", self.args.batch_size)
        logging.info("  Num steps = %d", self.num_train_optimization_steps)

        global_tr_loss = 0
        for epoch_ in range(self.args.num_train_epochs):

            tr_loss = 0.
            nb_tr_examples, nb_tr_steps = 0, 0
            print('training epoch', epoch_)

            for step, batch in enumerate(train_dataloader):

                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
                logits = self.model(input_ids, segment_ids, input_mask)

                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits, label_ids.view(-1))
                elif output_mode == "regression":
                    loss_mse = MSELoss()
                    loss = loss_mse(logits.view(-1), label_ids.view(-1))

                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                global_tr_loss += loss.item()
                nb_tr_examples += label_ids.size(0)
                nb_tr_steps += 1

                # evaluation and save model
                if global_step % self.args.eval_step == 0 or \
                        global_step == len(train_dataloader)-1:

                    logging.info("  Epoch = {} iter {} step".format(epoch_, global_step))
                    logging.info("  Num examples = %d", len(eval_examples))
                    logging.info(f"  Previous best = {best_dev_acc}")

                    loss = tr_loss / (step + 1)
                    global_avg_loss = global_tr_loss / (global_step + 1)

                    self.model.eval()
                    result = self._do_eval(self.model, task_name, eval_dataloader, output_mode, eval_labels, num_labels)
                    result['global_step'] = global_step
                    result['loss'] = loss
                    result['global_loss'] = global_avg_loss
                    print("eval acc", result['acc'])

                    preds = logits.detach().cpu().numpy()
                    train_label = label_ids.cpu().numpy()
                    if output_mode == "classification":
                        preds = np.argmax(preds, axis=1)
                        
                    elif output_mode == "regression":
                        preds = np.squeeze(preds)
                    result['train_batch_acc'] = list(compute_metrics(task_name, preds, train_label).values())[0]

                    result_to_file(result, output_eval_file)

                    save_model = False

                    if task_name in acc_tasks and result['acc'] > best_dev_acc:
                        best_dev_acc = result['acc']
                        save_model = True

                    if task_name in corr_tasks and result['corr'] > best_dev_acc:
                        best_dev_acc = result['corr']
                        save_model = True

                    if task_name in mcc_tasks and result['mcc'] > best_dev_acc:
                        best_dev_acc = result['mcc']
                        save_model = True

                    if save_model:
                        self._save()

                        if task_name == "mnli":
                            logging.info('MNLI-mm Evaluation')
                            result = self._do_eval(self.model, 'mnli-mm', mm_eval_dataloader, output_mode, mm_eval_labels, num_labels)
                            result['global_step'] = global_step
                            if not os.path.exists(self.output_dir + '-MM'):
                                os.makedirs(self.output_dir + '-MM')
                            tmp_output_eval_file = os.path.join(self.output_dir + '-MM', "eval_results.txt")
                            result_to_file(result, tmp_output_eval_file)


                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1

    def _save(self):
        logging.info("******************** Save model ********************")
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        output_model_file = os.path.join(self.output_dir, 'model')
        output_config_file = os.path.join(self.output_dir, 'model')
        torch.save(model_to_save.state_dict(), output_model_file)
        # model_to_save.config.to_json_file(output_config_file)

    def _setup_grad_scale_stats(self):
        self.grad_scale_stats = {'weight': None, \
                                 'bias': None, \
                                 'layer_norm': None}
        self.ema_grad = 0.9
