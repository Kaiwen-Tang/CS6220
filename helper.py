import logging
import os
import string
import random
import torch

def generate_job_id():
  return ''.join(random.sample(string.ascii_letters+string.digits, 5))

def init_logging(log_path):

  if not os.path.isdir(os.path.dirname(log_path)):
    print("Log path does not exist. Create a new one.")
    os.makedirs(os.path.dirname(log_path))
  if os.path.exists(log_path):
    print("%s already exists. replace it with current experiment." % log_path)
    os.system('rm %s' % log_path)

  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  logFormatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')

  fileHandler = logging.FileHandler(log_path)
  fileHandler.setFormatter(logFormatter)
  logger.addHandler(fileHandler)

  consoleHandler = logging.StreamHandler()
  consoleHandler.setFormatter(logFormatter)
  logger.addHandler(consoleHandler)

def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        logging.info("{0}: {1}".format(k, v))

def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return (- targets_prob * student_likelihood).mean()

def result_to_file(result, file_name):
    with open(file_name, "a") as writer:
        logging.info("***** Eval results *****")
        for key in sorted(result.keys()):
            if result[key]>0.0:
                logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
