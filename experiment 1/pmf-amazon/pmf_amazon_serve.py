import os
import numpy as np
from openrec.legacy import ImplicitModelTrainer
from openrec.legacy.utils import ImplicitDataset
from openrec.legacy.recommenders import PMF
from openrec.legacy.utils.evaluators import AUC, Recall, Precision, NDCG
from openrec.legacy.utils.samplers import PointwiseSampler
from openrec.legacy.utils.evaluators import ImplicitEvalManager

os.system("wget https://s3.amazonaws.com/cornell-tech-sdl-rec-bias/dataset/amazon/rsrf_user_data_train.npy")
os.system("wget https://s3.amazonaws.com/cornell-tech-sdl-rec-bias/dataset/amazon/rsrf_user_data_val.npy")
os.system("wget https://s3.amazonaws.com/cornell-tech-sdl-rec-bias/dataset/amazon/rsrf_user_data_test.npy")

os.system("wget https://s3.amazonaws.com/cornell-tech-sdl-rec-bias/best-models/pmf-amazon/pmf-amazon-auc.data-00000-of-00001")
os.system("wget https://s3.amazonaws.com/cornell-tech-sdl-rec-bias/best-models/pmf-amazon/pmf-amazon-auc.index")
os.system("wget https://s3.amazonaws.com/cornell-tech-sdl-rec-bias/best-models/pmf-amazon/pmf-amazon-auc.meta")

raw_data = dict()
raw_data['train_data'] = np.load("rsrf_user_data_train.npy")
raw_data['val_data'] = np.load("rsrf_user_data_val.npy")
raw_data['test_data'] = np.load("rsrf_user_data_test.npy")
raw_data['max_user'] = 99473
raw_data['max_item'] = 450166
batch_size = 8000
test_batch_size = 1000
display_itr = 5000

train_dataset = ImplicitDataset(raw_data['train_data'], raw_data['max_user'], raw_data['max_item'], name='Train')
val_dataset = ImplicitDataset(raw_data['val_data'], raw_data['max_user'], raw_data['max_item'], name='Val')
test_dataset = ImplicitDataset(raw_data['test_data'], raw_data['max_user'], raw_data['max_item'], name='Test')

model = PMF(batch_size=batch_size, max_user=train_dataset.max_user(), max_item=train_dataset.max_item(), 
            dim_embed=100, opt='Adam', sess_config=None, l2_reg=0.001)
sampler = PointwiseSampler(batch_size=batch_size, dataset=train_dataset, pos_ratio=0.2, num_process=5)
model_trainer = ImplicitModelTrainer(batch_size=batch_size, test_batch_size=test_batch_size, 
    train_dataset=train_dataset, model=model, sampler=sampler, 
    eval_save_prefix="pmf-amazon")
auc_evaluator = AUC()
recall_evaluator = Recall(recall_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
precision_evaluator = Precision(precision_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
ndcg_evaluator = NDCG(ndcg_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

model.load("pmf-amazon-auc")

model_trainer._eval_manager = ImplicitEvalManager(evaluators=[auc_evaluator, recall_evaluator, ndcg_evaluator, precision_evaluator])
model_trainer._num_negatives = 200
model_trainer._exclude_positives([test_dataset])
model_trainer._sample_negatives(seed=10)
model_trainer._eval_save_prefix = "pmf-amazon-test"
model_trainer._evaluate_partial(test_dataset)

