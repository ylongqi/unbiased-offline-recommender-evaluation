import os
import numpy as np
from openrec.legacy import ImplicitModelTrainer
from openrec.legacy.utils import ImplicitDataset
from openrec.legacy.recommenders import BPR
from openrec.legacy.utils.evaluators import AUC, Recall, Precision, NDCG
from openrec.legacy.utils.samplers import PairwiseSampler
from openrec.legacy.utils.evaluators import ImplicitEvalManager

os.system("wget https://s3.amazonaws.com/cornell-tech-sdl-rec-bias/dataset/citeulike/rsrf_user_data_train.npy")
os.system("wget https://s3.amazonaws.com/cornell-tech-sdl-rec-bias/dataset/citeulike/rsrf_user_data_val.npy")
os.system("wget https://s3.amazonaws.com/cornell-tech-sdl-rec-bias/dataset/citeulike/rsrf_user_data_test.npy")

os.system("wget https://s3.amazonaws.com/cornell-tech-sdl-rec-bias/best-models/bpr-citeulike/bpr-citeulike.data-00000-of-00001")
os.system("wget https://s3.amazonaws.com/cornell-tech-sdl-rec-bias/best-models/bpr-citeulike/bpr-citeulike.index")
os.system("wget https://s3.amazonaws.com/cornell-tech-sdl-rec-bias/best-models/bpr-citeulike/bpr-citeulike.meta")

raw_data = dict()
raw_data['train_data'] = np.load("rsrf_user_data_train.npy")
raw_data['val_data'] = np.load("rsrf_user_data_val.npy")
raw_data['test_data'] = np.load("rsrf_user_data_test.npy")
raw_data['max_user'] = 5551
raw_data['max_item'] = 16980
batch_size = 8000
test_batch_size = 1000
display_itr = 5000

train_dataset = ImplicitDataset(raw_data['train_data'], raw_data['max_user'], raw_data['max_item'], name='Train')
val_dataset = ImplicitDataset(raw_data['val_data'], raw_data['max_user'], raw_data['max_item'], name='Val')
test_dataset = ImplicitDataset(raw_data['test_data'], raw_data['max_user'], raw_data['max_item'], name='Test')

bpr_model = BPR(batch_size=batch_size, max_user=train_dataset.max_user(), max_item=train_dataset.max_item(),
                dim_embed=50, opt='Adam', sess_config=None, l2_reg=0.01)
sampler = PairwiseSampler(batch_size=batch_size, dataset=train_dataset, num_process=1)
model_trainer = ImplicitModelTrainer(batch_size=batch_size, test_batch_size=test_batch_size,
    train_dataset=train_dataset, model=bpr_model, sampler=sampler,
    eval_save_prefix="bpr-citeulike")
auc_evaluator = AUC()
recall_evaluator = Recall(recall_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
precision_evaluator = Precision(precision_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
ndcg_evaluator = NDCG(ndcg_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

bpr_model.load("bpr-citeulike")

model_trainer._eval_manager = ImplicitEvalManager(evaluators=[auc_evaluator, recall_evaluator, ndcg_evaluator, precision_evaluator])
model_trainer._num_negatives = 200
model_trainer._exclude_positives([train_dataset, val_dataset, test_dataset])
model_trainer._sample_negatives(seed=10)
model_trainer._eval_save_prefix = "bpr-citeulike-test"
model_trainer._evaluate_partial(test_dataset)

