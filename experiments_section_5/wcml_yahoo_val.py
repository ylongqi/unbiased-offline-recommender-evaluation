import os
import numpy as np
from openrec.legacy import ImplicitModelTrainer
from openrec.legacy.utils import ImplicitDataset
from openrec.legacy.recommenders import WCML
from openrec.legacy.utils.evaluators import AUC
from openrec.legacy.utils.samplers import NPairwiseSampler
from openrec.legacy.utils.evaluators import ImplicitEvalManager

os.system("wget https://s3.amazonaws.com/cornell-tech-sdl-rec-bias/dataset/yahoo/training_arr.npy")
os.system("wget https://s3.amazonaws.com/cornell-tech-sdl-rec-bias/dataset/yahoo/validation_arr.npy")

os.system("wget https://s3.amazonaws.com/cornell-tech-sdl-rec-bias/best-models/wcml-yahoo/wcml-yahoo.data-00000-of-00001")
os.system("wget https://s3.amazonaws.com/cornell-tech-sdl-rec-bias/best-models/wcml-yahoo/wcml-yahoo.meta")
os.system("wget https://s3.amazonaws.com/cornell-tech-sdl-rec-bias/best-models/wcml-yahoo/wcml-yahoo.index")

raw_data = dict()
raw_data['train_data'] = np.load("training_arr.npy")
raw_data['val_data'] = np.load("validation_arr.npy")
raw_data['max_user'] = 15401
raw_data['max_item'] = 1001
batch_size = 8000
test_batch_size = 1000
display_itr = 1000

train_dataset = ImplicitDataset(raw_data['train_data'], raw_data['max_user'], raw_data['max_item'], name='Train')
val_dataset = ImplicitDataset(raw_data['val_data'], raw_data['max_user'], raw_data['max_item'], name='Val')

wcml_model = WCML(batch_size=batch_size, max_user=train_dataset.max_user(), max_item=train_dataset.max_item(), 
    dim_embed=50, neg_num=5, l2_reg=0.001, opt='Adam', sess_config=None)
sampler = NPairwiseSampler(batch_size=batch_size, dataset=train_dataset, negativenum=5, num_process=4)
model_trainer = ImplicitModelTrainer(batch_size=batch_size, test_batch_size=test_batch_size,
                                     train_dataset=train_dataset, model=wcml_model, sampler=sampler,
                                     eval_save_prefix="wcml-yahoo",
                                     item_serving_size=500)
auc_evaluator = AUC()

wcml_model.load("wcml-yahoo")

model_trainer._eval_manager = ImplicitEvalManager(evaluators=[auc_evaluator])
model_trainer._num_negatives = 300
model_trainer._exclude_positives([train_dataset, val_dataset])
model_trainer._sample_negatives(seed=10)
model_trainer._eval_save_prefix = "wcml-yahoo-val"
model_trainer._evaluate_partial(val_dataset)

