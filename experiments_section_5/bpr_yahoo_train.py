import os
import numpy as np
from openrec.legacy import ImplicitModelTrainer
from openrec.legacy.utils import ImplicitDataset
from openrec.legacy.recommenders import BPR
from openrec.legacy.utils.evaluators import AUC
from openrec.legacy.utils.samplers import PairwiseSampler

os.system("wget https://s3.amazonaws.com/cornell-tech-sdl-rec-bias/dataset/yahoo/training_arr.npy")
os.system("wget https://s3.amazonaws.com/cornell-tech-sdl-rec-bias/dataset/yahoo/validation_arr.npy")

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

bpr_model = BPR(batch_size=batch_size, max_user=train_dataset.max_user(), max_item=train_dataset.max_item(), 
    dim_embed=50, l2_reg=0.001, opt='Adam', sess_config=None)
sampler = PairwiseSampler(batch_size=batch_size, dataset=train_dataset , num_process=4)
model_trainer = ImplicitModelTrainer(batch_size=batch_size, test_batch_size=test_batch_size,
                                     train_dataset=train_dataset, model=bpr_model, sampler=sampler,
                                     eval_save_prefix="bpr-yahoo",
                                     item_serving_size=666)
auc_evaluator = AUC()

model_trainer.train(num_itr=10001, display_itr=display_itr, eval_datasets=[val_dataset],
                    evaluators=[auc_evaluator], num_negatives=200)

