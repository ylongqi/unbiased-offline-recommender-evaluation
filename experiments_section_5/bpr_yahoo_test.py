import os
import numpy as np
from openrec.legacy import ImplicitModelTrainer
from openrec.legacy.utils import ImplicitDataset
from openrec.legacy.utils.evaluators import ImplicitEvalManager
from openrec.legacy.recommenders import BPR
from openrec.legacy.utils.evaluators import AUC
from openrec.legacy.utils.samplers import PairwiseSampler


os.system("wget https://s3.amazonaws.com/cornell-tech-sdl-rec-bias/dataset/yahoo/training_arr.npy")
os.system("wget https://s3.amazonaws.com/cornell-tech-sdl-rec-bias/dataset/yahoo/test_arr_pos.npy")
os.system("wget https://s3.amazonaws.com/cornell-tech-sdl-rec-bias/dataset/yahoo/test_arr_neg.npy")

os.system("wget https://s3.amazonaws.com/cornell-tech-sdl-rec-bias/best-models/bpr-yahoo/bpr-yahoo.data-00000-of-00001")
os.system("wget https://s3.amazonaws.com/cornell-tech-sdl-rec-bias/best-models/bpr-yahoo/bpr-yahoo.meta")
os.system("wget https://s3.amazonaws.com/cornell-tech-sdl-rec-bias/best-models/bpr-yahoo/bpr-yahoo.index")

raw_data = dict()
raw_data['train_data'] = np.load("training_arr.npy")
raw_data['test_data_pos'] = np.load("test_arr_pos.npy")
raw_data['test_data_neg'] = np.load("test_arr_neg.npy")
raw_data['max_user'] = 15401
raw_data['max_item'] = 1001
batch_size = 8000
test_batch_size = 1000
display_itr = 1000

train_dataset = ImplicitDataset(raw_data['train_data'], raw_data['max_user'], raw_data['max_item'], name='Train')
val_dataset = ImplicitDataset(raw_data['val_data'], raw_data['max_user'], raw_data['max_item'], name='Val')

model = BPR(batch_size=batch_size, max_user=train_dataset.max_user(), max_item=train_dataset.max_item(), 
    dim_embed=50, l2_reg=0.001, opt='Adam', sess_config=None)
sampler = PairwiseSampler(batch_size=batch_size, dataset=train_dataset, num_process=4)
model_trainer = ImplicitModelTrainer(batch_size=batch_size, test_batch_size=test_batch_size,
                                     train_dataset=train_dataset, model=model, sampler=sampler,
                                     eval_save_prefix="bpr-yahoo",
                                     item_serving_size=666)
auc_evaluator = AUC()

model.load("bpr-yahoo")


model_trainer._eval_manager = ImplicitEvalManager(evaluators=[auc_evaluator])
model_trainer._num_negatives = 200
model_trainer._exclude_positives([train_dataset, test_dataset_pos, test_dataset_neg])
model_trainer._sample_negatives(seed=10)

model_trainer._eval_save_prefix = "bpr-yahoo-test-pos"
model_trainer._evaluate_partial(test_dataset_pos)

model_trainer._eval_save_prefix = "bpr-yahoo-test-neg"
model_trainer._evaluate_partial(test_dataset_neg)


def eq(infilename, infilename_neg, trainfilename, gamma=-1.0):
    infile = open(infilename, 'rb')
    infile_neg = open(infilename_neg, 'rb')
    P = pickle.load(infile)
    infile.close()
    P_neg = pickle.load(infile_neg)
    infile_neg.close()
    NUM_NEGATIVES = P["num_negatives"]
    #
    for theuser in P["users"]:
        neg_items = list(P_neg["user_items"][theuser][NUM_NEGATIVES:])
        neg_scores = list(P_neg["results"][theuser][NUM_NEGATIVES:])
        P["user_items"][theuser] = list(neg_items) + list(P["user_items"][theuser][NUM_NEGATIVES:])
        P["results"][theuser] = list(neg_scores) + list(P["results"][theuser][NUM_NEGATIVES:])
    #
    Zui = dict()
    Ni = dict()
    # fill in dictionary Ni
    trainset = np.load(trainfilename)
    for i in trainset['item_id']:
        if i in Ni:
            Ni[i] += 1
        else:
            Ni[i] = 1
    del trainset
    # count #users with non-zero item frequencies
    nonzero_user_count = 0
    for theuser in P["users"]:
        pos_items = P["user_items"][theuser][len(P_neg["user_items"][theuser][NUM_NEGATIVES:]):]
        for pos_item in pos_items:
            if pos_item in Ni:
                nonzero_user_count += 1
                break
    # fill in dictionary Zui
    for theuser in P["users"]:
        all_scores = np.array(P["results"][theuser])
        pos_items = P["user_items"][theuser][len(P_neg["user_items"][theuser][NUM_NEGATIVES:]):]
        pos_scores = P["results"][theuser][len(P_neg["results"][theuser][NUM_NEGATIVES:]):]
        for i, pos_item in enumerate(pos_items):
            pos_score = pos_scores[i]
            Zui[(theuser, pos_item)] = float(np.sum(all_scores > pos_score))
    # calculate per-user scores
    sum_user_auc = 0.0
    sum_user_recall = 0.0
    for theuser in P["users"]:
        numerator_auc = 0.0
        numerator_recall = 0.0
        denominator = 0.0
        for theitem in P["user_items"][theuser][len(P_neg["user_items"][theuser][NUM_NEGATIVES:]):]:
            if theitem not in Ni:
                continue
            pui = np.power(Ni[theitem], (gamma + 1) / 2.0)
            numerator_auc += (1 - Zui[(theuser, theitem)] / len(P["user_items"][theuser])) / pui
            if Zui[(theuser, theitem)] < 1:
                numerator_recall += 1.0 / pui
            denominator += 1 / pui
        if denominator > 0:
            sum_user_auc += numerator_auc / denominator
            sum_user_recall += numerator_recall / denominator
    
    return {
        "auc"       : sum_user_auc / nonzero_user_count,
        "recall"    : sum_user_recall / nonzero_user_count
    }


print(eq("bpr-yahoo-test-pos_evaluate_partial.pickle", "bpr-yahoo-test-neg_evaluate_partial.pickle", "training_arr.npy"))







