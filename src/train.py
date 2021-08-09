import pickle
import sys
import yaml
import lightgbm as lgb
import numpy as np
import mlflow
import rebase as rb
import os


tag = os.environ.get('HP_SEARCH')
if tag:
    rb.set_tag('HP_SEARCH', tag)


in_path = sys.argv[1]
out_path = sys.argv[2]

with open(in_path, 'rb') as f:
    train_set = pickle.load(f)



valid_sets = [train_set]
valid_names = ['train']

evals_result = {}

model_params = yaml.safe_load(open('params.yaml'))['train']
print(model_params)


results = lgb.cv(model_params, train_set, nfold=3, stratified=False)
num_rounds = max(np.argmin(results['quantile-mean']), 50)

model_params['num_trees'] = num_rounds
if 'early_stopping' in model_params:
    del model_params['early_stopping']

rb.log_params(model_params)


gbm = lgb.train(model_params,
                train_set,
                valid_sets=valid_sets,
                valid_names=valid_names,
                evals_result=evals_result,
                verbose_eval=False,
                callbacks=None)


print(evals_result)
score = evals_result['train'][model_params['objective']][-1]
print(score)

rb.log_metric('score', score)
with open(out_path, 'wb') as f:
    pickle.dump(gbm, f)
