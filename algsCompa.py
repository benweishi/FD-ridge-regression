
'''
The coefficient R^2 is defined as (1 - u/v), where u is the residual
sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
sum of squares ((y_true - y_true.mean()) ** 2).sum().
Best possible score is 1.0 and it can be negative.
'''
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from tqdm import tqdm
from models.ridge import Ridge
from models.frequent_directions import FrequentDirections, RobustFrequentDirections, ISVD
from models.randomProjections import RandomProjections, Hashing
from datasets.low_rank_regression import LowRankRegression
import time

dp = 13
d = 2**dp
n_samples = 2**(dp+2)
eval_size = n_samples // 8
test_size = n_samples // 8
effective_rank = 0.1
random_state = 0
make_data_params = dict(n_features=d,
                        effective_rank=effective_rank,
                        noise=1,
                        correlation=2,
                        random_state=random_state,
                        random_dir=False)
# data
print("Init data generator ({0}).".format(d))
start_time = time.time()
data = LowRankRegression(**make_data_params)
print(time.time() - start_time)
s = data.sigmas
s_df = pd.DataFrame(data=s, columns=['s'])
s_df.to_csv('./output/data_sigmas.csv', sep=' ', index_label='i')
print("Generating data ({}).".format(n_samples))
start_time = time.time()
X_train, y_train = data.sampleData(n_samples)
print(time.time() - start_time)
print("Generating data ({}).".format(eval_size))
start_time = time.time()
X_eval, y_eval = data.sampleData(eval_size)
print(time.time() - start_time)
print("Generating data ({}).".format(test_size))
start_time = time.time()
X_test, y_test = data.sampleData(test_size)
print(time.time() - start_time)

# train
###############################################################################
# train RR
print("Training Full Ridge Regression.")
start_time = time.time()
ridge_regression = Ridge(d=d)
ridge_regression.fit(X_train, y_train)
print(time.time() - start_time)
# train ohter models
print("Training all models.")
algs = {'FD': [], 'RFD': [], 'iSVD': [], 'RP': [], 'Hashing': []}
ells = np.array([2**p for p in range(4, dp, 1)])
pbar = tqdm(total=len(ells), ascii='#')
for ell in ells:
    n_batch = n_samples // ell
    algs['FD'].append(FrequentDirections(d=d, ell=ell))
    algs['RFD'].append(RobustFrequentDirections(d=d, ell=ell))
    algs['iSVD'].append(ISVD(d=d, ell=ell))
    algs['RP'].append(RandomProjections(d=d, ell=ell))
    algs['Hashing'].append(Hashing(d=d, ell=ell))
    for i in range(n_batch):
        for alg_list in algs.values():
            alg_list[-1].partial_fit(X_train[i*ell:(i+1)*ell], y_train[i*ell:(i+1)*ell])
    pbar.update(1)
pbar.close()

# evaluate
###############################################################################
# training time
print("Evaluate running time.")
start_time = time.time()
times = {}
for key, value in algs.items():
    times[key] = [alg.get_time() for alg in value]
print(time.time() - start_time)
times_df = pd.DataFrame.from_dict(times).set_index(ells)
times_df.to_csv('./output/time.csv', sep=' ', index_label='ell')
# gamma choice
print("Evaluate gamma and score for RFDs.")
start_time = time.time()
gs = [4**(p) for p in range(dp//2 - 10, dp//2, 1)]
score = {}
score['gamma'] = gs
score['RR'] = [r2_score(y_test, ridge_regression.predict(X_test, g)) for g in gs]
score['RFD16'] = [r2_score(y_test, algs['RFD'][0].predict(X_test, g)) for g in gs]
score['RFD64'] = [r2_score(y_test, algs['RFD'][2].predict(X_test, g)) for g in gs]
score['RFD256'] = [r2_score(y_test, algs['RFD'][4].predict(X_test, g)) for g in gs]
print(time.time() - start_time)
gamma_df = pd.DataFrame.from_dict(score).set_index('gamma')
gamma_df.to_csv('./output/gamma-score.csv', sep=' ')
# RFD l-score, gamma = [0, 0.1, 1, 10]
print("RFD with fixed gammas.")
start_time = time.time()
score = {}
score['rfd0'] = [r2_score(y_eval, alg.predict(X_eval), gamma=0) for alg in algs['RFD']]
score['rfd1'] = [r2_score(y_eval, alg.predict(X_eval, gamma=gs[0])) for alg in algs['RFD']]
score['rfd3'] = [r2_score(y_eval, alg.predict(X_eval, gamma=gs[3])) for alg in algs['RFD']]
score['rfd6'] = [r2_score(y_eval, alg.predict(X_eval, gamma=gs[6])) for alg in algs['RFD']]
score['rfd9'] = [r2_score(y_eval, alg.predict(X_eval, gamma=gs[9])) for alg in algs['RFD']]
print(time.time() - start_time)
score_df = pd.DataFrame.from_dict(score).set_index(ells)
score_df.to_csv('./output/rfd-ell-score.csv', sep=' ', index_label='ell')
# model compare
print("Scores and errors with evaluated gammas.")
start_time = time.time()
scores = {}
gammas = {}
errors = {}
for key, value in algs.items():
    eval_scores = [[r2_score(y_eval, alg.predict(X_eval, g)) for g in gs] for alg in value]
    eval_scores = np.array(eval_scores)
    best_g_i = np.argmax(eval_scores, axis=1)
    gammas[key] = np.array(gs)[best_g_i]
    scores[key] = [r2_score(y_test, value[i].predict(X_test, gammas[key][i])) for i in range(len(value))]
    errors[key] = [np.linalg.norm(ridge_regression.get_coef(gammas[key][i]) - value[i].get_coef(gammas[key][i])) / np.linalg.norm(ridge_regression.get_coef(gammas[key][i])) for i in range(len(value))]
print(time.time() - start_time)
gammas_df = pd.DataFrame.from_dict(gammas).set_index(ells)
gammas_df.to_csv('./output/gammas.csv', sep=' ', index_label='ell')
scores_df = pd.DataFrame.from_dict(scores).set_index(ells)
scores_df.to_csv('./output/scores.csv', sep=' ', index_label='ell')
errors_df = pd.DataFrame.from_dict(errors).set_index(ells)
errors_df.to_csv('./output/errors.csv', sep=' ', index_label='ell')
timeVscore_df = pd.concat([times_df, scores_df], axis=1)
for key, value in algs.items():
    timeVscore_df[key].to_csv('./output/t2s_{}.csv'.format(key), header=['time', 'score'], sep=' ', index_label='ell')
