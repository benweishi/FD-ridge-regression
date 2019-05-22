
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


d = 1000
n_samples = 1000
eval_size = 200
test_size = 200
effective_rank = 0.1
random_state = 0
make_data_params = dict(n_features=d,
                        effective_rank=effective_rank,
                        noise=1,
                        correlation=2,
                        random_state=random_state)
# data
data = LowRankRegression(**make_data_params)
s = data.sigmas
s_df = pd.DataFrame(data=s, columns=['s'])
s_df.to_csv('./output/data_sigmas.csv', sep=' ', index_label='i')
X_train, y_train = data.sampleData(n_samples)
X_eval, y_eval = data.sampleData(eval_size)
X_test, y_test = data.sampleData(test_size)

# train
###############################################################################
# train RR
ridge_regression = Ridge(d=d)
ridge_regression.fit(X_train, y_train)
# train ohter models
algs = {'FD': [], 'RFD': [], 'iSVD': [], 'RP': [], 'Hashing': []}
ells = np.arange(10, 101, 10, dtype=np.int)
pbar = tqdm(total=100, ascii='#')
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
    pbar.update(10)
pbar.close()

# evaluate
###############################################################################
# training time
times = {}
for key, value in algs.items():
    time = [alg.train_time for alg in value]
    times[key] = time
times_df = pd.DataFrame.from_dict(times).set_index(ells)
times_df.to_csv('./output/time.csv', sep=' ', index_label='ell')
# gamma choice
gs = [2**(p) for p in range(-9, 9, 1)]
score = {}
score['gamma'] = gs
score['RR'] = [r2_score(y_test, ridge_regression.predict(X_test, g)) for g in gs]
score['RFD10'] = [r2_score(y_test, algs['RFD'][0].predict(X_test, g)) for g in gs]
score['RFD50'] = [r2_score(y_test, algs['RFD'][4].predict(X_test, g)) for g in gs]
score['RFD100'] = [r2_score(y_test, algs['RFD'][9].predict(X_test, g)) for g in gs]
gamma_df = pd.DataFrame.from_dict(score).set_index('gamma')
gamma_df.to_csv('./output/gamma-score.csv', sep=' ')
# RFD l-score, gamma = [0, 0.1, 1, 10]
score = {}
score['rfd0'] = [r2_score(y_test, alg.predict(X_test)) for alg in algs['RFD']]
score['rfd1'] = [r2_score(y_test, alg.predict(X_test, gamma=1.0)) for alg in algs['RFD']]
score['rfd10'] = [r2_score(y_test, alg.predict(X_test, gamma=10.0)) for alg in algs['RFD']]
score['rfd30'] = [r2_score(y_test, alg.predict(X_test, gamma=30.0)) for alg in algs['RFD']]
score['rfd100'] = [r2_score(y_test, alg.predict(X_test, gamma=100.0)) for alg in algs['RFD']]
score_df = pd.DataFrame.from_dict(score).set_index(ells)
score_df.to_csv('./output/rfd-ell-score.csv', sep=' ', index_label='ell')
# model compare
scores = {}
gammas = {}
errors = {}
for key, value in algs.items():
    eval_scores = [[r2_score(y_eval, alg.predict(X_eval, g)) for g in gs] for alg in value]
    eval_scores = np.array(eval_scores)
    best_g_i = np.argmax(eval_scores, axis=1)
    gammas[key] = np.array(gs)[best_g_i]
    scores[key] = [r2_score(y_eval, value[i].predict(X_eval, gammas[key][i])) for i in range(len(value))]
    errors[key] = [np.linalg.norm(ridge_regression.get_coef(gammas[key][i]) - value[i].get_coef(gammas[key][i])) / np.linalg.norm(ridge_regression.get_coef(gammas[key][i])) for i in range(len(value))]
gammas_df = pd.DataFrame.from_dict(gammas).set_index(ells)
gammas_df.to_csv('./output/gammas.csv', sep=' ', index_label='ell')
scores_df = pd.DataFrame.from_dict(scores).set_index(ells)
scores_df.to_csv('./output/scores.csv', sep=' ', index_label='ell')
errors_df = pd.DataFrame.from_dict(errors).set_index(ells)
errors_df.to_csv('./output/errors.csv', sep=' ', index_label='ell')
