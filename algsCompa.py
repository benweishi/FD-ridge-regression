
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
from models.frequent_directions import FrequentDirections, ISVD
from models.randomProjections import RandomProjections, Hashing
from datasets.low_rank_regression import LowRankRegression
import time

dp = 13
n_FD_ells = 6
n_RP_ells = 9
d = 2**dp
n_samples = 2**(dp+4)
eval_size = n_samples // 8
test_size = n_samples // 8
effective_rank = 0.1
noise = 1
random_state = 0
make_data_params = dict(n_features=d,
                        effective_rank=effective_rank,
                        noise=noise,
                        correlation=2,
                        random_state=random_state,
                        random_dir=False)
# data
print("d={}; n_samples={}.".format(d, n_samples))
print("Init data generator.")
start_time = time.time()
data = LowRankRegression(**make_data_params)
print(time.time() - start_time)
theorem_gamma = (d * noise**2) / np.linalg.norm(data.coefs)**2
s = data.sigmas
s_df = pd.DataFrame(data=s, columns=['s'])
s_df.to_csv('./output/data_sigmas.csv', sep=' ', index_label='i')

# train
###############################################################################
# init models
print("Init models.")
FD_ells = np.array([2**p for p in range(dp - n_FD_ells, dp, 1)])
RP_ells = np.array([2**p for p in range(dp - n_FD_ells, dp - n_FD_ells + n_RP_ells, 1)])
print("FD_ells:", FD_ells)
print("RP_ells:", RP_ells)
np.savetxt('FD_ells.csv', FD_ells, delimiter=' ', header='ell')
np.savetxt('RP_ells.csv', RP_ells, delimiter=' ', header='ell')
###
algs = []
train_times = []
algs.append(Ridge(d=d))
train_times.append(0)
FD_idx = []
iSVD_idx = []
RP_idx = []
Hash_idx = []
RFD_idx = []
i = 1
for ell in FD_ells:
    algs.append(FrequentDirections(d=d, ell=ell))
    train_times.append(0)
    FD_idx.append(i)
    i += 1
for ell in FD_ells:
    algs.append(ISVD(d=d, ell=ell))
    train_times.append(0)
    iSVD_idx.append(i)
    i += 1
for ell in RP_ells:
    algs.append(RandomProjections(d=d, ell=ell))
    train_times.append(0)
    RP_idx.append(i)
    i += 1
for ell in RP_ells:
    algs.append(Hashing(d=d, ell=ell))
    train_times.append(0)
    Hash_idx.append(i)
    i += 1
for ell in FD_ells:
    RFD_idx.append(i)
    i += 1
# start training
batch_size = 2**(dp - n_FD_ells)
n_batch = n_samples // batch_size
print("Start training, batch size: {}; number batches: {}.".format(batch_size, n_batch))
pbar = tqdm(total=n_batch, ascii='#')
for i in range(n_batch):
    X, y = data.sampleData(batch_size)
    for j in range(len(algs)):
        start_time = time.time()
        algs[j].partial_fit(X, y)
        train_times[j] += time.time() - start_time
    pbar.update(1)
pbar.close()
train_times = np.array(train_times)
# Output training time
print('Full Ridge Regression training time:', train_times[0])
pd.DataFrame({'ell': FD_ells, 'FD': train_times[FD_idx], 'iSVD': train_times[iSVD_idx]}).set_index('ell').to_csv('output/FD_train_time.csv', sep=' ')
pd.DataFrame({'ell': RP_ells, 'RP': train_times[RP_idx], 'Hashing': train_times[Hash_idx]}).set_index('ell').to_csv('output/RP_train_time.csv', sep=' ')
# evaluate
###############################################################################
# query time
print('Compute coefficients.')
gs = np.array([8**(p) for p in range(dp//2 - 9, dp//2, 1)])
np.savetxt('gammas.csv', gs, delimiter=' ', header='gamma')
query_times = []
coefs = []
pbar = tqdm(total=len(algs), ascii='#')
for i in range(len(algs)):
    start_time = time.time()
    coefs.append([algs[i].compute_coef(gamma) for gamma in gs])
    query_times.append((time.time() - start_time)/len(gs))
    pbar.update(1)
pbar.close()
pbar = tqdm(total=len(FD_ells), ascii='#')
for i in range(1, 1+len(FD_ells), 1):
    start_time = time.time()
    coefs.append([algs[i].compute_coef(gamma, True) for gamma in gs])
    query_times.append((time.time() - start_time)/len(gs))
    pbar.update(1)
pbar.close()
query_times = np.array(query_times)
coefs = np.array(coefs)
print('Full Ridge Regression mean query time:', query_times[0])
pd.DataFrame({'ell': FD_ells, 'FD': query_times[FD_idx], 'iSVD': query_times[iSVD_idx]}).set_index('ell').to_csv('output/FD_query_time.csv', sep=' ')
pd.DataFrame({'ell': RP_ells, 'RP': query_times[RP_idx], 'Hashing': query_times[Hash_idx]}).set_index('ell').to_csv('output/RP_query_time.csv', sep=' ')
pd.DataFrame({'ell': FD_ells, 'FD': query_times[FD_idx]+train_times[FD_idx], 'iSVD': query_times[iSVD_idx]+train_times[iSVD_idx]}).set_index('ell').to_csv('output/FD_time.csv', sep=' ')
pd.DataFrame({'ell': RP_ells, 'RP': query_times[RP_idx]+train_times[RP_idx], 'Hashing': query_times[Hash_idx]+train_times[Hash_idx]}).set_index('ell').to_csv('output/RP_time.csv', sep=' ')
# evaluation
print("Cross validation.")
X, y = data.sampleData(eval_size)
scores = np.array([[r2_score(y, X @ coef) for coef in g_coefs] for g_coefs in coefs])
errors = np.array([[np.linalg.norm(o_coef - coef) for coef, o_coef in zip(g_coefs, coefs[0])] for g_coefs in coefs])
best_idx = np.argmax(scores, axis=1)
best_scores = scores[range(len(scores)), best_idx]
best_coefs = coefs[range(len(coefs)), best_idx]
best_gammas = gs[best_idx]
print('Full Ridge Regression best score:', best_scores[0])
score_header = ['RR']
score_header.extend(['FD_{}'.format(ell) for ell in FD_ells])
score_header.extend(['iSVD_{}'.format(ell) for ell in FD_ells])
score_header.extend(['RP_{}'.format(ell) for ell in RP_ells])
score_header.extend(['Hashing_{}'.format(ell) for ell in RP_ells])
score_header.extend(['RFD_{}'.format(ell) for ell in FD_ells])
gamma_scores = pd.DataFrame(scores.T, index=gs, columns=score_header)
gamma_scores.index.names = ['gamma']
gamma_scores.to_csv('output/gamma-scores.csv', sep=' ')
print('Full Ridge Regression best gamma:', best_gammas[0])
pd.DataFrame({'ell': FD_ells, 'FD': best_gammas[FD_idx], 'iSVD': best_gammas[iSVD_idx], 'RFD': best_gammas[RFD_idx]}).set_index('ell').to_csv('output/FD_best_gamma.csv', sep=' ')
pd.DataFrame({'ell': RP_ells, 'RP': best_gammas[RP_idx], 'Hashing': best_gammas[Hash_idx]}).set_index('ell').to_csv('output/RP_best_gamma.csv', sep=' ')
# test
X, y = data.sampleData(eval_size)
test_scores = np.array([r2_score(y, X @ coef) for coef in best_coefs])
print('Full Ridge Regression test score:', test_scores[0])
pd.DataFrame({'ell': FD_ells, 'FD': test_scores[FD_idx], 'iSVD': test_scores[iSVD_idx], 'RFD': test_scores[RFD_idx]}).set_index('ell').to_csv('output/FD_test_score.csv', sep=' ')
pd.DataFrame({'ell': RP_ells, 'RP': test_scores[RP_idx], 'Hashing': test_scores[Hash_idx]}).set_index('ell').to_csv('output/RP_test_score.csv', sep=' ')
# time-scores
pd.DataFrame({'time': query_times[FD_idx]+train_times[FD_idx], 'score': test_scores[FD_idx]}).to_csv('output/FD_time_score.csv', sep=' ', index=False)
pd.DataFrame({'time': query_times[iSVD_idx]+train_times[iSVD_idx], 'score': test_scores[iSVD_idx]}).to_csv('output/iSVD_time_score.csv', sep=' ', index=False)
pd.DataFrame({'time': query_times[FD_idx]+train_times[FD_idx], 'score': test_scores[RFD_idx]}).to_csv('output/RFD_time_score.csv', sep=' ', index=False)
pd.DataFrame({'time': query_times[RP_idx]+train_times[RP_idx], 'score': test_scores[RP_idx]}).to_csv('output/RP_time_score.csv', sep=' ', index=False)
pd.DataFrame({'time': query_times[Hash_idx]+train_times[Hash_idx], 'score': test_scores[Hash_idx]}).to_csv('output/Hash_time_score.csv', sep=' ', index=False)
