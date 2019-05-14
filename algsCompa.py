
'''
The coefficient R^2 is defined as (1 - u/v), where u is the residual
sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
sum of squares ((y_true - y_true.mean()) ** 2).sum().
Best possible score is 1.0 and it can be negative.
'''
from sklearn.model_selection import train_test_split
from numpy.linalg import svd, norm
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from tqdm import tqdm
#from sklearn.linear_model import Ridge
from models.ridge import Ridge
from models.frequent_directions import FrequentDirections, RobustFrequentDirections, ISVD
from models.randomProjections import RandomProjections, Hashing
from datasets.samples_generator import make_regression
from datasets.low_rank_regression import LowRankRegression


d = 2000
n_samples = 2000
test_size = 500
effective_rank = 0.1
random_state = 0
make_data_params = dict(n_features=d,
                        effective_rank=effective_rank,
                        noise=0.2,
                        correlation=2,
                        random_state=random_state)
# data
data = LowRankRegression(**make_data_params)
s = data.sigmas
s_df = pd.DataFrame(data=s, columns=['s'])
s_df.to_csv('./output/data_sigmas.csv', sep=' ', index_label='i')
# gamma choice
X_train, y_train = data.sampleData(n_samples)
X_test, y_test = data.sampleData(test_size)
print("mean of abs(X):", np.abs(X_train).mean())
print("mean of abs(y):", np.abs(y_train).mean())
g_score = []
for p in range(-12, 10, 1):
    g = 2**(p)
    ridge_regression = Ridge(d=d, gamma=g)
    ridge_regression.fit(X_train, y_train)
    y_pred = ridge_regression.predict(X_test)
    y_score = r2_score(y_test, y_pred)
    g_score.append([g, y_score])
g_df = pd.DataFrame(data=g_score, columns=['g', 'score'])
g_df.to_csv('./output/gamma_choice.csv', sep=' ', index=False)
best_g = g_df.loc[g_df.score.idxmax(), 'g']
print('best gamma:', best_g)
print('ridge score:', g_df.loc[g_df.score.idxmax(), 'score'])
ridge_regression = Ridge(d=d, gamma=best_g)
ridge_regression.fit(X_train, y_train)
# w_ridge = ridge_regression.coef_
w_ridge = ridge_regression.get_params()
#print("ridge coefs:", w_ridge)
# fitting
epses = []
scores = []
times = []
results = []  # ell * alg * result
ells = []
pbar = tqdm(total=100, ascii='#')
for ell in range(10, 101, 10):
    ells.append(ell)
    batch_size = ell
    n_batch = n_samples // batch_size
    # models
    algs = []
    algs.append(FrequentDirections(gamma=best_g, d=d, ell=batch_size))
    algs.append(RobustFrequentDirections(gamma=best_g, d=d, ell=batch_size))
    algs.append(ISVD(gamma=best_g, d=d, ell=batch_size))
    algs.append(RandomProjections(gamma=best_g, d=d, ell=batch_size))
    algs.append(Hashing(gamma=best_g, d=d, ell=batch_size))
    for i in range(n_batch):
        for alg in algs:
            alg.partial_fit(X_train[i*batch_size:(i+1)*batch_size], y_train[i*batch_size:(i+1)*batch_size])
    # evaluate
    result = []
    for alg in algs:
        w_est = alg.get_params()
        eps = np.linalg.norm(w_ridge - w_est) / np.linalg.norm(w_ridge)
        y_pred = alg.predict(X_test)
        score = r2_score(y_test, y_pred)
        result.append([eps, score, alg.train_time])
    results.append(result)
    pbar.update(10)
pbar.close()
results = np.array(results)  # ell * alg * result
names = ['FD', 'RFD', 'iSVD', 'RP', 'Hashing']
result_df = pd.DataFrame(data=results[:, :, 0], columns=names, index=ells)
result_df.to_csv('./output/eps.csv', sep=' ', index_label='ell')
result_df = pd.DataFrame(data=results[:, :, 1], columns=names, index=ells)
result_df.to_csv('./output/score.csv', sep=' ', index_label='ell')
result_df = pd.DataFrame(data=results[:, :, 2], columns=names, index=ells)
result_df.to_csv('./output/time.csv', sep=' ', index_label='ell')
