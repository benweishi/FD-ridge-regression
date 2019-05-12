
'''
The coefficient R^2 is defined as (1 - u/v), where u is the residual
sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
sum of squares ((y_true - y_true.mean()) ** 2).sum().
Best possible score is 1.0 and it can be negative.
'''
from datasets.samples_generator import make_regression
from sklearn.model_selection import train_test_split
from numpy.linalg import svd
from models.frequent_directions import FrequentDirections
from sklearn.metrics import r2_score


batch_size = 10
n_batch = 10
test_size = 10
d = 15
effective_rank = 2
random_state = 0
algs = []
make_data_params = dict(n_samples=batch_size*n_batch+test_size,
                        n_features=d,
                        effective_rank=effective_rank,
                        tail_strength=0.01,
                        noise=0.2,
                        coef_range=10,
                        coef=True,
                        random_state=random_state)
# data
X, y, w = make_regression(**make_data_params)
print("model coefs:", w)
_, s, Vt = svd(X)
print("X's sigular values:", s)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                    random_state=random_state)
# models
alg = FrequentDirections(gamma=0.01, d=d, ell=effective_rank*10)
# fitting
for i in range(n_batch):
    alg.partial_fit(X_train[i*batch_size:(i+1)*batch_size], y_train[i*batch_size:(i+1)*batch_size])
    '''
    for alg in algs:
        alg.partial_fit(X, y)
    '''
# evaluate
w_opt = None
for alg in algs:
    pass
w_est = alg.get_params()
print("estimated coefs:", w_est)
y_pred = alg.predict(X_test)
score = r2_score(y_test, y_pred)
print(score)
