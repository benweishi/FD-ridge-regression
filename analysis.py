import numpy as np
import time
import os
import copy
import math
import sys
from sklearn.metrics import r2_score
from tqdm import tqdm
from tqdm import trange
from datasets.low_rank_regression import LowRankRegression
from datasets.KSLC import KSLC
from models.ridge import RR
from models.fdrr import FFDRR, iSVDRR, RFDRR, NOFDRR
from models.prr import RPRR, CSRR
from collections.abc import Iterable


def train(alg, dataset, batch_size):
    """Train 1 alg with dataset."""
    n_batch = math.ceil(dataset.n_train / batch_size)
    if alg is None:
        for b in trange(n_batch, desc=f'Train {alg}', leave=False):
            X, y = dataset.trainData(batch_size)
    else:
        for b in trange(n_batch, desc=f'Train {alg}', leave=False):
            X, y = dataset.trainData(batch_size)
            alg.fit(X, y)


def query(alg_list, gamma_list):
    """Query 1 or list of trained alg with gamma or gamma_list.

    Args:
        alg_list (list of rr algorithm): shape a
        gamma_list (numpy.ndarray): dtype('float'), shape g

    Returns:
        Coefs_list (numpy.ndarray): dtype('float'), shape a*g*d
    """
    alg_is_list = isinstance(alg_list, Iterable)
    gamma_is_list = isinstance(gamma_list, Iterable)
    if not alg_is_list:
        alg_list = [alg_list]
    if not gamma_is_list:
        gamma_list = np.array([gamma_list])
    Coefs_list = np.empty((len(alg_list), len(gamma_list), alg_list[0].d))
    for j in trange(len(alg_list), desc=f'Query {len(alg_list)} algorithms.', leave=False):
        for i in trange(len(gamma_list), desc=f'Query {alg_list[j]} with {len(gamma_list)} gammas', leave=False):
            Coefs_list[j, i] = alg_list[j].coefs(gamma_list[i])
    if not gamma_is_list:
        Coefs_list = Coefs_list.squeeze(axis=1)
    if not alg_is_list:
        Coefs_list = Coefs_list.squeeze(axis=0)
    return Coefs_list


def pred(Cs, Xs):
    """Pred b using Coefs in Coefs_list.

    Args:
        Cs (numpy.ndarray): dtype('float'), coefs_list, shape shape*d
        Xs (numpy.ndarray): dtype('float'), shape shape*nt*d

    Returns:
        preds_list (numpy.ndarray): dtype('float'), shape shape*n_test
    """
    Cs = np.expand_dims(Cs, axis=-1)  # reshape to shape*d*1
    result = Xs@Cs  # shape shape*nt*1
    return result.squeeze(axis=-1)  # shape*nt


def make_datasets(p=8, r=0):
    """Generate datasets for the further experiments. Currently 3 datasets.

    Args:
        p: Size of datasets. Dimension is 2**p, n_train is 2**(p+2)

    Returns:
        A dictionary of datasets, currently 3, 'LR','HR','TEMP'.
    """

    print(f'Generating 3 dataset.')
    dataset_dict = {}
    make_data_params = dict(n_features=2**p,
                            n_samples=2**(p+2),
                            eval_size=2**p,
                            test_size=2**p,
                            effective_rank=0.1,
                            noise=4,
                            correlation=1,
                            random_state=r,
                            rotate='dct4',
                            name='LR')
    dataset_dict['LR'] = LowRankRegression(**make_data_params)
    make_data_params = dict(n_features=2**p,
                            n_samples=2**(p+2),
                            eval_size=2**p,
                            test_size=2**p,
                            effective_rank=0.5,
                            noise=4,
                            correlation=1,
                            random_state=r,
                            rotate='dct4',
                            name='HR')
    dataset_dict['HR'] = LowRankRegression(**make_data_params)
    dataset_dict['TEMP'] = KSLC(n_features=2**p, n_samples=2**(p+2),
                                test_size=2**p, random_state=r, name='TEMP')
    return dataset_dict


def data_summary(dataset, out_path, repeat_times=5):
    np.savetxt(os.path.join(out_path, 'data_summary.txt'), np.array([dataset.d, dataset.n_train, dataset.n_test]), fmt='%d', header='n_features n_train n_test')
    S = []
    for r in trange(repeat_times, desc=f'{dataset.name} singular, Repeat {repeat_times} time'):
        dataset.reset(r)
        X, y = dataset.trainData(dataset.n_train)
        s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
        S.append(s)
    np.savetxt(os.path.join(out_path, 'data_singular.txt'), np.array(S).mean(axis=0))


def best_gamma(ds, out_path, repeat_times=5, start=1):
    """Find best gamma value on ds for RR.
    
    Args:
        ds (instance): .datasets.*
        out_path (str): Path to save gammas.txt
        repeat_times (int, optional): Defaults to 5.
        start (number, optional): First and smallest gamma value to try. Defaults to 1.
    
    Returns:
        number: Choice of gamma value (i.e. the gamma value with smallest pred. error)

    Outputs:
        '{out_path}data_name/gammas.txt': (numpy.savetxt): dtype('float'), shape 2*g, [gammas, errors]
    """
    # Train RR
    rrs = []  # shape r
    Xts = []  # shape r*nt*d
    yts = []  # shape r*nt
    for r in trange(repeat_times, desc=f'Training RR for best_gamma of {ds.name}, Repeat {repeat_times} time'):
        ds.reset(r)
        rrs.append(RR(d=ds.d, ell=ds.d))
        train(rrs[-1], ds, batch_size=ds.d)
        Xt, yt = ds.testData()
        Xts.append(Xt)
        yts.append(yt)
    Xts = np.array(Xts)  # shape r*nt*d
    yts = np.array(yts)  # shape r*nt
    # Try gammas
    print('Trying gamma: ', end='')
    gamma_list = []
    pred_error_list = []
    gamma = start
    n_pass_best_gamma = 0
    best_error = float("inf")
    while n_pass_best_gamma < 4:
        print(gamma, end='')
        gamma_list.append(gamma)
        coefs = query(rrs, gamma)  # shape r*d
        yps = pred(coefs, Xts)  # shape r*nt
        pred_error = np.mean((yps-yts)**2)
        pred_error_list.append(pred_error)
        if pred_error < best_error:
            best_error = pred_error
        else:
            n_pass_best_gamma += 1
        gamma *= 2
    print()
    # Save results
    np.savetxt(os.path.join(out_path, 'gammas.txt'), 
            np.array([gamma_list, pred_error_list]), 
            footer='row1: gamma, row2: mean_pred_error')
    return gamma_list[np.argmin(pred_error_list)]


def data_only(ds, out_path, repeat_times=5):
    """Data only experiments. Trainning data fetching time for different batch size.
    
    Args:
        ds (instance): .datasets.*
        out_path (str): Path to save gammas.txt
        repeat_times (int, optional): Defaults to 5.
 
    Outputs:
        './data_name/data_only/ells.txt': (numpy.savetxt): fmt('%d'), shape l
        './data_name/data_only/train_time.txt': (numpy.savetxt): fmt('%.18e'), shape l
    """
    out_path = os.path.join(out_path, 'data_only/')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    p = int(math.log2(ds.d))
    ells = np.array([2**w for w in range(0, p+3)], dtype=np.int)
    train_time_list = []
    for ell in tqdm(ells, desc=f'{ds.name} data time for {len(ells)} different batch sizes'):
        t_list = []
        for r in trange(repeat_times, desc=f'Batch size {ell}. Repeat {repeat_times} time'):
            ds.reset(r)
            start_time = time.time()
            train(None, ds, batch_size=ell)
            t_list.append(time.time() - start_time)
        train_time_list.append(np.mean(t_list))
    # Save results
    np.savetxt(os.path.join(out_path, 'ells.txt'), ells, fmt='%d')
    np.savetxt(os.path.join(out_path, 'train_time.txt'), np.array(train_time_list))


def base_line(ds, out_path, gamma, repeat_times=5):
    """RR experiments.
    
    Args:
        ds ([type]): [description]
        out_path ([type]): [description]
        gamma (number): Choosen gamma value.
        repeat_times (int, optional): Defaults to 5.
 
    Outputs:
        './data_name/RR/ells.txt': (numpy.savetxt): fmt('%d'), shape l
        './data_name/RR/train_time.txt': (numpy.savetxt): fmt('%.18e'), shape l
        './data_name/RR/query_time.txt': (numpy.savetxt): fmt('%.18e'), shape l
        './data_name/RR/coefs.txt': (numpy.savetxt): fmt('%.18e'), shape r*d
        './data_name/RR/pred_error.txt': (numpy.savetxt): fmt('%.18e'), shape r
   """
    data_time = np.loadtxt(os.path.join(out_path, 'data_only/', 'train_time.txt'))  # shape l
    data_ell = np.loadtxt(os.path.join(out_path, 'data_only/', 'ells.txt'), dtype=int)  # shape l
    # RR with 1 and d batch size.
    out_path = os.path.join(out_path, 'RR/')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    ells = np.array([1, ds.d], dtype=np.int)
    results = run_algs(ds, RR, ells, gamma, repeat_times=repeat_times)
    train_time, query_time, coefs, pred_errors = results
    train_time -= data_time[np.isin(data_ell, ells)]
    # Check coefs and pred_errors are same for different batch size
    for i in range(len(pred_errors)-1):
        np.testing.assert_almost_equal(coefs[i], coefs[i+1])
        np.testing.assert_almost_equal(pred_errors[i], pred_errors[i+1])
    # Save results
    np.savetxt(os.path.join(out_path, 'ells.txt'), ells, fmt='%d')
    np.savetxt(os.path.join(out_path, 'train_time.txt'), 
            train_time)
    np.savetxt(os.path.join(out_path, 'query_time.txt'), 
            query_time)
    np.save(os.path.join(out_path, 'coefs.npy'), coefs[0])
    np.savetxt(os.path.join(out_path, 'pred_error.txt'), 
            pred_errors)
        

def analysis_algs(ds, out_path, gamma, repeat_times=5):
    data_time = np.loadtxt(os.path.join(out_path, 'data_only/', 'train_time.txt'))  # shape l
    data_ell = np.loadtxt(os.path.join(out_path, 'data_only/', 'ells.txt'), dtype=int)  # shape l
    p = int(math.log2(ds.d))
    algs1 = [FFDRR, iSVDRR, RFDRR, NOFDRR]
    ells1 = np.array([2**w for w in range(p-6, p)], dtype=np.int)
    results = run_algs(ds, algs1, ells1, gamma, repeat_times=repeat_times)
    train_time1, query_time1, coefs1, pred_errors1 = results
    algs2 = [RPRR, CSRR]
    ells2 = np.array([2**w for w in range(p-3, p+3)], dtype=np.int)
    results = run_algs(ds, algs2, ells2, gamma, repeat_times=repeat_times)
    train_time2, query_time2, coefs2, pred_errors2 = results
    # - data_time
    train_time1 -= data_time[np.isin(data_ell, ells1)]
    train_time2 -= data_time[np.isin(data_ell, ells2)]
    # combine results of 6 algs
    algs = algs1 + algs2  # concatenate 2 lists
    ells = np.concatenate((np.tile(ells1, (len(algs1), 1)), np.tile(ells2, (len(algs2), 1))))  # shape a*l
    train_time = np.concatenate((train_time1, train_time2))  # a*l
    query_time = np.concatenate((query_time1, query_time2))  # a*l
    coefs = np.concatenate((coefs1, coefs2))  # a*l*r*d
    pred_errors = np.concatenate((pred_errors1, pred_errors2))  # a*l
    # coef_errors
    opt_coef = np.load(os.path.join(out_path, 'RR/', 'coefs.npy'))  # shape r*d
    coef_errors = np.linalg.norm(coefs - opt_coef, axis=-1) / np.linalg.norm(opt_coef, axis=-1)  # shape: a*l*r
    coef_errors = coef_errors.mean(axis=-1)  # shape: a*l
    # Save results
    for i in range(len(algs)):
        alg_path = os.path.join(out_path, f'{algs[i].name}/')
        if not os.path.exists(alg_path):
            os.makedirs(alg_path)
        np.savetxt(os.path.join(alg_path, 'ells.txt'), ells[i], fmt='%d')
        np.savetxt(os.path.join(alg_path, 'train_time.txt'), 
                train_time[i])
        np.savetxt(os.path.join(alg_path, 'query_time.txt'), 
                query_time[i])
        np.save(os.path.join(alg_path, 'coef_error.npy'), coef_errors[i])
        np.savetxt(os.path.join(alg_path, 'pred_error.txt'), 
                pred_errors[i])


def run_algs(ds, algs, ells, gamma, repeat_times=5):
    """Run 1 or list of alg with dataset, repeat `repeat_times` times.
    
    Args:
        ds: 1 dataset
        algs (list of rr algorithm): shape a
        ells (integer): list of ell, shape l
        gamma (number): Choosen gamma value.
        repeat_times (int, optional): Defaults to 5.

    Returns:
        train_time_list (numpy.ndarray): dtype('float'), shape: a*l
        query_time_list (numpy.ndarray): dtype('float'), shape: a*l
        coefs_list (numpy.ndarray): dtype('float'), shape a*l*r*d
        pred_error_list (numpy.ndarray): dtype('float'), shape a*l.
    """
    algs_is_list = isinstance(algs, Iterable)
    if not algs_is_list:
        algs = [algs]
    train_time_list = []
    query_time_list = []
    coefs_list = []
    pred_error_list = []
    for alg in tqdm(algs, desc=f'{ds.name} with {len(algs)} algorithms'):
        train_time_list.append([])
        query_time_list.append([])
        coefs_list.append([])
        pred_error_list.append([])
        for ell in tqdm(ells, desc=f'{alg.name} with {len(ells)} batch sizes', leave=False):
            train_time_list[-1].append([])
            query_time_list[-1].append([])
            coefs_list[-1].append([])
            pred_error_list[-1].append([])
            for r in trange(repeat_times, desc=f'Batch_size {ell}, Repeat {repeat_times} time', leave=False):
                ds.reset(r)
                start_time = time.time()
                rr = alg(d=ds.d, ell=ell)
                train(rr, ds, ell)
                train_time = time.time() - start_time
                start_time = time.time()
                coefs = query(rr, gamma)  # shape d
                query_time = time.time() - start_time
                Xt, yt = ds.testData()
                yp = pred(coefs, Xt)  # shape n_test
                pred_error = np.mean((yp-yt)**2)
                train_time_list[-1][-1].append(train_time)
                query_time_list[-1][-1].append(query_time)
                coefs_list[-1][-1].append(coefs)
                pred_error_list[-1][-1].append(pred_error)
    train_time_list = np.mean(train_time_list, axis=-1)
    query_time_list = np.mean(query_time_list, axis=-1)
    coefs_list = np.array(coefs_list)
    pred_error_list = np.mean(pred_error_list, axis=-1)
    if not algs_is_list:
        train_time_list = train_time_list[0]
        query_time_list = query_time_list[0]
        coefs_list = coefs_list[0]
        pred_error_list = pred_error_list[0]
    return train_time_list, query_time_list, coefs_list, pred_error_list


def dVtime(n_p=12, ell_p=4, d_p=15, out_path='./output/dVtime/', repeat_times=2):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    effective_rank = 2**ell_p
    ell = 2**ell_p
    make_data_params = dict(n_features=2**(n_p-2),
                            n_samples=2**n_p,
                            eval_size=2**(n_p-2),
                            test_size=2**(n_p-2),
                            effective_rank=effective_rank,
                            noise=4,
                            correlation=1,
                            random_state=0,
                            rotate='dct4',
                            name='synthetic')
    d_list = np.array(range(ell_p+1, d_p))
    np.savetxt(os.path.join(out_path, 'ds.txt'), 2**d_list, fmt='%d')
    # data only
    train_time_list = []  # shape len(d_list)
    for d_p in tqdm(d_list, desc=f'data only, {len(d_list)} ds'):
        make_data_params['n_features'] = 2**d_p
        ds = LowRankRegression(**make_data_params)
        t_list = []
        for r in trange(repeat_times, desc=f'd={ds.d}. Repeat {repeat_times} time'):
            ds.reset(r)
            rr = None
            start_time = time.time()
            train(rr, ds, batch_size=ell)
            t_list.append(time.time() - start_time)
        train_time_list.append(np.mean(t_list))
    data_time = np.array(train_time_list)  # shape len(d_list)
    # run
    train_time_list = []
    query_time_list = []
    algs = [RR, iSVDRR, FFDRR, RFDRR, NOFDRR, RPRR, CSRR]
    for d_p in tqdm(d_list, desc=f'{len(algs)} algs, {len(d_list)} ds'):
        make_data_params['n_features'] = 2**d_p
        ds = LowRankRegression(**make_data_params)
        results = run_algs(ds, algs, [ell], 1024, repeat_times=repeat_times)
        train_time, query_time, coefs, pred_errors = results
        train_time = np.squeeze(train_time, axis=-1)  # shape len(algs)
        query_time = np.squeeze(query_time, axis=-1)  # shape len(algs)
        train_time_list.append(train_time)
        query_time_list.append(query_time)
    train_times = np.array(train_time_list).T  # shape len(algs)*len(d_list)
    query_times = np.array(query_time_list).T  # shape len(algs)*len(d_list)
    train_times -= data_time  # minus data time
    for i in range(0, len(algs)):
        np.savetxt(os.path.join(out_path, f'{algs[i].name}.txt'), np.stack((train_times[i], query_times[i])), footer='row1: train_time; row2: query_time')


def nVtime(d_p=10, ell_p=4, n_p=15, out_path='./output/nVtime/', repeat_times=2):
    gamma=1
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    effective_rank = 2**ell_p
    ell = 2**ell_p
    make_data_params = dict(n_features=2**d_p,
                            n_samples=2**n_p,
                            eval_size=2**d_p,
                            test_size=2**d_p,
                            effective_rank=effective_rank,
                            noise=4,
                            correlation=1,
                            random_state=0,
                            rotate='dct4',
                            name='synthetic')
    algs = [RR, iSVDRR, FFDRR, RFDRR, NOFDRR, RPRR, CSRR]
    train_time_list = []
    query_time_list = []
    for r in range(repeat_times):
        train_time_list.append([])
        query_time_list.append([])
        ds = LowRankRegression(**make_data_params)
        rrs = []
        for alg in algs:
            rrs.append(alg(ds.d, ell))
            train_time_list[-1].append([0])
            query_time_list[-1].append([0])
        n_batch = math.ceil(ds.n_train / ell)
        for b in trange(n_batch, desc=f'nVtime, {n_batch} batchs.'):
            X, y = ds.trainData(ell)
            for i in range(len(rrs)):
                start_time = time.time()
                rrs[i].fit(X, y)
                train_time_list[-1][i].append(train_time_list[-1][i][-1]+time.time()-start_time)
                start_time = time.time()
                coefs = query(rrs[i], gamma)
                query_time_list[-1][i].append(query_time_list[-1][i][-1]+time.time()-start_time)
    train_times = np.array(train_time_list).mean(axis=0)  # shape a*b
    query_times = np.array(query_time_list).mean(axis=0)  # shape a*b
    for i in range(0, len(rrs)):
        np.savetxt(os.path.join(out_path, f'{rrs[i].name}.txt'), np.stack((train_times[i], query_times[i])), footer='row1: train_time; row2: query_time')


def algs_time_error(p=11, out_path='./output/algs_time_error/', random_state=0, repeat_times=10):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    dataset_dict = make_datasets(p=p, r=random_state)
    for ds in dataset_dict.values():
        print(f'Start dataset "{ds.name}"')
        ds_path = os.path.join(out_path, f'{ds.name}/')
        if not os.path.exists(ds_path):
            os.makedirs(ds_path)
        data_summary(dataset=ds, out_path=ds_path, repeat_times=repeat_times)
        gamma = best_gamma(ds=ds, out_path=ds_path, repeat_times=repeat_times, start=1)
        print(f'best gamma: {gamma}')
        data_only(ds=ds, out_path=ds_path, repeat_times=repeat_times)
        base_line(ds=ds, out_path=ds_path, gamma=gamma, repeat_times=repeat_times)
        analysis_algs(ds=ds, out_path=ds_path, gamma=gamma, repeat_times=repeat_times)


def analysis(out_path='./output/', dp=11, random_state=0, repeat_times=10):
    p = dp  # d = 2**p
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    algs_time_error(p=11, out_path=os.path.join(out_path, 'algs_time_error/'),
                    random_state=random_state, repeat_times=10)
    dVtime(n_p=8, ell_p=6, d_p=15, out_path=os.path.join(
        out_path, 'dVtime/'), repeat_times=repeat_times)
    nVtime(d_p=11, ell_p=6, n_p=15, out_path=os.path.join(
        out_path, 'nVtime/'), repeat_times=repeat_times)


if __name__ == '__main__':
    analysis(out_path='./output/', dp=11, random_state=0, repeat_times=10)
