import argparse
import json
import os
import ConfigSpace
from hpbandster.optimizers.bohb import BOHB
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pickle
import numpy as np
import random
from copy import deepcopy
import GPyOpt
from objective_funcs.objective import TuneNN
from objective_funcs.sync_objective import sync_func
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', default="egg", type=str, help='specifies the target task name')
parser.add_argument('--bo_method', default="tpe", type=str, help='bo method used: bohb, tpe, gpbo')
parser.add_argument('--n_iters', default=30, type=int, help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./results/", type=str,
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="../data/", type=str, help='specifies the path to the tabular data')
parser.add_argument('--n_init', type=int, default=10, help='number of initial data')
parser.add_argument('--seed', type=int, default=42, help='random seed')

args = parser.parse_args()

print(args)
np.random.seed(args.seed)
random.seed(args.seed)

# define the objective problem
# b = TuneNN(data_dir=args.data_dir, task=task_name, seed=args.seed, bo_method=args.bo_method)
b = sync_func(task=args.task_name, seed=args.seed, bo_method=args.bo_method)

# create the result saving path
output_dir = os.path.join(args.output_path, args.task_name)
os.makedirs(output_dir, exist_ok=True)
result_path = os.path.join(output_dir, f"{args.bo_method}s{args.seed}")


if args.bo_method == 'bohb':

    # ---- BO strategy A: BOHB ------

    # define the search space
    search_space = b.get_configuration_space()

    # modify the objective function format to apply this BO method
    class MyWorker(Worker):
        def compute(self, config, budget, **kwargs):
            y, cost = b.eval(config, budget=int(budget))
            return ({
                'loss': float(y),
                'info': float(cost)})

    # specify configurations of BOHB:
    hb_run_id = f'{args.seed}'
    min_bandwidth = 0.3
    num_workers = 1
    min_budget = 30
    max_budget = 90

    # initialise BOHB
    NS = hpns.NameServer(run_id=hb_run_id, host='localhost', port=0)
    ns_host, ns_port = NS.start()
    workers = []
    for i in range(num_workers):
        w = MyWorker(nameserver=ns_host, nameserver_port=ns_port,
                     run_id=hb_run_id, id=i)
        w.run(background=True)
        workers.append(w)

    bohb = BOHB(configspace=search_space, run_id=hb_run_id,
                eta=3, min_budget=min_budget, max_budget=max_budget,
                nameserver=ns_host, nameserver_port=ns_port,
                ping_interval=10, min_bandwidth=min_bandwidth)

    # run BOHB
    results = bohb.run(int(args.n_iters+args.n_init), min_n_workers=num_workers)
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    # process the returned results to give the same format
    bo_results = []
    curr_best = np.inf
    for item in results.data.items():
        key, datum = item
        budget = datum.budget
        hyperparams = datum.config
        object_values = datum.results[budget]['loss']
        if object_values < curr_best:
            curr_best = object_values
            curr_best_hyperparam = hyperparams

        if budget == max_budget:
            bo_results.append((hyperparams, object_values))

    print(f'Best hyperparams={curr_best_hyperparam} with objective value={curr_best}')

elif args.bo_method == 'tpe':

    # ------ BO strategy B: TPE ------

    # define the search space
    search_space = b.get_search_space()

    # initialise and run TPE
    trials = Trials()
    best_hyperparam = fmin(b.eval,
                space=search_space,
                algo=tpe.suggest,
                max_evals=int(args.n_iters+args.n_init),
                trials=trials)
    # process the returned results to give the same format
    bo_results = [(item['config'], item['loss']) for item in trials.results]
    best_objective_value = np.min([y['loss'] for y in trials.results])
    print(f'Best hyperparams={best_hyperparam} with objective value={best_objective_value}')

elif args.bo_method == 'gpbo':

    # ------ BO strategy C: GPyOpt ------

    # define the search space
    search_space = b.get_search_space()

    # initialise and run GPyOpt
    gpyopt = GPyOpt.methods.BayesianOptimization(f=b.eval, domain=search_space,
                                                initial_design_numdata=args.n_init,
                                                acquisition_type='EI', model_type='GP',
                                                model_update_interval=5, verbosity=True,
                                                normalize_Y=False)
    gpyopt.run_optimization(max_iter=args.n_iters)

    # process the returned results to give the same format
    Y_queried = gpyopt.Y
    X_queried = gpyopt.X
    best_hyperparam = np.atleast_2d(gpyopt.x_opt)
    best_objective_value = gpyopt.Y_best[-1]

    bo_results = []
    for j, x in enumerate(X_queried):
        bo_results.append((x, Y_queried[j]))

    print(f'Best hyperparams={best_hyperparam} with objective value={best_objective_value}')


pickle.dump(bo_results, open(result_path, 'wb'))
