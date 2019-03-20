import os

from skopt import load
from skopt import callbacks
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver

from hyperspace import create_hyperspace


def run_hyperspace(objective_function, search_bounds, 
                   network_config, hyper_params, params, args):
    """Launch Hyperspace on Summit.

    Parameters
    ----------
    search_bounds : list of tuples/lists

    objective_function : function

    hyper_params : dict

    params : dict

    args : argparse object
    """
    hspace = create_hyperspace(search_bounds)
    space = hspace[args.jobid]

    checkpoint_file = os.path.join(args.hyperspace_results_path, f'hyperspace{args.jobid}')
    checkpoint_saver = CheckpointSaver(checkpoint_file, compress=9)

    try:
        res = load(checkpoint_file)
        x0 = res.x_iters
        y0 = res.func_vals
    except FileNotFoundError:
        print(f'No previous save point for space hyperspace{args.jobid}')
        # Need to randomly sample the bounds to prime the optimization.
        x0 = space.rvs(1)
        y0 = None

    gp_minimize(
        lambda x: objective_function(
            x, 
            network_config, 
            hyper_params, 
            params
        ),                              # the function to minimize
        space,                          # the bounds on each dimension of x
        x0=x0,                          # already examined values for x
        y0=y0,                          # observed values for x0
        acq_func="LCB",                 # the acquisition function (optional)
        n_calls=20,                     # the number of evaluations of f including at x0
        n_random_starts=0,              # the number of random initialization points
        callback=[checkpoint_saver],
        random_state=777
    )
