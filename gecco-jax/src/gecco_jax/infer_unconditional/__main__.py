import math
import argparse
import os
import jax
import numpy as np
import torch
import equinox as eqx
from tqdm.auto import tqdm

import gecco_jax

def execute(args):
    config = gecco_jax.load_config(args.config_path)
    key = jax.random.PRNGKey(args.seed)
    model = config.make_model(key=key)
    
    model = eqx.tree_deserialise_leaves(
        args.checkpoint_path,
        like=model,
    )
    model = eqx.tree_inference(model, value=True)
    model = eqx.tree_at(
        where=lambda model: model.schedule.n_solver_steps,
        pytree=model,
        replace=args.n_solver_steps,
    )
    
    n_batches = int(math.ceil(args.n_samples / args.batch_size))

    if args.sampler == 'ode':
        sample_fn = lambda key: model.sample(
            (args.n_points, 3),
            None,
            n=args.batch_size,
            key=key,
        )
    elif args.sampler == 'sde':
        sample_fn = lambda key: model.sample_stochastic(
            (args.n_points, 3),
            None,
            n=args.batch_size,
            key=key,
            s_churn=args.sde_churn,
        )
    
    samples = []
    for key in tqdm(jax.random.split(key, n_batches)):
        smp = sample_fn(key)
        samples.append(np.asarray(smp))
    samples = np.concatenate(samples, axis=0)
    torch.save(torch.from_numpy(samples), args.save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('checkpoint_path', type=str)
    parser.add_argument('--save-path', type=str, default=None)
    parser.add_argument('--n-points', type=int, default=2048)
    parser.add_argument('--n-solver-steps', type=int, default=128)
    parser.add_argument('--n-samples', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--sampler', choices=['ode', 'sde'], default='ode')
    parser.add_argument('--sde-churn', type=float, default=0.5)

    args = parser.parse_args()

    assert args.checkpoint_path.endswith('.eqx')

    if args.save_path is None:
        args.save_path = f'{args.checkpoint_path}_samples.pt'

    execute(args)

if __name__ == '__main__':
    main()
