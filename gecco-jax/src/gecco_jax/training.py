import os
import contextlib
from dataclasses import dataclass, field
import math
import shutil
import time
from collections import defaultdict
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import warnings

import jax
import optax
import equinox as eqx
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

from gecco_jax.config import latest_checkpoint
from gecco_jax.models import Diffusion
from gecco_jax.types import NaNError, PyTree, Example
from gecco_jax.models.util import splitter
from gecco_jax.metrics import Metric, LossMetric, LogpMetric, MetricPmapWrapper
from gecco_jax.config import CHECKPOINT_SAVE_RE, CHECKPOINT_SAVE_TEMPLATE

@eqx.filter_jit
def check_all_finite(model):
    def _is_finite_fn(carry, value):
        if not eqx.is_inexact_array(value):
            return carry
        return carry or jax.numpy.isfinite(value).all()

    return jax.tree_util.tree_reduce(
        _is_finite_fn,
        model,
        initializer=True,
    )

def unshard(pytree):
    return jax.tree_map(
        lambda maybe_array: maybe_array[0] if eqx.is_array(maybe_array) else maybe_array,
        pytree,
    )

def shard_key(key):
    keys = jax.random.split(key, jax.device_count())
    return jax.device_put_sharded(list(keys), jax.devices())

def _is_replicated(pytree):
    def _is_replicated_one(value: Any) -> bool:
        if not isinstance(value, jax.Array):
            return False
        return len(value.sharding.device_set) > 1

    return jax.tree_util.tree_reduce(
        lambda carry, maybe_arr: carry or _is_replicated_one(maybe_arr),
        pytree,
        False,
    )

def replicate_pytree(pytree):
    def replicate_one(maybe_array):
        if not eqx.is_array(maybe_array):
            return maybe_array
        return jax.device_put_replicated(maybe_array, jax.devices())
    return jax.tree_map(replicate_one, pytree)

class MockWriter:
    def __getattr__(self, name):
        assert name.startswith('add_')
        
        def mock_add_fn(tag: str, *, global_step: int, **kwargs):
            assert len(kwargs) >= 1 # at least one value-like argument
            assert isinstance(tag, str)
            assert isinstance(global_step, int)
            
        return mock_add_fn

def enumerate_from(iterable, from_: int):
    for id, element in enumerate(iterable):
        yield id + from_, element

class Stepper:
    def __init__(self, make_step):
        self.make_step = make_step
        self.inner = None
        self.ostatic = None
        self.otreedef = None
    
    def __call__(self, model, x, ctx, key, opt_state, ema_state):
        input_tree = (model, x, ctx, key, opt_state, ema_state)

        iparams, istatic = eqx.partition(input_tree, eqx.is_array)
        ileaves, itreedef = jax.tree_util.tree_flatten(iparams)

        if self.ostatic is None or self.otreedef is None:
            output_tree = (jax.numpy.array(0.), model, opt_state, ema_state)
            oparams, self.ostatic = eqx.partition(output_tree, eqx.is_array)
            _oleaves, self.otreedef = jax.tree_util.tree_flatten(oparams)

        if self.inner is None:
            def inner(leaves):
                params = jax.tree_util.tree_unflatten(itreedef, leaves)
                args = eqx.combine(params, istatic)
                output_tree = self.make_step(*args)

                params, _static = eqx.partition(output_tree, eqx.is_array)
                leaves, _treedef = jax.tree_util.tree_flatten(params)
                return leaves
            self.inner = jax.pmap(inner, axis_name='device')

        oleaves = self.inner(ileaves)
        oparams = jax.tree_util.tree_unflatten(self.otreedef, oleaves)
        return eqx.combine(oparams, self.ostatic)

@dataclass
class Trainer:
    model: Diffusion
    train_dataloader: Iterable[Example]
    val_dataloader: Union[Iterable[Example], List[Iterable[Example]]]
    save_path: str
    save_every: int = 100_000
    num_steps: int = 1_000_000
    metrics: Sequence[Metric] = (LogpMetric(), )
    optim: Callable = optax.adabelief(learning_rate=3e-4)
    loss_scale: float = 1.0
    ema_alpha: float = 0.999
    n_validation_batches: Optional[int] = None
    callbacks: Iterable[Callable] = ()
    check_for_nan: bool = False
    seed: int = 5678
    profile_path: Optional[str] = None
    train_in_inference_mode: bool = False
    skip_smoke_test: bool = False
    devices: List[Any] = field(default_factory=jax.local_devices)
    initial_step_number: int = 0
    current_best_metric: Dict[str, Tuple[int, float]] = field(default_factory=dict)
    force_pmap: bool = False
    keep_all_checkpoints: bool = False

    ema_model: eqx.Module = None
    opt_state: Optional[PyTree] = None
    _single_loss_fn: Optional[Callable] = None
    loss_fn: Optional[Callable] = None
    val_key: Optional[jax.random.PRNGKey] = None
    train_key: Optional[jax.random.PRNGKey] = None

    def __post_init__(self):
        print(f'Trainer save_path={self.save_path}.')
        keys = splitter(jax.random.PRNGKey(self.seed))

        if not isinstance(self.model, eqx.Module):
            assert hasattr(self.model, '__call__'), self.model
            self.model = self.model(key=next(keys))

        self._single_loss_fn = partial(
            type(self.model).batch_loss_fn,
            loss_scale=self.loss_scale,
        )

        self.metrics = self.metrics + (LossMetric(self.loss_scale), )
        
        if self.should_pmap:
            self.metrics = tuple(map(MetricPmapWrapper, self.metrics))

        self.loss_fn = self._single_loss_fn

        self.val_key = next(keys)
        self.train_key = next(keys)

        os.makedirs(
            self._metric_save_dir,
            exist_ok=True,
        )
    
    @property
    def should_pmap(self) -> bool:
        return len(self.devices) > 1 or self.force_pmap

    def _maybe_unshard(self, pytree: PyTree) -> PyTree:
        if self.should_pmap:
            return unshard(pytree)
        else:
            return pytree

    def _maybe_shard_key(self, key):
        if self.should_pmap:
            return shard_key(key)
        else:
            return key
    
    def _replicate_model(self):
        if _is_replicated(self.model):
            raise AssertionError('Model is already replicated!')

        self.model = replicate_pytree(self.model)
        self.loss_fn = eqx.filter_pmap(self._single_loss_fn)
    
    def save(self, dirname: str):
        checkpoint_dir = os.path.join(self.save_path, dirname)
        if os.path.exists(checkpoint_dir):
            warnings.warn(f'Seemingly overwriting contents of {dirname=}')
        os.makedirs(checkpoint_dir)

        eqx.tree_serialise_leaves(
            os.path.join(checkpoint_dir, 'ema.eqx'),
            self._maybe_unshard(self.ema_model),
        )
        eqx.tree_serialise_leaves(
            os.path.join(checkpoint_dir, 'opt.eqx'),
            self._maybe_unshard(self.opt_state),
        )
        eqx.tree_serialise_leaves(
            os.path.join(checkpoint_dir, 'model.eqx'),
            self._maybe_unshard(self.model),
        )
    
    def load(self, dirname: str):
        self.ema_model = eqx.tree_deserialise_leaves(
            os.path.join(dirname, 'ema.eqx'),
            like=self.ema_model,
        )
        self.opt_state = eqx.tree_deserialise_leaves(
            os.path.join(dirname, 'opt.eqx'),
            like=self.opt_state,
        )
        self.model = eqx.tree_deserialise_leaves(
            os.path.join(dirname, 'model.eqx'),
            like=self.model,
        )
        print(f'Loaded from {dirname=}.')

    def _remove_old_checkpoints(self, step: int):
        ''' remove checkpoints older than `step` '''
        for path in os.listdir(self.save_path):
            if (m := CHECKPOINT_SAVE_RE.match(path)) is None:
                continue

            checkpoint_step = int(m.group(1))
            if checkpoint_step < step:
                shutil.rmtree(os.path.join(self.save_path, path))

    def recover_from_checkpoint(
        self,
        fail_if_unavailable: bool = False,
    ):
        try:
            checkpoint_path, start_step = latest_checkpoint(
                self.save_path,
                return_step_number=True,
            )
        except IOError:
            if fail_if_unavailable:
                print('No checkpoint found, exiting')
                raise 
            else:
                print('No checkpoint found, starting from scratch')
                return self

        self.load(checkpoint_path)
        self.initial_step_number = start_step + 1
        return self

    @property
    def inference_model(self):
        return eqx.tree_inference(self.ema_model, value=True)
        
    def _torch_to_jax(self, data: Example):
        return data.torch_to('pmap' if self.should_pmap else 'jnp')

    def _metrics_single_dataset(
        self,
        dataloader: Iterable[Example],
        n_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        model = self.inference_model
        outputs = defaultdict(lambda : list())
        key = self.val_key

        for val_step, data in enumerate(tqdm(dataloader, total=n_batches)):
            xyz, raw_ctx, _ = self._torch_to_jax(data.discard_extras())
            key, *keys = jax.random.split(key, len(self.metrics) + 1)
            
            for metric_fn, metric_key in zip(self.metrics, keys):
                metric_values = metric_fn(model, xyz, raw_ctx, metric_key)

                for subname, value in metric_values.items():
                    # flatten just in case, to not break concatenate
                    outputs[f'{metric_fn.name}/{subname}'].append(np.asarray(value).flatten())

            if n_batches is not None and (val_step == n_batches):
                break
        
        return {k: np.mean(np.concatenate(v)) for k, v in outputs.items()}
    
    def metrics_loop(
        self,
        n_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        if isinstance(self.val_dataloader, (list, tuple)):
            metrics = {}
            assert all(hasattr(loader, 'name') for loader in self.val_dataloader)

            for subset in self.val_dataloader:
                submetrics = self._metrics_single_dataset(
                    subset,
                    n_batches=n_batches,
                )
                metrics.update(**{f'{subset.name}/{k}': v for k, v in submetrics.items()})
            return metrics
        else:
            return self._metrics_single_dataset(
                self.val_dataloader,
                n_batches=n_batches,
            )
    
    def validation_phase(
        self,
        step: int,
        logger: Union[SummaryWriter, MockWriter],
        _smoke_test=False,
    ):
        if _smoke_test:
            n_batches = 2
        else:
            n_batches = self.n_validation_batches

        metrics = self.metrics_loop(n_batches=n_batches)

        val_phase_id = step // self.save_every
        for k, v in metrics.items():
            logger.add_scalar(
                f'val-means/{k}',
                scalar_value=v,
                global_step=val_phase_id,
            )

            self._maybe_save_best_metric(k, v, step, _smoke_test)

        inference_model_unsharded = self._maybe_unshard(self.inference_model)
        for callback in self.callbacks:
            callback(
                model=inference_model_unsharded,
                logger=logger,
                epoch=val_phase_id,
            )
    
    def _maybe_save_best_metric(
        self,
        metric_key: str,
        metric_value: float,
        step: int,
        _smoke_test: bool,
    ):
        if ('chamfer_distance' not in metric_key) and ('logp/total' not in metric_key):
            # FIXME: ugly hardcoding
            return 
        
        if metric_key in self.current_best_metric:
            best_step, best_value = self.current_best_metric[metric_key]
            # FIXME: ugly hardcoding
            higher_is_better = 'logp' in metric_key.lower()
            if higher_is_better:
                current_is_better = metric_value > best_value
            else:
                current_is_better = metric_value < best_value

            if current_is_better:
                path_to_delete = self._metric_save_path(metric_key, best_step)
                path_to_create = self._metric_save_path(metric_key, step)
                self.current_best_metric[metric_key] = (step, metric_value)
            else:
                path_to_delete = None
                path_to_create = None
        else:
            path_to_delete = None
            path_to_create = self._metric_save_path(metric_key, step)
            self.current_best_metric[metric_key] = (step, metric_value)
        
        if _smoke_test:
            assert path_to_delete == None
            # this way we will delete the checkpoint immediately
            path_to_delete = path_to_create
            del self.current_best_metric[metric_key]
        
        if path_to_create is not None:
            print(f'Saving new best score at {path_to_create=}.')
            self.save(self._metric_save_path(metric_key, step))
        if path_to_delete is not None:
            shutil.rmtree(path_to_delete)
        
    @property
    def _metric_save_dir(self) -> str:
        return os.path.join(self.save_path, 'best-checkpoints')

    def _metric_save_path(self, metric_key: str, metric_step: int) -> str:
        key_no_under = metric_key.replace('/', '__')
        return os.path.join(self._metric_save_dir, f'{key_no_under}-step-{metric_step}')

    def train_step(
        self,
        data: Example,
        step: int,
        logger,
    ):
        xyz, raw_ctx, _ = self._torch_to_jax(data.discard_extras())

        step_key, self.train_key = jax.random.split(self.train_key, 2)
        step_keys = self._maybe_shard_key(step_key)

        loss_value, new_model, new_opt_state, new_ema_model = self._jit_make_step(
            self.model,
            xyz,
            raw_ctx,
            step_keys,
            self.opt_state,
            self.ema_model,
        )

        loss_value = float(loss_value.mean())

        self.model = new_model
        self.opt_state = new_opt_state
        self.ema_model = new_ema_model

        logger.add_scalar(
            'train/loss',
            scalar_value=loss_value,
            global_step=step,
        )

        return loss_value

        
    def fit(self):
        if self.should_pmap and not _is_replicated(self.model):
            print('Replicating model')
            self._replicate_model()
        else:
            print('Not replicating model')

        self.ema_model = self.model

        # potentially turn on training mode
        self.model = eqx.tree_inference(
            self.model,
            value=not self.train_in_inference_mode,
        )

        step_kw = dict(
            opt_update=self.optim.update,
            loss_scale=self.loss_scale,
            ema_alpha=self.ema_alpha,
        )

        if self.should_pmap:
            opt_init = jax.pmap(self.optim.init)
            make_step_single = partial(
                self.model.make_step,
                **step_kw,
                is_distributed=True,
            )
            self._jit_make_step = Stepper(make_step_single)
        else:
            opt_init = self.optim.init
            make_step = partial(self.model.make_step, **step_kw, is_distributed=False)
            self._jit_make_step = eqx.filter_jit(make_step)

        self.opt_state = opt_init(eqx.filter(self.model, eqx.is_inexact_array))

        if self.profile_path is None and not self.skip_smoke_test:
            print('Running a test validation phase...')
            self.validation_phase(step=0, logger=MockWriter(), _smoke_test=True)
            print('Success.')
        else:
            print('Skipping validation smoke test.')

        loss_ema = None
        loss_avg = 0
        pbar = tqdm(self.train_dataloader, total=self.num_steps, mininterval=10.0, maxinterval=15.0)
        logger = SummaryWriter(f'{self.save_path}/tensorboard')
        try:
            with pbar, logger:
                for step, data in enumerate_from(pbar, self.initial_step_number):
                    if self.profile_path is not None and step == 20:
                        jax.profiler.start_trace(self.profile_path)
                        profile_start = time.perf_counter()

                    if self.profile_path is not None:
                        step_ctx = jax.profiler.StepTraceAnnotation("train", step_num=step)
                    else:
                        step_ctx = contextlib.nullcontext()

                    with step_ctx:
                        loss_value = self.train_step(data, step, logger)
                    
                    if not math.isfinite(loss_value):
                        raise NaNError("NaN loss")

                    loss_avg += (loss_value - loss_avg) / (step + 1)
                    if loss_ema is None:
                        loss_ema = loss_value
                    else:
                        loss_ema = loss_value * 0.1 + loss_ema * 0.9

                    pbar.set_postfix(loss=loss_ema)

                    if step % self.save_every == (self.save_every - 1):
                        self.save(os.path.join(self.save_path, CHECKPOINT_SAVE_TEMPLATE.format(step)))
                        logger.add_scalar(
                            'train/mean_loss',
                            scalar_value=loss_avg,
                            global_step=step // self.save_every,
                        )
                        self.validation_phase(step=step, logger=logger)
                        if step > self.save_every and not self.keep_all_checkpoints:
                            self._remove_old_checkpoints(step)
                        
                    if step == self.num_steps:
                        break

                    if self.profile_path is not None and step == 25:
                        profile_end = time.perf_counter()
                        print(f'Timed segment elapsed {profile_end - profile_start}s.')
                        jax.profiler.stop_trace()
                        return
        except Exception as e:
            if not isinstance(e, KeyboardInterrupt):
                torch.save(data, f'{self.save_path}/offending-data.pth') 
                raise
        finally:
            self.save(f'final-checkpoint-{step}')
            print('Saved final checkpoint.')
    
def train(*args, recover_from_checkpoint: bool = True, **kwargs): # for backwards compatibility
    trainer = Trainer(*args, **kwargs)
    if recover_from_checkpoint:
        trainer = trainer.recover_from_checkpoint()
    
    trainer.fit()

    return trainer
