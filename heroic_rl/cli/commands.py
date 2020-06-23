"""
This module contains all available CLI commands.
"""

import os.path as osp
from functools import partial

import click

from ..inference.service import run as serve_inference
from ..inference.simulation import run as run_inference
from ..render.tui import run as run_render
from ..train import Brain, Plans, Rewards
from ..train.cfg import (
    ArchitectureCfg,
    HyperparametersCfg,
    MapCfg,
    SpellsCfg,
    TrainingCfg,
)
from ..train.experiment import TRAIN_CFG_FILENAME
from ..train.experiment import run as run_experiment
from .types import HOST_PORT


def _prefix_cb(ctx, param, value, prefix):
    if value is not None:
        if prefix not in ctx.params:
            ctx.params[prefix] = {}
        if isinstance(value, tuple):
            value = list(value)
        ctx.params[prefix][param.human_readable_name] = value


cmd = click.command
arg = click.argument

opt = partial(click.option, show_default=True)
hyper_opt = partial(
    click.option,
    show_default=True,
    callback=partial(_prefix_cb, prefix="hyperparameters"),
)
arch_opt = partial(
    click.option, show_default=True, callback=partial(_prefix_cb, prefix="architecture")
)
spells_opt = partial(
    click.option, show_default=True, callback=partial(_prefix_cb, prefix="spells")
)
map_opt = partial(
    click.option, show_default=True, callback=partial(_prefix_cb, prefix="map")
)


#########
# Train #
#########


@cmd()
@arg("exp_name", metavar="EXPERIMENT_NAME")
# General options
@opt(
    "-c",
    "--cpus",
    type=int,
    default=TrainingCfg.DEFAULT_CPUS,
    help="Number of subprocesses to spawn.",
)
@opt(
    "-s",
    "--steps",
    type=int,
    default=TrainingCfg.DEFAULT_STEPS,
    help="Number of steps of interaction (state-action pairs) "
    + "for the agent and the environment in each epoch.",
)
@opt(
    "-e",
    "--epochs",
    type=int,
    default=TrainingCfg.DEFAULT_EPOCHS,
    help="Number of epochs of interaction (equivalent to number of policy "
    + "updates) to perform.",
)
@opt(
    "-a",
    "--num-agents",
    type=int,
    default=TrainingCfg.DEFAULT_NUM_AGENTS,
    help="Number of agents to train. This can be used to simultaneously train "
    + "multiple agents, which can be useful with `selfplay` plan - agents will "
    + "play against each other.",
)
@opt(
    "-f",
    "--save-freq",
    "save_frequency_epochs",
    type=int,
    default=TrainingCfg.DEFAULT_SAVE_FREQUENCY_EPOCHS,
    help="How often (in epochs) to save agent state.",
)
@opt(
    "-b",
    "--batch-size",
    type=int,
    default=TrainingCfg.DEFAULT_BATCH_SIZE,
    help="Number of steps for single update step. Never greater than steps per "
    + "epoch per process (steps / number of processes). Bounded by total GPU "
    + "RAM.",
)
@opt(
    "-d",
    "--data-dir",
    type=click.Path(file_okay=False),
    default=TrainingCfg.DEFAULT_DATA_DIR,
    help="Path to data directory where all experiments are stored.",
)
@opt(
    "--epochs-per-agent",
    type=int,
    default=TrainingCfg.DEFAULT_EPOCHS_PER_AGENT,
    help="Number of epochs to train single agent for before yielding training "
    + "to next agent, relevant if there are multiple agents.",
)
@opt(
    "--server",
    "servers",
    type=HOST_PORT,
    multiple=True,
    default=TrainingCfg.DEFAULT_SERVERS,
    help="Address of training server(s). Multiple addresses may be specified.",
)
@opt(
    "--plan",
    type=click.Choice(Plans.all()),
    default=TrainingCfg.DEFAULT_PLAN,
    help="Training plan to use.",
)
@opt(
    "--reward",
    type=click.Choice(Rewards.all()),
    default=TrainingCfg.DEFAULT_REWARD,
    help="Reward function to use.",
)
@opt(
    "--seed",
    type=int,
    default=TrainingCfg.DEFAULT_SEED,
    help="Seed for random number generators.",
)
@opt(
    "--spatial-mask/--no-spatial-mask",
    "spatial_mask_static_enabled",
    default=TrainingCfg.DEFAULT_SPATIAL_MASK_STATIC_ENABLED,
    help="Whether to enable static spatial mask for spells (imposes casting rules).",
)
@opt(
    "--dynamic-spatial-mask/--no-dynamic-spatial-mask",
    "spatial_mask_dynamic_enabled",
    default=TrainingCfg.DEFAULT_SPATIAL_MASK_DYNAMIC_ENABLED,
    help="Whether to enable dynamic spatial mask for spells - modifies casting "
    + "rules according to observations.",
)
# Hyperparameter options
@hyper_opt(
    "--gamma",
    default=HyperparametersCfg.DEFAULT_GAMMA,
    type=float,
    help="Discount factor. (Always between 0 and 1.)",
)
@hyper_opt(
    "--lambda",
    "lam",
    default=HyperparametersCfg.DEFAULT_LAM,
    type=float,
    help="Lambda for GAE-Lambda. (Always between 0 and 1, close to 1.)",
)
@hyper_opt(
    "--target-kl",
    default=HyperparametersCfg.DEFAULT_TARGET_KL,
    type=float,
    help="Roughly what KL divergence we think is appropriate between new "
    + "and old policies after an update. This will get used for early "
    + "stopping. (Usually small, 0.01 or 0.05.)",
)
@hyper_opt(
    "--grad-clipping/--no-grad-clipping",
    "grad_clipping_enabled",
    default=HyperparametersCfg.DEFAULT_GRAD_CLIPPING_ENABLED,
    help="Whether to use gradient clipping.",
)
@hyper_opt(
    "--max-grad-norm",
    type=float,
    default=HyperparametersCfg.DEFAULT_MAX_GRAD_NORM,
    help="Max grad norm for gradient clipping.",
)
@hyper_opt(
    "--value-clipping/--no-value-clipping",
    "value_clipping_enabled",
    default=HyperparametersCfg.DEFAULT_VALUE_CLIPPING_ENABLED,
    help="Whether to use value clipping.",
)
@hyper_opt(
    "--clip-range-vf",
    type=float,
    default=HyperparametersCfg.DEFAULT_CLIP_RANGE_VF,
    help="Clip range for value function.",
)
@hyper_opt(
    "--vf-loss-coef",
    type=float,
    default=HyperparametersCfg.DEFAULT_VF_LOSS_COEF,
    help="How much value loss affects total loss (weighted).",
)
@hyper_opt(
    "--pi-loss-coef",
    type=float,
    default=HyperparametersCfg.DEFAULT_PI_LOSS_COEF,
    help="How much policy loss affects total loss (weighted).",
)
@hyper_opt(
    "--train-pi-iters",
    type=int,
    default=HyperparametersCfg.DEFAULT_TRAIN_PI_ITERS,
    help="Maximum number of gradient descent steps to take on policy loss per "
    + "epoch. (Early stopping may cause optimizer to take fewer than this.)",
)
@hyper_opt(
    "--train-v-iters",
    type=int,
    default=HyperparametersCfg.DEFAULT_TRAIN_V_ITERS,
    help="Number of gradient descent steps to take on value loss per epoch.",
)
@hyper_opt(
    "--vf-reg/--no-vf-reg",
    "vf_reg_enabled",
    default=HyperparametersCfg.DEFAULT_VF_REG_ENABLED,
    help="Whether to use value function regularization.",
)
@hyper_opt(
    "--vf-reg-l2",
    "vf_reg",
    type=float,
    default=HyperparametersCfg.DEFAULT_VF_REG,
    help="Value function l2 regularization (only for separate policy/value network).",
)
@hyper_opt(
    "--pi-lr",
    type=float,
    default=HyperparametersCfg.DEFAULT_PI_LR,
    help="Learning rate for policy optimizer.",
)
@hyper_opt(
    "--vf-lr",
    type=float,
    default=HyperparametersCfg.DEFAULT_VF_LR,
    help="Learning rate for value function optimizer.",
)
@hyper_opt(
    "--clip-vf-output/--no-clip-vf-output",
    default=HyperparametersCfg.DEFAULT_CLIP_VF_OUTPUT,
    help="Use value function clipping to [-1,1].",
)
@hyper_opt(
    "--bias-noops/--no-bias-noops",
    default=HyperparametersCfg.DEFAULT_BIAS_NOOPS,
    help="Add initial negative bias to noop spells, to make agent cast them "
    + "less frequently.",
)
# Architecture options
@arch_opt(
    "--rnn/--no-rnn",
    default=ArchitectureCfg.DEFAULT_RNN,
    help="Whether to use rnn policy.",
)
@arch_opt(
    "--unified-policy-value/--no-unified-policy-value",
    default=ArchitectureCfg.DEFAULT_UNIFIED_POLICY_VALUE,
    help="Unified network for policy and value.",
)
# Spells options
@spells_opt(
    "--spawn-noop/--no-spawn-noop",
    default=SpellsCfg.DEFAULT_SPAWN_NOOP,
    help="Whether to enable spell no-op.",
)
@spells_opt(
    "--spell-noop/--no-spell-noop",
    default=SpellsCfg.DEFAULT_SPELL_NOOP,
    help="Whether to enable spawn no-op.",
)
# Map options
@map_opt(
    "--bin-width",
    default=MapCfg.DEFAULT_BIN_WIDTH,
    help="Width of X bin - controls horizontal coordinate discretization.",
)
def train(**kwargs):
    """Initialize and start a new training experiment from scratch."""
    cfg = TrainingCfg()
    cfg.update({k: v for k, v in kwargs.items() if v is not None})
    run_experiment(cfg)


##########
# Resume #
##########


@cmd()
@arg(
    "exp_path", metavar="EXPERIMENT_PATH", type=click.Path(exists=True, file_okay=False)
)
def resume(exp_path):
    """Restore and continue an existing training experiment."""
    cfg = TrainingCfg.load(osp.join(exp_path, TRAIN_CFG_FILENAME))
    run_experiment(cfg, is_resume=True)


#########
# Serve #
#########


@cmd()
@arg(
    "model_path",
    metavar="SIMPLE_SAVE_PATH",
    type=click.Path(exists=True, file_okay=False),
)
@opt(
    "--port", "-p", type=int, default=5000, show_default=True, help="Serve at this port"
)
@opt(
    "--bind",
    "-b",
    type=str,
    default="localhost",
    show_default=True,
    help="Bind address",
)
def serve(model_path, port, bind):
    """
    Serve trained agent in inference mode.

    Provide a path to simple_save checkpoint directory.
    """
    serve_inference(model_path, bind_address=bind, port=port)


##########
# Render #
##########


@cmd()
@arg(
    "model_path",
    metavar="SIMPLE_SAVE_PATH",
    type=click.Path(exists=True, file_okay=False),
)
@opt("--seed", type=int, default=TrainingCfg.DEFAULT_SEED, help="Random seed value.")
@opt(
    "--server",
    type=HOST_PORT,
    default=("127.0.0.1:8081"),
    help="Address of simulation server.",
)
def render(model_path, server, seed):
    """
    Simulates and renders a battle in ncurses TUI.

    Runs inference with trained agent for left player.
    """
    run_render(model_path, server, seed)


############
# Simulate #
############


@cmd()
@arg(
    "model_path",
    metavar="SIMPLE_SAVE_PATH",
    type=click.Path(exists=True, file_okay=False),
)
@opt("--seed", type=int, default=TrainingCfg.DEFAULT_SEED, help="Random seed value.")
@opt(
    "-n", "--num-battles", type=int, default=100, help="Number of battles to simulate."
)
@opt(
    "-b",
    "--brain",
    type=click.Choice([b.name for b in Brain if b != Brain.UNDEFINED_DIFFICULTY]),
    default=Brain.UTILITY_9.name,
    help="Adversary brain to use for opponent.",
)
@opt(
    "--server",
    type=HOST_PORT,
    default=("127.0.0.1:8081"),
    help="Address of simulation server.",
)
def simulate(model_path, seed, num_battles, server, brain):
    """Run inference with trained agent in simulation mode.

    Simulates a number of battles and outputs win rate.
    """
    host, port = server.split(":")
    run_inference(model_path, host, port, num_battles, seed, Brain[str(brain)])
