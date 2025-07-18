import logging
import os
import warnings
from typing import List, Sequence
import numpy as np
import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
import yaml 

def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration
    Modifies DictConfig in place.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger()

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionaly saves:
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    hparams['run_id'] = trainer.logger.experiment[0].id
    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()

def summarize(eval_dict, prefix): # removes list entries from dictionary for faster logging
    # for set in list(eval_dict) : 
    eval_dict_new = {}
    for key in list(eval_dict) :
        if type(eval_dict[key]) is not list :
            eval_dict_new[prefix + '/' + key] = eval_dict[key]
    return eval_dict_new

def get_yaml(path): # read yaml 
    with open(path, "r") as stream:
        try:
            file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return file

def get_checkpoint(cfg, path): 
    checkpoint_path = path

    checkpoint_to_load = cfg.get("checkpoint",'last') # default to last.ckpt 
    all_checkpoints = os.listdir(checkpoint_path + '/checkpoints')
    hparams = get_yaml(path+'/csv//hparams.yaml')
    wandbID = hparams['run_id']
    checkpoints = {}
    for fold in range(cfg.get('num_folds',1)):
        checkpoints[f'fold-{fold+1}'] = [] # dict to store the checkpoints with their path for different folds

    if checkpoint_path.endswith('.ckpt'):
        for fold in checkpoints:
            checkpoints[fold] = checkpoint_path
        return wandbID, checkpoints

    if checkpoint_to_load == 'last':
        matching_checkpoints = [c for c in all_checkpoints if "last" in c]
        matching_checkpoints.sort(key = lambda x: x.split('fold-')[1][0:1])
        for fold, cp_name in enumerate(matching_checkpoints):
            checkpoints[f'fold-{fold+1}'] = checkpoint_path + '/checkpoints/' + cp_name
    elif 'best' in checkpoint_to_load :
        matching_checkpoints = [c for c in all_checkpoints if "last" not in c]
        matching_checkpoints.sort(key = lambda x: x.split('loss-')[1][0:4]) # sort by loss value -> increasing
        for fold in checkpoints:
            for cp in matching_checkpoints:
                if fold in cp:
                    checkpoints[fold].append(checkpoint_path + '/checkpoints/' + cp)
            if not 'best_k' in checkpoint_to_load: # best_k loads the k best checkpoints 
                checkpoints[fold] = checkpoints[fold][0] # get only the best (first) checkpoint of that fold
    return wandbID, checkpoints


def calc_interres(dims,fac,num_pooling,k,p,s):
    dims = [int(x/fac) for x in dims]
    if len(dims)==2:
        w,h = dims 
        d = None
    else:
        w,h,d = dims
    for i in range(num_pooling):
        w = int((w-k+2*p)/s +1)
        h = int((h-k+2*p)/s +1)
        if d is not None: 
            d = int((d-k+2*p)/s +1)
    return [w,h] if d is None else [w,h,d] 