import os
import sys
import json
import copy
import time
import datetime
import warnings
import signal
import cProfile

import hydra
import torch
import matplotlib.pyplot as plt
import numpy as np
import utils

from transformers import set_seed
from omegaconf import OmegaConf, DictConfig, listconfig


from plotter import Plotter
from optimizer import Optimizer
from current_functions import CurrentFunctions
from scorers import BasicScorer, MinMaxScorer, ComplexityScorer
from mloggers import ConsoleLogger, FileLogger, MultiLogger, LogLevel
from ed import ED
from sr import SR

from typing import Dict, Tuple, List, Any
from collections.abc import Callable


class Workspace(object):
    """
    Workspace class for running the symbolic regression experiment.
    """
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.allowed_exp_types = cfg.allowed_exp_types
        
        # Output setup
        self.root_dir = cfg.get("root", os.getcwd())
        self.output_dir = cfg.get("output_dir", "output")
        model_folder_name = cfg.model.name.strip()

        # Raise an exception if the experiment type is wrong. This is for organization purposes; experiment results are saved into folders
        # associated with the experiment types. 
        self.exp_type = cfg.initialization.exp_type
        if self.exp_type not in self.allowed_exp_types:
            raise Exception(f"Wrong experiment type {self.exp_type}!")
        
        if "/" in model_folder_name:
            model_folder_name = model_folder_name.split("/")[-1]
        experiment_folder_name = os.path.join(cfg.experiment.function.group, cfg.experiment.function.name) if hasattr(cfg.experiment.function, "group") else cfg.experiment.function.name
        self.output_path = os.path.join(self.root_dir, self.output_dir, experiment_folder_name, model_folder_name, self.exp_type, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/")
        while os.path.exists(self.output_path):
            self.output_path = os.path.join(self.root_dir, self.output_dir, experiment_folder_name, model_folder_name, self.exp_type, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + str(np.random.randint(0, 1000)) + "/")
        os.makedirs(self.output_path)
        os.makedirs(self.output_path + "posteriors/")

        # Logger setup
        cfg.logger.run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        loggers_list = cfg.logger.loggers
        log_level = LogLevel[cfg.logger.get("level", "INFO")]
        loggers = []
        for logger in loggers_list:
            if logger == "console":
                loggers.append(ConsoleLogger(default_priority=log_level))
            elif logger == "file":
                loggers.append(FileLogger(os.path.join(self.output_path, 'log.json'), default_priority=log_level))
            elif logger == "":
                pass
            else:
                print(f'[WARNING] Logger "{logger}" is not supported')
        self.logger = MultiLogger(loggers, default_priority=log_level)
        self.logger.info(f"Project root: {self.root_dir}.")
        self.logger.info(f"Logging to {self.output_path}.")
        job_id = utils.get_job_id()
        self.logger.info(f"Slurm job ID: {job_id}.") if job_id is not None else None

        # Redirect warnings to logger
        warnings.filterwarnings("default")
        warnings.showwarning = lambda *args, **kwargs: self.logger.warning(str(args[0]))
        
        # RNG setup
        if not hasattr(cfg, "seed") or cfg.seed is None or cfg.seed == -1:
            random_seed = int(time.time() * 1e6) % (2**32)
            if not self.cfg.random_seed_each_run:
                self.cfg.seed = random_seed
            self.logger.info(f"Seed not specified, using random seed: {random_seed}.")
        else:
            random_seed = self.cfg.seed
            self.logger.info(f"Using seed: {random_seed}.")

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed) if torch.cuda.is_available() else None
        set_seed(random_seed)

        if torch.cuda.is_available():
            torch.cuda.init()

        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)

        if cfg.get("use_bfloat16", False):
            self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.dtype = torch.float16

        self.logger.info(f"Using device: {self.device} with dtype: {self.dtype}.")
        if torch.cuda.is_available() and ('cuda' in cfg.device or 'auto' in cfg.device):
            self.logger.info(f"Device name: {torch.cuda.get_device_name()}.")
        
        self.cache_dir = self.cfg.model.get("cache_dir", os.environ.get("HF_HOME", None))
        if self.cache_dir == "":
            self.cache_dir = os.environ.get("HF_HOME", None)
        
        if self.cache_dir is not None:
            os.environ['HF_HOME'] = self.cache_dir 
            os.environ['TRANSFORMERS_CACHE'] = os.environ['HF_HOME']
        self.logger.info(f"Cache dir: {os.environ.get('HF_HOME', None)}.")
        
        # Experiment settings
        self.data_folder = cfg.experiment.function.train_points.get("data_folder", None)
        self.data_folder = os.path.join(self.root_dir, self.data_folder) if self.data_folder is not None else None

        self.train_domain_d = cfg.ED.train_domain_d
        self.min_train_points = cfg.ED.train_domain_d[0]
        self.max_train_points = cfg.ED.train_domain_d[1]
        self.num_init_pts_k = cfg.initialization.num_init_pts_k
        self.xs_noise_std = cfg.experiment.function.train_points.xs_noise_std
        self.ys_noise_std = cfg.experiment.function.train_points.ys_noise_std

        self.logger.info(f"Randomly sampled {self.num_init_pts_k} initial points.")

        self.test_domain_d = cfg.initialization.test_domain_d
        self.min_test_points = cfg.initialization.test_domain_d[0]
        self.max_test_points = cfg.initialization.test_domain_d[1]
        self.num_test_pts_m = cfg.initialization.num_test_pts_m

        factor = (self.max_test_points - self.min_test_points) / (self.max_train_points - self.min_train_points)
        self.num_extended_test_pts = int(self.num_test_pts_m * factor)
            
        self.logger.info(f"Randomly sampled {self.num_test_pts_m} train points on the dense grid.")
        self.logger.info(f"Randomly sampled {self.num_extended_test_pts} test points.")

        self.num_variables = cfg.experiment.function.num_variables
        self.num_digits = cfg.initialization.num_data_digits

        self.r2_tolerance = cfg.initialization.epsilon_r
        self.c_tolerance = cfg.initialization.epsilon_c
        self.exp_budget_n = cfg.ED.exp_budget_n
    
        self.checkpoints = cfg.checkpoints

        self.parameter_names = cfg.experiment.function.parameters
        self.true_function_name = cfg.experiment.function.test_function

        # Replace parameters in the true_function_name with their values from true_parameters
        if hasattr(cfg.experiment.function, "parameters") and hasattr(cfg.experiment.function, "true_parameter_values"):
            for param, value in zip(self.parameter_names, cfg.experiment.function.true_parameter_values.values()):
                self.true_function_name = self.true_function_name.replace(param, str(value))

        # Convert the true_function_name string into a callable function
        self.true_function = utils.string_to_function(self.true_function_name, self.num_variables)

        # Points setup
        if cfg.experiment.function.train_points.generate_points:
            add_extremes = cfg.experiment.function.train_points.add_extremes if hasattr(cfg.experiment.function.train_points, "add_extremes") else False
            random = cfg.experiment.function.train_points.random if hasattr(cfg.experiment.function.train_points, "random") else False
            self.train_points = self.generate_points(self.true_function, self.min_train_points, self.max_train_points, self.num_init_pts_k, xs_noise_std=self.xs_noise_std, ys_noise_std=self.ys_noise_std, random_points=random, add_extremes=add_extremes)
            self.train_points = utils.string_to_array(self.train_points)
            np.save(os.path.join(self.output_path, "train_points.npy"), self.train_points)

            self.R2_dense_train_points = self.generate_points(self.true_function, self.min_train_points, self.max_train_points, self.num_test_pts_m, xs_noise_std=self.xs_noise_std, ys_noise_std=self.ys_noise_std, random_points=False, add_extremes=add_extremes)
            self.R2_dense_train_points = utils.string_to_array(self.R2_dense_train_points)
            np.save(os.path.join(self.output_path, "R2_dense_train_points.npy"), self.R2_dense_train_points)
            
            self.test_points = self.generate_points(self.true_function, self.min_test_points, self.max_test_points, self.num_extended_test_pts,random_points=False)
            self.test_points = utils.string_to_array(self.test_points)
            np.save(os.path.join(self.output_path, "test_points.npy"), self.test_points)

        else:
            assert self.data_folder is not None, "No data folder specified."
            assert os.path.exists(self.data_folder), f"Data folder {self.data_folder} does not exist."
            assert os.path.exists(os.path.join(self.data_folder, 'train_points.npy')), f"Train points file {os.path.join(self.data_folder, 'train_points.npy')} does not exist."
            assert os.path.exists(os.path.join(self.data_folder, 'test_points.npy')), f"Test points file {os.path.join(self.data_folder, 'test_points.npy')} does not exist."
            
            train_points_file = os.path.join(self.data_folder, 'train_points.npy')
            self.train_points = utils.load_points(train_points_file)
            self.logger.info(f"Loaded train points from {train_points_file}.")

            R2_dense_train_points_file = os.path.join(self.data_folder, 'R2_dense_train_points.npy')
            self.R2_dense_train_points = utils.load_points(R2_dense_train_points_file)
            self.logger.info(f"Loaded R2 dense train points from {R2_dense_train_points_file}.")
            
            test_points_file = os.path.join(self.data_folder, 'test_points.npy')
            self.test_points = utils.load_points(test_points_file)
            self.logger.info(f"Loaded test points from {test_points_file}.")
            
        self.num_train_points = len(self.train_points)
        self.num_test_points = len(self.test_points)

        self.logger.info(f"Train points: {utils.array_to_string(self.train_points, self.num_digits)}.")
        if self.num_test_points > 100:
            self.logger.info(f"Not logging test points as there are more than 100 ({self.num_test_points}).")
        else:
            self.logger.info(f"Test points: {utils.array_to_string(self.test_points, self.num_digits)}.")

        # Optimizer settings
        self.optimizer = Optimizer(cfg, self.train_points, self.logger)

        # Plotter setup
        self.save_frames = cfg.plotter.save_frames if hasattr(cfg.plotter, "save_frames") else False
        if self.save_frames:
            os.makedirs(self.output_path + "frames/")
        
        self.save_video = cfg.plotter.save_video if hasattr(cfg.plotter, "save_video") else True
        self.save_video = False if self.num_variables > 2 else self.save_video
     
        self.plotter = Plotter(cfg, self.train_points, self.test_points, self.output_path)
        self.plotter.plot_points(save_fig=True, plot_test=False)

        # Prompts
        self.prompts_path = os.path.join(self.root_dir, cfg.prompts_path)
        self.num_to_sample_b = cfg.SR.num_to_sample_b
        self.num_best_funcs_c = cfg.SR.num_best_funcs_c

        # Base SR prompt
        with open(os.path.join(self.prompts_path, cfg.prompt_folder, cfg.sr_prompt_name), "r") as f:
            self.base_sr_prompt = f.read()
            self.base_sr_prompt = self.base_sr_prompt.format(points="{points}", domain=self.train_domain_d, num_best_funcs_c=self.num_best_funcs_c, 
                                        functions="{functions}", num_to_sample_b=self.num_to_sample_b, num_variables=self.num_variables, 
                                        variables_list=[f"x{i+1}" for i in range(self.num_variables)])

        # Base ED prompt
        with open(os.path.join(self.prompts_path, cfg.prompt_folder, cfg.ed_prompt_name), "r") as f:
            self.base_ed_prompt = f.read()
            self.base_ed_prompt = self.base_ed_prompt.format(budget_remaining="{budget_remaining}", num_init_pts_k=self.num_init_pts_k, 
                                        prior_data=utils.array_to_string(self.train_points, self.num_digits), llm_sampled_data="{llm_sampled_data}", 
                                        domain=self.train_domain_d)
        
        # Initial ED prompt
        with open(os.path.join(self.prompts_path, cfg.prompt_folder, cfg.ed_initial_prompt_name), "r") as f:
            self.initial_ed_prompt = f.read()
            self.initial_ed_prompt = self.initial_ed_prompt.format(budget_remaining=self.exp_budget_n, num_init_pts_k=self.num_init_pts_k, 
                                        prior_data=utils.array_to_string(self.train_points, self.num_digits), domain=self.train_domain_d)
        
        # Model settings
        self.model_name = cfg.model.name
        self.model = None
        self.input_img = None
    
        self.logger.info(f"Base SR Prompt: {self.base_sr_prompt} with input image {self.input_img}.")
        self.logger.info(f"Base ED Prompt: {self.base_ed_prompt} with input image {self.input_img}.")
        self.logger.info(f"Initial ED Prompt: {self.initial_ed_prompt} with input image {self.input_img}.")

        self.tokenizer_pad = cfg.model.tokenizer_pad
        self.tokenizer_padding_side = cfg.model.tokenizer_padding_side

        self.max_new_tokens = cfg.model.max_new_tokens
        self.top_p = cfg.model.top_p
        self.top_k = cfg.model.top_k
        self.num_beams = cfg.model.num_beams

        self.temperature = cfg.model.temperature
        if cfg.model.temperature_schedule:
            self.temperature_scheduler = torch.optim.lr_scheduler.ExponentialLR(torch.optim.Adam([torch.tensor(self.temperature)], lr=1), gamma=cfg.model.temperature_schedule_gamma)
        else:
            self.temperature_scheduler = None
            
        model_args = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "num_beams": self.num_beams,
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": 0,
            "tokenizer_pad": self.tokenizer_pad,
            "tokenizer_padding_side": self.tokenizer_padding_side,
            "seed": random_seed,
            "api_key_path": os.path.join(self.root_dir, cfg.model.api_key_path) if hasattr(cfg.model, "api_key_path") else None,
            "organization_id_path": os.path.join(self.root_dir, cfg.model.organization_id_path) if hasattr(cfg.model, "organization_id_path") else None,
        }

        if torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0):
            model_args['attn_implementation'] = 'flash_attention_2'
            model_args['use_flash_attn'] = True
            self.logger.info("Using Flash Attention 2")
        
        self.model = utils.load_model(self.model_name, self.device, self.dtype, self.cache_dir, model_args)
        self.logger.info("Model loaded - {model_name}.".format(model_name=self.model_name))

        # Scorer settings
        if "basic" in cfg.experiment.scorer.name.lower():
            self.scorer = BasicScorer(self.train_points, rounding=cfg.experiment.scorer.rounding, scientific=cfg.experiment.scorer.scientific)
            self.test_scorer = BasicScorer(self.test_points, rounding=cfg.experiment.scorer.rounding, scientific=cfg.experiment.scorer.scientific)
        elif "minmax" in cfg.experiment.scorer.name.lower():
            min_score = cfg.experiment.scorer.min_score
            max_score = cfg.experiment.scorer.max_score
            self.scorer = MinMaxScorer(self.train_points, min_score=min_score, max_score=max_score, rounding=cfg.experiment.scorer.rounding, scientific=cfg.experiment.scorer.scientific)
            self.test_scorer = MinMaxScorer(self.test_points, min_score=min_score, max_score=max_score, rounding=cfg.experiment.scorer.rounding, scientific=cfg.experiment.scorer.scientific)
        elif "complexity" in cfg.experiment.scorer.name.lower():
            self.logger.info(f"Complexity scorer with lambda {cfg['experiment']['scorer']['lambda']} and max nodes {cfg.experiment.scorer.max_nodes}.")
            alternative = False
            if hasattr(cfg.experiment.scorer, "alternative") and cfg.experiment.scorer.alternative:
                alternative = True
                self.logger.info("Using alternative complexity scorer.")
            self.scorer = ComplexityScorer(self.train_points, rounding=cfg.experiment.scorer.rounding, scientific=cfg.experiment.scorer.scientific, lam=cfg['experiment']['scorer']['lambda'], max_nodes=cfg.experiment.scorer.max_nodes, alternative=alternative)
            self.dense_train_scorer = ComplexityScorer(self.R2_dense_train_points, rounding=cfg.experiment.scorer.rounding, scientific=cfg.experiment.scorer.scientific, lam=cfg['experiment']['scorer']['lambda'], max_nodes=cfg.experiment.scorer.max_nodes, alternative=alternative)
            self.test_scorer = ComplexityScorer(self.test_points, rounding=cfg.experiment.scorer.rounding, scientific=cfg.experiment.scorer.scientific, lam=cfg['experiment']['scorer']['lambda'], max_nodes=cfg.experiment.scorer.max_nodes, alternative=alternative)
        else:
            self.logger.error(f"Scorer {cfg.experiment.scorer.name} not supported.")
            exit(1)

        # Seed functions
        self.seed_functions = {}
        min_seed_functions = max(5, self.num_to_sample_b) # If the prompt size is small (e.g. 1) we still want to generate a few seed functions to avoid getting stuck
        gen_time = 0
        if cfg.experiment.generate_seed_functions:
            self.seed_functions, gen_time = self.generate_seed_functions()
            if (len(self.seed_functions) < min_seed_functions):
                self.logger.warn(f"Could not generate {min_seed_functions} seed functions. Generated {len(self.seed_functions)} seed functions.")
        else:
            self.seed_functions = {name: utils.string_to_function(name, self.num_variables) for name in cfg.experiment.seed_functions.functions}
            self.logger.info(f"Loaded seed functions: {self.seed_functions}.")
        self.current_functions = CurrentFunctions(self.seed_functions, self.scorer, self.optimizer, self.num_best_funcs_c, self.logger, self.num_variables)
        self.logger.info(f"Current functions: {self.current_functions.functions}.")
        self.logger.info(f"Current scores: {self.current_functions.scores}.")

        if len(self.current_functions.functions) < self.num_to_sample_b:
            self.logger.warning(f"Could not generate {self.num_to_sample_b} seed functions. Generated {len(self.current_functions.functions)} seed functions.")
            if len(self.current_functions.functions) == 0:
                self.logger.error("No seed functions generated. Raising exception.")
                raise Exception("No seed functions generated.")
        else:
            self.logger.info(f"Succesfully generated {len(self.current_functions.functions)} seed functions in {gen_time} seconds.")

        # Results json
        self.results = {
            "experiment_name": self.cfg.experiment.function.name,
            "seed": random_seed,
            "train_points": utils.array_to_string(self.train_points, self.num_digits),
            "R2_dense_train_points": utils.array_to_string(self.R2_dense_train_points, self.num_digits),
            "test_points": utils.array_to_string(self.test_points, self.num_digits),
            "best_expr": "",
            "best_function": "",
            "scores": [],
            "R2_trains": [],
            "R2_tests": [],
            "R2_alls": [],
            "best_scores": [],
            "best_scores_normalized": [],
            "iterations": 0,
            "tries_per_iteration": [],
            "generations_per_iteration": [],
            "num_unique": len(self.current_functions.functions),
            "best_found_at": 0,
            "sympy_equivalent": False,
            "temperatures": [],
            "times": {
                "iteration": [],
                "seed_function_generation": gen_time,
                "generation_per_iteration": [],
                "optimization_per_iteration": [],
            }
        }
        if "test_function" in self.cfg.experiment.function:
            self.results["test_function"] = self.cfg.experiment.function.test_function

        # Save config
        with open(self.output_path + "config.yaml", "w") as f:
            OmegaConf.save(self.cfg, f)

        # Generate the ED and SR phases
        self.ED = ED(self.cfg, self.logger, self.base_ed_prompt, self.initial_ed_prompt, self.model, utils.array_to_string(self.train_points, self.num_digits))
        self.SR = SR(self.cfg, self.logger, self.current_functions, self.base_sr_prompt, self.temperature_scheduler,
                     self.results, self.model, self.optimizer, self.scorer, self.test_points, self.plotter, self.true_function,
                     self.output_path)

    def generate_points(self, function: Callable, min_points: float, max_points: float, num: int, xs_noise_std: float = 0, ys_noise_std: float = 0, 
                        random_points: bool = False, add_extremes: bool = True) -> str:
        """ 
        Generates points from a given function, with optional noise.

        Parameters
        ----------
        function -> the function to generate points from.
        min_points -> the minimum value of the points to generate.
        max_points -> the maximum value of the points to generate.
        num -> the number of points to generate.
        xs_noise_std -> the standard deviation of the noise to add to the xs.
        ys_noise_std -> the standard deviation of the noise to add to the ys.
        random_points -> whether to generate random points instead of a grid/meshgrid.
        add_extremes -> whether to add points at the extreme values of the interval manually to ensure they are included.

        Returns
        -------
        points -> the points as a string.
        """
        min_value = copy.deepcopy(min_points)
        max_value = copy.deepcopy(max_points)
        if type(min_points) != list and type(min_points) != listconfig.ListConfig:
            min_points = [min_points] * self.num_variables
        if type(max_points) != list and type(max_points) != listconfig.ListConfig:
            max_points = [max_points] * self.num_variables
        min_points = np.array(min_points, dtype=np.float32)
        max_points = np.array(max_points, dtype=np.float32)
        
        points_per_dim = int(np.floor(num**(1/self.num_variables)))
        self.logger.info(f"Generating {points_per_dim} points per dimension for a total of {points_per_dim**self.num_variables} points.")
        
        if random_points:
            # Add points at the extreme values of the interval manually to ensure they are included
            # This depends on the number of dimensions
            # For example, in 1D if the interval is [0, 1] we need to add points at 0 and 1
            # In 2D, if the interval is [(0, 0), (1, 1)] we need to add points at (0, 0), (0, 1), (1, 0), (1, 1)
            if add_extremes:
                variable_ranges = np.array([[min_points[i], max_points[i]] for i in range(self.num_variables)])
                extreme_points = np.array(np.meshgrid(*variable_ranges)).T.reshape(-1, self.num_variables)
                self.logger.info(f"Adding {len(extreme_points)} extreme points ({extreme_points}).")
            
            # Reshape min and max points to match the random shape. Currently min and max are of shape (num_variables,), so we need to add n dimensions of size points_per_dim by copying the min and max values
            # For example, if min is [0, 1] and num_variables is 2 and points_per_dim is 3, we need to reshape min to an array of shape (2, 3, 3) with all values being 0 and 1 across the last dimension
            random_shape = tuple([self.num_variables, *([points_per_dim] * self.num_variables)])
            min_points = np.expand_dims(min_points, axis=tuple(range(1, self.num_variables + 1)))
            max_points = np.expand_dims(max_points, axis=tuple(range(1, self.num_variables + 1)))
            max_points += 1e-10 # Add small eps to max_points as the rightmost value is not included in np.random.uniform
            for i in range(1, self.num_variables+1):
                min_points = np.repeat(min_points, points_per_dim, axis=i)
                max_points = np.repeat(max_points, points_per_dim, axis=i)
            Xs = np.random.uniform(min_points, max_points, random_shape)
            
        else:
            Xs = np.meshgrid(*[np.linspace(min_points[i], max_points[i], points_per_dim) for i in range(self.num_variables)])
        Xs = np.array(Xs)
        if xs_noise_std:
            Xs += np.random.normal(0, xs_noise_std, Xs.shape)
        pts = np.array(list(zip(*[x.flat for x in Xs])))

        ys = utils.eval_function(function, pts, self.num_variables).T
            
        if ys_noise_std:
            ys += np.random.normal(0, ys_noise_std, ys.shape)
        
        if random_points and add_extremes:
            pts = np.concatenate((extreme_points, pts))
            extreme_ys = utils.eval_function(function, extreme_points, self.num_variables).T
            ys = np.concatenate((extreme_ys, ys))
        
        points = np.concatenate((pts, ys.reshape(-1, 1)), axis=1)
        if add_extremes and len(points) > num:
            # Remove random points to account for the extra extremes
            # The points are sampled randomly so removing from the end is the same as removing random indices
            self.logger.info(f"Removing {len(points)-num} randomly generated points: {points[num:]}")
            points = points[:num]
        while any(np.isinf(points[:, -1])):
            # Remove points where the function is infinite
            inf_indices = np.where(np.isinf(points))
            self.logger.info(f"Removing {len(inf_indices)} points where the function is infinite.")
            points = np.delete(points, inf_indices[0], axis=0)
            
            if len(points) < num and random_points:
                # Generate new points to replace the infinite ones
                self.logger.info(f"Recursively generating {num-len(points)} new points.")
                new_points = self.generate_points(function, min_value, max_value, num-len(points), xs_noise_std, ys_noise_std, random_points, add_extremes=False)
                points = np.concatenate((points, new_points))

        return utils.array_to_string(points, self.num_digits)
    
    def generate_seed_functions(self) -> Tuple[Dict[str, Any], float]:
        """
        Generates initial seed functions for the experiment.

        Parameters
        ----------

        Returns
        -------
        seed_functions -> the generated seed functions.
        gen_time -> the time it took to generate the seed functions.
        """
        generation_tokens = self.cfg.experiment.seed_functions.generation_tokens if hasattr(self.cfg.experiment.seed_functions, "generation_tokens") else 512
        max_tries = self.cfg.experiment.seed_functions.max_tries if hasattr(self.cfg.experiment.seed_functions, "max_tries") else 10
        seed_functions = {}
        
        seed_prompt = self.cfg.get("model").get("seed_function_prompt", None)
        assert seed_prompt is not None, "Seed function prompt not specified."
        seed_prompt = os.path.join(self.prompts_path, seed_prompt)
        
        with open(seed_prompt, "r") as f:
            prompt = f.read()
        img_path = os.path.join(self.output_path, "points.png") if self.input_img else None

        prompt = prompt.format(points=utils.array_to_string(self.train_points, self.num_digits), num_variables=self.num_variables, 
                               variables_list=[f"x{i+1}" for i in range(self.num_variables)])
        self.logger.info("Prompt for seed functions generation:")
        self.logger.info(prompt)
        
        start_time = time.perf_counter()
        with torch.inference_mode():
            for i in range(max_tries):
                # Generate seed functions using the model
                self.logger.info(f"Attempt {i+1} of {max_tries} to generate seed functions.")
                seeds = self.model.generate(prompt, return_prompt=False, image_files=img_path, temperature=self.temperature, max_new_tokens=generation_tokens)
                self.logger.info("Model output for seed functions: " + seeds)

                # Parse model output
                for seed in seeds.split("\n"):
                    if "x" not in seed:
                        self.logger.info(f"Skipping line {seed} as it does not contain 'x' and is likely not a function.")  
                        continue
                    if "Error" in seed:
                        self.logger.info(f"Skipping line {seed} as it contains 'Error'.")
                        continue
                    seed = utils.clean_function(seed)
                    self.logger.info(f"Seed function: {seed}.")
                    if seed == "":
                        continue
                    try:
                        valid, reason = utils.is_valid_function(seed, None, self.num_variables)
                        self.logger.info(f"Function {seed}. Valid: {valid}. Reason: {reason}.")
                        if valid:
                            function = utils.string_to_function(seed, self.num_variables)
                            seed_functions[seed] = function
                    except Exception as e:
                        self.logger.warning(f"Could not parse line {seed}.")
                        self.logger.warning(str(e))
                        pass
                # Here we continue even if we already have enough seed functions, as we might not have enough valid seed functions after optimization
                # Perhaps a better approach should be optimizing here directly and exiting if we have enough valid seed functions
        end_time = time.perf_counter()

        self.logger.info(f"Generated seed functions: {seed_functions}.")
        return seed_functions, end_time - start_time
    
    def perform_checkpoint(self, iteration: int, main_timer_start: float) -> None:
        """
        Performs checkpoint operations.
        """
        self.logger.info(f"Checkpoint {iteration}. Saving results.")
        results_checkpoint = copy.deepcopy(self.results)
        checkpoint_timer_end = time.perf_counter()
        results_checkpoint["times"]["total"] = checkpoint_timer_end - main_timer_start
        
        best_expr = self.current_functions.get_best_function(return_coeff=True)
        best_function = self.current_functions.get_best_function(return_coeff=False)
        test_score = self.test_scorer.score(best_function)
        results_checkpoint["test_score"] = test_score
        results_checkpoint["best_function"] = str(best_function)
        results_checkpoint["best_expr"] = str(best_expr)
        r2_train, r2_test, r2_all = self.SR.get_R2_scores(best_function, self.all_train_points, self.test_points)
        results_checkpoint["r2_train"] = r2_train
        results_checkpoint["r2_test"] = r2_test
        results_checkpoint["r2_all"] = r2_all
        results_checkpoint["final_complexity"] = utils.count_nodes(best_function)
        with open(self.output_path + f"results_checkpoint_{iteration}.json", "w") as f:
            json.dump(results_checkpoint, f)
        self.logger.info(f"Checkpoint {iteration} saved.")


    def run(self) -> int:
        """
        Runs the main experiment, iterating and generating new functions until the tolerance is reached.

        Returns 0 if the experiment fails, 1 otherwise.
        """
        main_timer_start = time.perf_counter()
        if self.save_video:
            frames = []

        # Check if one of the generated seed functions is already below the cost tolerance
        # Here, "score" is actually the cost value 
        best_expr = self.current_functions.get_best_function(return_coeff=True)
        best_function = self.current_functions.get_best_function(return_coeff=False)
        score = self.current_functions.scores[best_expr]

        if score <= self.c_tolerance:
            self.all_train_points = utils.array_to_string(self.train_points, self.num_digits)

            r2_train, r2_test, r2_all = self.SR.get_R2_scores(best_function, self.train_points, self.test_points)
            r2_dense_train, _, _, = self.SR.get_R2_scores(best_function, self.R2_dense_train_points, self.test_points)

            self.logger.info(f"The seed function {best_expr} is already below the C tolerance {self.c_tolerance}.")
            self.logger.info(f"Best function: {best_function}. R2 (train): {r2_train}.")

            self.results["scores"].append(score) if score != np.inf else self.results["scores"].append("inf")
            self.results["best_scores"].append(self.current_functions.scores[best_expr])
            self.results["best_scores_normalized"].append(self.current_functions.norm_scores[best_expr])
            self.results["best_found_at"] = 0
            self.results["temperatures"].append(self.temperature)
        
            if self.save_video:
                frame, ax = self.plotter.record_frame(best_function, best_function, r2_test, self.true_function, -1, plot_true=True)
                if self.save_frames:
                    frame.savefig(self.output_path + "frames/" + f"{0}.png")
                frames.append(frame)
        else:
            # Start the main loop
            budget_remaining = self.exp_budget_n
            llm_sampled_data = ""
            llm_sampled_data_with_IG = ""
            og_train_points = utils.array_to_string(self.train_points, self.num_digits)

            while budget_remaining > 0:
                iteration = self.exp_budget_n - budget_remaining

                # perform experimental design
                proposed_design = self.ED.propose_design(llm_sampled_data_with_IG, llm_sampled_data, budget_remaining, iteration)
                
                # sample data accordingly and update train points and round to self.num_digits digits
                # new_point is an np array of a float, so we first access it and then round
                new_point_y = utils.eval_function(self.true_function, np.array([proposed_design]), self.num_variables)
                new_point_y = round(new_point_y.item(), self.num_digits)

                if iteration == 0:
                    llm_sampled_data = str((proposed_design, new_point_y))
                else:
                    llm_sampled_data = llm_sampled_data + ", " + str((proposed_design, new_point_y))
                self.all_train_points = og_train_points + ", " + llm_sampled_data

                # Update the training points to include the newly sampled points for everything they are used in
                self.optimizer.points = utils.string_to_array(self.all_train_points)
                self.scorer.points = utils.string_to_array(self.all_train_points)
                self.plotter.train_points = utils.string_to_array(self.all_train_points)

                # Don't forget to recompute scores with the newly sampled data point included before prompting the mode
                self.current_functions.scores, self.current_functions.norm_scores = self.scorer.score_current_functions(self.current_functions.functions)

                # Perform symbolic regression, optimization, compute cost and update results
                best_expr, best_func, best_score = self.SR.propose_exp_and_get_score(self.all_train_points, iteration, frames)

                # Update priors
                observed_data = utils.string_to_array(self.all_train_points)
                max_information_gain = 0
                best_posterior = None

                for exp, func in self.current_functions.functions.items():
                    # Replace all the coefficients in the expression with the optimized parameters, 
                    # except those in self.parameter_names
                    optimized_param_dict = func[1]
                    optimized_param_dict = { param: val for param, val in optimized_param_dict.items() if param not in self.parameter_names }
                    
                    replaced_exp = utils.replace_coefficients(str(exp), optimized_param_dict)

                    try:
                        likelihood = self.ED.compute_likelihood(observed_data, replaced_exp)
                        posterior, information_gain = self.ED.update_posterior(likelihood)

                        if information_gain > max_information_gain:
                            max_information_gain = information_gain
                            best_posterior = posterior
                    except:
                        continue

                # Update the prior and the prior entropy for the next iteration. If posterior is None, either all the functions resulted in 0 likelihoods or there was negative information gain. Then, keep the old prior as is and and the information gain as 0.
                if best_posterior is not None:
                    self.ED.prior = best_posterior  
                    self.ED.prior_entropy = self.ED.prior_entropy - max_information_gain

                new_data_with_IG = str((proposed_design, new_point_y, max_information_gain))    
                if iteration == 0:
                    llm_sampled_data_with_IG = new_data_with_IG
                else:
                    llm_sampled_data_with_IG += ", " + new_data_with_IG
                    
                budget_remaining -= 1

                # Save the posterior distribution plots
                fig, ax = self.ED.plot_posteriors(iteration)
                fig.savefig(self.output_path + f"posteriors/iteration_{iteration + 1}.png")

                # Check if the true function has been found
                if self.true_function is not None:
                    if utils.func_equals(best_func, self.true_function, self.num_variables):
                        self.logger.info(f"Function is equivalent to the true function.")
                        self.results["equivalent"] = True
                        break
                
                # Check if the tolerance is reached
                if best_score <= self.c_tolerance:
                    self.logger.info(f"Found a function with cost (train) = {score} below the tolerance {self.c_tolerance}.")
                    break
                
                if iteration in self.checkpoints:
                    self.perform_checkpoint(iteration, main_timer_start)

        # Save final results into a json
        main_timer_end = time.perf_counter()
        best_expr = self.current_functions.get_best_function(return_coeff=True)
        best_function = self.current_functions.get_best_function(return_coeff=False)
        best_popt = self.current_functions.optimized_params[best_expr]
        test_score = self.test_scorer.score(best_function)
        dense_training_score = self.dense_train_scorer.score(best_function)

        # Save final figures and animations
        if hasattr(self, "test_function_name") and self.true_function_name is not None:
            self.logger.info(f"True function: {self.true_function_name}")

        fig, ax = self.plotter.plot_results(best_function, self.true_function)
        fig.savefig(self.output_path + "final.png")

        if self.save_video and len(frames) > 0:
            self.plotter.record_video(frames)

        self.logger.info(f"Test score: {test_score}.")
        self.logger.info(f"Dense training score: {dense_training_score}.")
        self.logger.info(f"Best function: {best_function}. Score: {self.current_functions.scores[best_expr]} ({self.current_functions.norm_scores[best_expr]}).")
        
        self.results["best_function"] = str(best_function)
        self.results["best_expr"] = str(best_expr)
        self.results["best_popt"] = str(best_popt)
        self.results["test_score"] = test_score
        self.results["dense training score"] = dense_training_score
        self.results["times"]["total"] = main_timer_end - main_timer_start + self.results["times"]["seed_function_generation"]
        self.results["times"]["avg_generation"] = np.mean(self.results["times"]["generation_per_iteration"]) if len(self.results["times"]["generation_per_iteration"]) > 0 else 0
        self.results["times"]["avg_optimization"] = np.mean(self.results["times"]["optimization_per_iteration"]) if len(self.results["times"]["optimization_per_iteration"]) > 0 else 0


        r2_train, r2_test, r2_all = self.SR.get_R2_scores(best_function, utils.string_to_array(self.all_train_points), self.test_points)
        r2_dense_train, _, _, = self.SR.get_R2_scores(best_function, self.R2_dense_train_points, self.test_points)
        self.results["r2_train"] = r2_train
        self.results["r2_dense_train"] = r2_dense_train
        self.results["r2_test"] = r2_test
        self.results["r2_all"] = r2_all
        self.logger.info(f"R2 train: {np.round(r2_train, 6)}. R2 test: {np.round(r2_test, 6)}. R2 all: {np.round(r2_all, 6)}.")
        
        final_complexity = utils.count_nodes(best_function)
        self.results["final_complexity"] = final_complexity
        self.logger.info(f"Number of nodes in final expression tree: {final_complexity}.")

        r2_train_sucess = bool(r2_train >= self.r2_tolerance)
        r2_dense_train_sucess = bool(r2_dense_train >= self.r2_tolerance)
        r2_test_sucess = bool(r2_test >= self.r2_tolerance)
        r2_all_sucess = bool(r2_all >= self.r2_tolerance)

        self.results["r2_train_sucess"] = r2_train_sucess
        self.results["r2_dense_)train_sucess"] = r2_dense_train_sucess
        self.results["r2_test_sucess"] = r2_test_sucess
        self.results["r2_all_sucess"] = r2_all_sucess

        with open(self.output_path + "results.json", "w") as f:
            json.dump(self.results, f)

        return r2_dense_train_sucess
        
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    success = 0
    num_exps = cfg.initialization.num_exps_l
    num_finished = 0
    for _ in range(num_exps):
        try: 
            print(f"Running experiment {_ + 1} of {num_exps}...")
            workspace = Workspace(cfg)
            success += workspace.run()
            num_finished += 1
        except Exception as e:
            print(f"Experiment {_ + 1} failed. {e}")
            continue

    print(f"Success ratio: {success}/{num_exps}")
    print(f"Number of experiments that completed without errors: {num_finished}/{num_exps}")
    
def dump_profile():
    profiler.disable()
    job_id = utils.get_job_id()
    print(f"Dumping profile to {os.path.join(os.getcwd(), 'profiles', 'profile')}_{job_id if job_id is not None else 'local'}")
    if not os.path.exists("./profiles"):
            os.makedirs("./profiles")
    profiler.dump_stats(f"./profiles/profile_{job_id if job_id is not None else 'local'}")

def signal_handler(sig, frame):
    # Ignore warnings, as otherwise we break the logger
    warnings.filterwarnings("ignore")
    dump_profile()
    print(f"Detecting signal {sig}. Dumping profile to {os.path.join(os.getcwd(), 'profiles', 'profile')}_{job_id if job_id is not None else 'local'}")
    sys.stdout.flush()
    if sig == signal.SIGTERM or sig == signal.SIGINT:
        sys.exit(1)

if __name__ == "__main__":
    # Run full profiler if env variable PROFILE is set
    do_profile = os.environ.get("PROFILE", False)
    print("Initializing profiler.")
    print("Profile will only be created if the code fails or is terminated.") if not do_profile else print("Profile will be created.")
    job_id = utils.get_job_id()

    # Set termination signal handlers to dump profile when terminated by SLURM
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    #signal.signal(signal.SIGCONT, signal_handler)
    
    # Setup profiler
    global profiler
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        main()
    except Exception as e:
        # Catch exceptions and dump profile
        print("Caught exception in main.")
        print(e)
        dump_profile()
        sys.exit(2)

    if do_profile:
        dump_profile()