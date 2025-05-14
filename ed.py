import sys
import numpy as np
import matplotlib.pyplot as plt
import utils

from scipy.stats import norm
from scipy.special import entr
from typing import Tuple, Callable 
from omegaconf import DictConfig
from mloggers import MultiLogger
from typing import Any, List


class ED(object):
    """
    Experimental design class for performing the ED phase. 
    """
    def __init__(self, cfg: DictConfig, logger: MultiLogger, ed_prompt: str, initial_ed_prompt: str, model: Any, train_pts: str) -> None:
        """
        Initializes the ED phase.

        Parameters
        ----------
        cfg : DictConfig -> The configuration file.
        logger : MultiLogger -> The logger to log to.
        """
        self.cfg = cfg
        self.logger = logger

        self.og_train_points = train_pts
        self.base_prompt = ed_prompt
        self.initial_prompt = initial_ed_prompt
        self.model = model

        self.train_domain_d = cfg.ED.train_domain_d
        self.temperature = cfg.model.temperature
        self.ed_retries = cfg.ED.retries
        self.num_digits = cfg.initialization.num_data_digits

        self.parameter_ranges = cfg.experiment.function.parameter_ranges
        self.true_a = cfg.experiment.function.true_parameter_values["a"]
        self.true_b = cfg.experiment.function.true_parameter_values["b"]
        self.noise_std = cfg.experiment.function.noise_std

        # Initialize uniform priors over a and b
        self.prior = np.ones((len(self.parameter_ranges['a']), len(self.parameter_ranges['b'])))
        self.prior /= self.prior.sum()  # Normalize to ensure it's a valid probability distribution
        self.prior_entropy = np.sum(entr(self.prior))

        # Make a copy of the uniform prior in case likelihood computation raises ValueError and we 
        # need to reset the prior to the original uniform prior
        self.uniform_prior = np.ones((len(self.parameter_ranges['a']), len(self.parameter_ranges['b'])))
        self.uniform_prior /= self.uniform_prior.sum()
        self.uniform_prior_entropy = np.sum(entr(self.uniform_prior))

    def compute_likelihood(self, observed_data: np.ndarray, exp: str) -> np.ndarray:
        """
        Computes the Gaussian likelihood of the observed data given a symbolic expression.

        Parameters
        ----------
        observed_data : np.ndarray -> The observed (x, f(x)) data points.
        exp : str -> The symbolic expression to evaluate in the coefficient form.
        popt : List -> The optimized parameters for the symbolic expression.

        Returns
        -------
        likelihood : np.ndarray -> The likelihood for each (a, b) pair.
        """
        likelihood = np.zeros_like(self.prior)
        a_values = self.parameter_ranges['a']
        b_values = self.parameter_ranges['b']

        for i, a in enumerate(a_values):
            for j, b in enumerate(b_values):
                replaced_exp = utils.replace_coefficients(exp, {"a": a, "b": b})
                func = utils.string_to_function(replaced_exp, self.cfg.experiment.function.num_variables)
                predicted = utils.eval_function(func, observed_data[:, :-1], self.cfg.experiment.function.num_variables)
                likelihood[i, j] = np.prod(norm.pdf(observed_data[:, -1], loc=predicted, scale=self.noise_std))

        return likelihood
    
    def update_posterior(self, likelihood: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Updates the posterior distribution using Bayes' rule and computes the information gain.

        Parameters
        ----------
        likelihood : np.ndarray -> The likelihood for each (a, b) pair.

        Returns
        -------
        posterior : np.ndarray -> The updated posterior distribution.
        information_gain : float -> The information gain from the prior to the posterior.
        """
        posterior = self.prior * likelihood

        # Normalize to ensure it's a valid probability distribution
        normalizing_constant = posterior.sum() 
        if normalizing_constant == 0:
            self.logger.warn("Posterior normalization constant is zero, most likely due to a very small likelihood. Raising exception.")
            raise ValueError("Posterior normalization constant is zero, most likely due to a very small likelihood. Raising exception.")

        posterior /= normalizing_constant

        # Compute Shannon entropy for the posterior
        posterior_entropy = np.sum(entr(posterior))

        information_gain = self.prior_entropy - posterior_entropy
        return posterior, information_gain

    def propose_design(self, info: str, only_points: str, budget_remaining: int, iteration: int) -> float:
        """
        Proposes a new experimental design, including information gain in the prompt.

        Parameters
        ----------
        info : str -> The past (x, f(x)) points and their information gains.
        budget_remaining : int -> The remaining budget.
        iteration : int -> The current iteration.

        Returns
        -------
        exp_design : float -> The proposed experimental design.
        """
        self.logger.info(f"ED Round {iteration + 1}.")
        retries_left = self.ed_retries
        if info == "":
            all_train_points = utils.string_to_array(self.og_train_points)
        else:
            all_train_points = utils.string_to_array(self.og_train_points + ", " + only_points)
        
        while retries_left > 0:
            if iteration == 0:
                prompt = self.initial_prompt
            else:
                prompt = self.base_prompt.format(budget_remaining=budget_remaining, llm_sampled_data=info)

            self.logger.info("Prompt: " + prompt)
            output = self.model.generate(prompt, return_prompt=False, image_files=None, temperature=self.temperature)
            self.logger.info("Model output: " + output)

            valid = False
            lines = output.split("\n")
            for line in lines:
                if "x" not in line:
                        self.logger.info(f"Skipping line {line} as it does not contain 'x' and is likely not the experimental design.")  
                        continue
                if "Error" in line:
                    self.logger.info(f"Skipping line {line} as it contains 'Error'.")
                    continue
                if line == "":
                    continue
                try:
                    if "x" in line:
                        words = line.split()
                        len_line = len(words)
                        for i in range(2, len_line):
                            if words[i-2] == "x" and words[i-1] == "=":
                                exp_design = float(words[i])
                                valid = True
                                break
                    else:
                        continue
                    if valid:
                        self.logger.info(f"Proposed experimental design for iteration {iteration + 1} is: {exp_design}")
                        break
                    else:
                        self.logger.warning("Found a line with x in it but it is not properly formatted.")
                except Exception as e:
                    self.logger.warning(f"Could not parse line {line}.")
                    self.logger.warning(str(e))
                    pass
            
            if not valid:
                self.logger.warning("Could not find a valid experimental design in the output. Retrying prompting.")
                retries_left -= 1
                continue
            else:
                retry = False
                exp_design = round(exp_design, self.num_digits)
                # loop through all train points and check if the proposed design is already in the list
                for point in all_train_points:
                    if point[0] == exp_design:
                        self.logger.warning("Proposed design already exists! Retrying prompting.") 
                        retries_left -= 1
                        retry = True
                        break
                if retry:
                    continue
                else:
                    return exp_design
            
        self.logger.warning("Could not find a valid experimental design in the output and no retries left. Raise exception.")
        raise Exception("Could not find a valid experimental design in the output and no retries left.")
    
    # Plot posterior distributions
    def plot_posteriors(self, iteration: int):
        fig, axs = plt.subplots(1, figsize=(18, 5))

        # Plot for (a, b)
        a, b = self.parameter_ranges['a'], self.parameter_ranges['b']
        A, B = np.meshgrid(a, b, indexing='ij')  # shape: (len(a), len(b))
        C = self.prior

        # Flatten everything to 1D
        A_flat = A.ravel()
        B_flat = B.ravel()
        C_flat = C.ravel()
        
        scatter = axs.scatter(A_flat, B_flat, c=C_flat, cmap='viridis')
        fig.colorbar(scatter, ax=axs, label='Posterior')
        axs.set_xlabel("a (volts)")
        axs.set_ylabel("b (seconds)")
        axs.set_title(f"Posterior Distribution for (a, b) (Iteration {iteration + 1})")
        axs.axvline(x=self.true_a, color='red', linestyle='--', label='True a')
        axs.axhline(y=self.true_b, color='green', linestyle='--', label='True b')
        axs.legend()
        return plt.gcf(), plt.gca()
