import sys
import numpy as np

from omegaconf import DictConfig
from mloggers import MultiLogger
from typing import Any



class ED(object):
    """
    Experimental design class for performing the ED phase. 
    """
    def __init__(self, cfg: DictConfig, logger: MultiLogger, train_pts: str) -> None:
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

        self.train_domain_d = cfg.ED.train_domain_d
        self.num_digits = cfg.initialization.num_data_digits
        self.min_training_d = cfg.ED.train_domain_d[0]
        self.max_training_d = cfg.ED.train_domain_d[1]

    def propose_random_design(self, iteration: int) -> float:
        """
        Proposes a new experimental design by randomly sampling from the domain. Serves as a 
        benchmark for comparison against the LLM-guided sampling.

        Parameters
        ----------
        iteration is to keep track of which iteration we are performing for logging purposes.

        Returns
        -------
        The new experimental design with two decimal digits.
        """
        self.logger.info(f"ED Round {iteration + 1}.")
        exp_design = round(np.random.uniform(self.min_training_d, self.max_training_d), self.num_digits)
        self.logger.info(f"Randomly sampled x_0 = {exp_design} from the domain ({self.min_training_d}, {self.max_training_d})...")
        return exp_design

