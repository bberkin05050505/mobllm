import sys
import utils

from omegaconf import DictConfig
from mloggers import MultiLogger
from typing import Any



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

    
    def propose_design(self, info: str, budget_remaining: int, iteration: int) -> float:
        """
        Prompts the LLM for experimental design, info to be included from past iterations. 
        Points sampled during initialization are passed into the prompt separately.

        Parameters
        ----------
        For the base ED algorithm, info contains the past (x, f(x)) points that were sampled by the LLM itself. 
        iteration is to keep track of which iteration we are performing for logging purposes. 

        Returns
        -------
        The new experimental design with two decimal digits.
        """
        self.logger.info(f"ED Round {iteration + 1}.")
        retries_left = self.ed_retries
        if info == "":
            all_train_points = utils.string_to_array(self.og_train_points)
        else:
            all_train_points = utils.string_to_array(self.og_train_points + ", " + info)
            
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
