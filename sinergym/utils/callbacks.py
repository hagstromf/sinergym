"""Custom Callbacks for stable baselines 3 algorithms."""

import os
import gc
from typing import Optional, Union
import csv

import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization, VecNormalize, is_vecenv_wrapped
from stable_baselines3.common.logger import Logger as SB3Logger

from sinergym.utils.evaluation import evaluate_policy
from sinergym.utils.wrappers import LoggerWrapper, NormalizeObservation

from sinergym.utils.logger import Logger, configure

from sinergym.utils.rewards import GausTrapReward

from collections import OrderedDict


class LoggerCallback(BaseCallback):
    """Custom callback for plotting additional values in tensorboard.
        :param ep_rewards: Here will be stored all rewards during episode.
        :param ep_powers: Here will be stored all consumption data during episode.
        :param ep_term_comfort: Here will be stored all comfort terms (reward component) during episode.
        :param ep_term_energy: Here will be stored all energy terms (reward component) during episode.
        :param num_comfort_violation: Number of timesteps in which comfort has been violated.
        :param ep_timesteps: Each timestep during an episode, this value increment 1.
    """

    def __init__(self, sinergym_logger=False, verbose=0):
        """Custom callback for plotting additional values in tensorboard.
        Args:
            sinergym_logger (boolean): Indicate if CSVLogger inner Sinergym will be activated or not.
        """
        super(LoggerCallback, self).__init__(verbose)

        self.sinergym_logger = sinergym_logger

        self.ep_rewards = []
        self.ep_powers = []
        self.ep_term_comfort = []
        self.ep_term_energy = []
        self.num_comfort_violation = 0
        self.ep_timesteps = 0

    def _on_training_start(self):
        # sinergym logger
        if is_wrapped(self.training_env, LoggerWrapper):
            if self.sinergym_logger:
                self.training_env.env_method('activate_logger')
            else:
                self.training_env.env_method('deactivate_logger')

        # record method depending on the type of algorithm

        if 'OnPolicyAlgorithm' in self.globals.keys():
            self.record = self.logger.record
        elif 'OffPolicyAlgorithm' in self.globals.keys():
            self.record = self.logger.record_mean
        else:
            raise KeyError

    def _on_step(self) -> bool:
        info = self.locals['infos'][-1]

        # OBSERVATION
        variables = self.training_env.get_attr('variables')[0]['observation']
        # log normalized and original values
        if self.training_env.env_is_wrapped(
                wrapper_class=NormalizeObservation)[0]:
            obs_normalized = self.locals['new_obs'][-1]
            obs = self.training_env.env_method('get_unwrapped_obs')[-1]
            for i, variable in enumerate(variables):
                self.record(
                    'normalized_observation/' + variable, obs_normalized[i])
                self.record(
                    'observation/' + variable, obs[i])
        # Only original values
        else:
            obs = self.locals['new_obs'][-1]
            for i, variable in enumerate(variables):
                self.record(
                    'observation/' + variable, obs[i])

        # ACTION
        variables = self.training_env.get_attr('variables')[0]['action']
        action = None
        # sinergym action received inner its own setpoints range
        action_ = info['action_']
        try:
            # network output clipped with gym action space
            action = self.locals['clipped_actions'][-1]
        except KeyError:
            try:
                action = self.locals['action'][-1]
            except KeyError:
                try:
                    action = self.locals['actions'][-1]
                except KeyError:
                    raise KeyError(
                        'Algorithm action key in locals dict unknown.')

        if self.training_env.get_attr('flag_discrete')[0]:
            action = self.training_env.get_attr('action_mapping')[0][action]
        for i, variable in enumerate(variables):
            if action is not None:
                self.record(
                    'action/' + variable, action[i])

            self.record(
                'action_simulation/' + variable, action_[i])

        # Store episode data
        try:
            self.ep_rewards.append(self.locals['rewards'][-1])
        except KeyError:
            try:
                self.ep_rewards.append(self.locals['reward'][-1])
            except KeyError:
                print('Algorithm reward key in locals dict unknown')

        self.ep_powers.append(info['total_power'])
        self.ep_term_comfort.append(info['comfort_penalty'])
        self.ep_term_energy.append(info['total_power_no_units'])
        if(info['comfort_penalty'] != 0):
            self.num_comfort_violation += 1
        self.ep_timesteps += 1

        # If episode ends, store summary of episode and reset
        try:
            done = self.locals['dones'][-1]
        except KeyError:
            try:
                done = self.locals['done'][-1]
            except KeyError:
                print('Algorithm done key in locals dict unknown')
        if done:
            # store last episode metrics
            self.episode_metrics = {}
            self.episode_metrics['ep_length'] = self.ep_timesteps
            self.episode_metrics['cumulative_reward'] = np.sum(
                self.ep_rewards)
            self.episode_metrics['mean_reward'] = np.mean(self.ep_rewards)
            self.episode_metrics['mean_power'] = np.mean(self.ep_powers)
            self.episode_metrics['cumulative_power'] = np.sum(self.ep_powers)
            self.episode_metrics['mean_comfort_penalty'] = np.mean(
                self.ep_term_comfort)
            self.episode_metrics['cumulative_comfort_penalty'] = np.sum(
                self.ep_term_comfort)
            self.episode_metrics['mean_power_penalty'] = np.mean(
                self.ep_term_energy)
            self.episode_metrics['cumulative_power_penalty'] = np.sum(
                self.ep_term_energy)
            try:
                self.episode_metrics['comfort_violation_time(%)'] = self.num_comfort_violation / \
                    self.ep_timesteps * 100
            except ZeroDivisionError:
                self.episode_metrics['comfort_violation_time(%)'] = np.nan

            # reset episode info
            self.ep_rewards = []
            self.ep_powers = []
            self.ep_term_comfort = []
            self.ep_term_energy = []
            self.ep_timesteps = 0
            self.num_comfort_violation = 0

        # During first episode, as it not finished, it shouldn't be recording
        if hasattr(self, 'episode_metrics'):
            for key, metric in self.episode_metrics.items():
                self.logger.record(
                    'episode/' + key, metric)

        return True

    def on_training_end(self):
        if is_wrapped(self.training_env, LoggerWrapper):
            self.training_env.env_method('activate_logger')



class FedHVACLoggerCallback(BaseCallback):
    # TODO: Check parameter list and update accordingly
    """Custom callback for plotting additional values in tensorboard and to console. This is used in the fed_hvac project.
        :param ep_rewards: Here will be stored all rewards during episode.
        :param ep_powers: Here will be stored all consumption data during episode.
        :param ep_term_comfort: Here will be stored all comfort terms (reward component) during episode.
        :param ep_term_energy: Here will be stored all energy terms (reward component) during episode.
        :param num_comfort_violation: Number of timesteps in which comfort has been violated.
        :param ep_timesteps: Each timestep during an episode, this value increment 1.
    """

    def __init__(self, sinergym_logger=False, client_id=0, verbose=0):
        """Custom callback for plotting additional values in tensorboard.
        Args:
            sinergym_logger (boolean): Indicate if CSVLogger inner Sinergym will be activated or not.
        """
        super(FedHVACLoggerCallback, self).__init__(verbose)

        self.sinergym_logger = sinergym_logger
        self.client_id = client_id

        self.ep_rewards = []
        self.ep_powers = []
        self.ep_term_comfort = []
        self.ep_term_energy = []
        self.num_comfort_violation = 0
        self.ep_timesteps = 0

        # Training starts at first month of the year 
        self.current_month = 1

    def _on_training_start(self):

        # sinergym logger
        if is_wrapped(self.training_env, LoggerWrapper):
            if self.sinergym_logger:
                self.training_env.env_method('activate_logger')
            else:
                self.training_env.env_method('deactivate_logger')

        # record method depending on the type of algorithm

        if 'OnPolicyAlgorithm' in self.globals.keys():
            self.record = self.logger.record
        elif 'OffPolicyAlgorithm' in self.globals.keys():
            self.record = self.logger.record_mean
        else:
            raise KeyError

        # Record training environment used (useful for identifying which client's metrics are
        # being printed to stdout in a federated learning scenario)
        if isinstance(self.training_env, VecEnv):
            self.training_env_name = self.training_env.get_attr('simulator')[0].env_name
        else:
            self.training_env_name = self.training_env.simulator.env_name


    def _on_step(self) -> bool:
        info = self.locals['infos'][-1]

        # OBSERVATION
        variables = self.training_env.get_attr('variables')[0]['observation']

        # log normalized and original values
        if is_vecenv_wrapped(self.training_env, VecNormalize):
            obs_normalized = self.locals['new_obs'][-1]
            obs = self.training_env.get_original_obs()[0]
            for i, variable in enumerate(variables):
                self.record(
                    'normalized_observation/' + variable, obs_normalized[i])
                self.record(
                    'observation/' + variable, obs[i])

            reward = self.training_env.get_original_reward()[0]
            self.ep_rewards.append(reward)
        # Only original values
        else:
            obs = self.locals['new_obs'][-1]
            for i, variable in enumerate(variables):
                self.record(
                    'observation/' + variable, obs[i])

            try:
                self.ep_rewards.append(self.locals['rewards'][-1])
            except KeyError:
                try:
                    self.ep_rewards.append(self.locals['reward'][-1])
                except KeyError:
                    print('Algorithm reward key in locals dict unknown')

        # ACTION
        variables = self.training_env.get_attr('variables')[0]['action']
        action = None
        # sinergym action received inner its own setpoints range
        action_ = info['action_']
        try:
            # network output clipped with gym action space
            action = self.locals['clipped_actions'][-1]
        except KeyError:
            try:
                action = self.locals['action'][-1]
            except KeyError:
                try:
                    action = self.locals['actions'][-1]
                except KeyError:
                    raise KeyError(
                        'Algorithm action key in locals dict unknown.')

        if self.training_env.get_attr('flag_discrete')[0]:
            action = self.training_env.get_attr('action_mapping')[0][action]
        for i, variable in enumerate(variables):
            if action is not None:
                self.record(
                    'action/' + variable, action[i])

            self.record(
                'action_simulation/' + variable, action_[i])

        self.ep_powers.append(info['total_power'])
        self.ep_term_comfort.append(info['comfort_penalty'])
        self.ep_term_energy.append(info['total_power_no_units'])

        self.num_comfort_violation += info['comfort_violation']        
        self.ep_timesteps += 1

        # If episode ends, store summary of episode and reset
        try:
            done = self.locals['dones'][-1]
        except KeyError:
            try:
                done = self.locals['done'][-1]
            except KeyError:
                print('Algorithm done key in locals dict unknown')
        if done:
            # Record client id (for summary display)
            self.record('Client', self.client_id)
            
            # store last episode metrics
            self.episode_metrics = {}
            self.episode_metrics['ep_length'] = self.ep_timesteps
            self.episode_metrics['cumulative_reward'] = np.sum(
                self.ep_rewards)
            self.episode_metrics['mean_reward'] = np.mean(self.ep_rewards)
            self.episode_metrics['mean_power'] = np.mean(self.ep_powers)
            self.episode_metrics['cumulative_power'] = np.sum(self.ep_powers)
            self.episode_metrics['mean_comfort_penalty'] = np.mean(
                self.ep_term_comfort)
            self.episode_metrics['cumulative_comfort_penalty'] = np.sum(
                self.ep_term_comfort)
            self.episode_metrics['mean_power_penalty'] = np.mean(
                self.ep_term_energy)
            self.episode_metrics['cumulative_power_penalty'] = np.sum(
                self.ep_term_energy)
            try:
                self.episode_metrics['comfort_violation_time(%)'] = self.num_comfort_violation / \
                    self.ep_timesteps * 100
            except ZeroDivisionError:
                self.episode_metrics['comfort_violation_time(%)'] = np.nan

            # reset episode info
            self.ep_rewards = []
            self.ep_powers = []
            self.ep_term_comfort = []
            self.ep_term_energy = []
            self.ep_timesteps = 0
            self.num_comfort_violation = 0

        # During first episode, as it not finished, it shouldn't be recording
        if hasattr(self, 'episode_metrics'):
            for key, metric in self.episode_metrics.items():
                self.logger.record(
                    'episode/' + key, metric)

        return True


    def on_training_end(self):
        if is_wrapped(self.training_env, LoggerWrapper):
            self.training_env.env_method('activate_logger')


class MonthlyLoggerCallback(BaseCallback):
    # TODO: Check parameter list and update accordingly
    """Custom callback for plotting additional values in tensorboard and to console. This is used in the fed_hvac project.
        :param ep_rewards: Here will be stored all rewards during episode.
        :param ep_powers: Here will be stored all consumption data during episode.
        :param ep_term_comfort: Here will be stored all comfort terms (reward component) during episode.
        :param ep_term_energy: Here will be stored all energy terms (reward component) during episode.
        :param num_comfort_violation: Number of timesteps in which comfort has been violated.
        :param ep_timesteps: Each timestep during an episode, this value increment 1.
    """

    def __init__(self, logger: SB3Logger, client_id=0, verbose=0):
        """Custom callback for plotting additional values in tensorboard.
        Args:
            sinergym_logger (boolean): Indicate if CSVLogger inner Sinergym will be activated or not.
        """
        super(MonthlyLoggerCallback, self).__init__(verbose)

        self.monthly_logger = logger
        self.client_id = client_id

        self.month_rewards = []
        self.month_powers = []
        self.month_term_energy = []
        self.month_term_comfort = []
        self.month_num_comfort_violation = 0
        self.month_timesteps = 0
        self.total_timesteps = 0

        self.month_metrics = OrderedDict()
        self.month_metrics['month_num'] = 0

        # Training starts at first month of the year 
        self.current_month = 1

        #self.log_path = self.training_env.get_attr('simulator')[0]._env_working_dir_parent + '/monthly_progress.csv'
        progress_header_list = [
            'month_num',
            'cumulative_reward',
            'mean_reward',
            'cumulative_power_consumption',
            'mean_power_consumption',
            'cumulative_comfort_penalty',
            'mean_comfort_penalty',
            'cumulative_power_penalty',
            'mean_power_penalty',
            'comfort_violation (%)',
            #'mean_comfort_violation',
            #'std_comfort_violation',
            #'cumulative_comfort_violation',
            'length(timesteps)',
            'time_elapsed(seconds)']
        self.progress_header = ''
        for element_header in progress_header_list:
            self.progress_header += element_header + ','
        self.progress_header = self.progress_header[:-1] + '\n'

    def _on_training_start(self):
        self.training_env_name = self.training_env.get_attr('simulator')[0].env_name
        self.log_progress_file = self.training_env.get_attr('simulator')[0]._env_working_dir_parent + '/monthly_progress.csv'
        

    def _on_step(self) -> bool:
        info = self.locals['infos'][-1]

        if info['month'] != self.current_month:
            
            # store last month metrics
            self.month_metrics['month_length'] = self.month_timesteps
            self.month_metrics['cumulative_reward'] = np.sum(self.month_rewards)
            self.month_metrics['mean_reward'] = np.mean(self.month_rewards)
            self.month_metrics['mean_power'] = np.mean(self.month_powers)
            self.month_metrics['cumulative_power'] = np.sum(self.month_powers)
            self.month_metrics['mean_comfort_penalty'] = np.mean(self.month_term_comfort)
            self.month_metrics['cumulative_comfort_penalty'] = np.sum(self.month_term_comfort)
            self.month_metrics['mean_power_penalty'] = np.mean(self.month_term_energy)
            self.month_metrics['cumulative_power_penalty'] = np.sum(self.month_term_energy)
            try:
                self.month_metrics['comfort_violation_time(%)'] = self.month_num_comfort_violation / \
                    self.month_timesteps * 100
            except ZeroDivisionError:
                self.month_metrics['comfort_violation_time(%)'] = np.nan
            
            for key, metric in self.month_metrics.items():
                #self.monthly_logger.record(key, metric)
                self.monthly_logger.record('month/' + key, metric)
            
            if self.verbose > 0:
                #print(f"SELF: {self}", flush=True)
                message = f"CLIENT {self.client_id}, "
                message += f"Training environment: {self.training_env_name} \n"
                message += f"Month: {self.current_month} \n"
                message += f"Mean reward: {self.month_metrics['mean_reward']} \n"
                message += f"Mean power: {self.month_metrics['mean_power']} \n"
                message += f"Mean comfort penalty: {self.month_metrics['mean_comfort_penalty']} \n"
                message += f"Mean power penalty: {self.month_metrics['mean_power_penalty']} \n"
                message += f"comfort_violation_time(%): {self.month_metrics['comfort_violation_time(%)']} \n"
                print(message, flush=True)

            # Dump metrics to output formats
            self.monthly_logger.dump(step=self.total_timesteps)

            # Create CSV file with header if it's required for progress.csv
            if not os.path.isfile(self.log_progress_file):
                with open(self.log_progress_file, 'a', newline='\n') as file_obj:
                    file_obj.write(self.progress_header)

            # building episode row
            row_contents = [
                self.month_metrics['month_num'],
                self.month_metrics['cumulative_reward'],
                self.month_metrics['mean_reward'],
                self.month_metrics['cumulative_power'],
                self.month_metrics['mean_power'],
                self.month_metrics['cumulative_comfort_penalty'],
                self.month_metrics['mean_comfort_penalty'],
                self.month_metrics['cumulative_power_penalty'],
                self.month_metrics['mean_power_penalty'],
                self.month_metrics['comfort_violation_time(%)'],
                #ep_mean_abs_comfort,
                #ep_std_abs_comfort,
                #ep_cumulative_abs_comfort,
                self.total_timesteps,
                info['time_elapsed']]

            with open(self.log_progress_file, 'a+', newline='') as file_obj:
                # Create a writer object from csv module
                csv_writer = csv.writer(file_obj)
                # Add contents of list as last row in the csv file
                csv_writer.writerow(row_contents)

            self.month_metrics['month_num'] += 1
            self.current_month = info['month']
            self.month_rewards = []
            self.month_powers = []
            self.month_term_energy = []
            self.month_term_comfort = []
            self.month_num_comfort_violation = 0
            self.month_timesteps = 0

        # Store monthly data
        if is_vecenv_wrapped(self.training_env, VecNormalize):
            reward = self.training_env.get_original_reward()[0]
            self.month_rewards.append(reward)
        else:
            try:
                self.month_rewards.append(self.locals['rewards'][-1])
            except KeyError:
                try:
                    self.month_rewards.append(self.locals['reward'][-1])
                except KeyError:
                    print('Algorithm reward key in locals dict unknown')

        self.month_powers.append(info['total_power'])
        self.month_term_comfort.append(info['comfort_penalty'])
        self.month_term_energy.append(info['total_power_no_units'])

        self.month_num_comfort_violation += info['comfort_violation']        
        self.month_timesteps += 1
        self.total_timesteps += 1

        return True


    def on_training_end(self):
        pass


class WeeklyLoggerCallback(BaseCallback):
    # TODO: Check parameter list and update accordingly
    """Custom callback for plotting additional values in tensorboard and to console. This is used in the fed_hvac project.
        :param ep_rewards: Here will be stored all rewards during episode.
        :param ep_powers: Here will be stored all consumption data during episode.
        :param ep_term_comfort: Here will be stored all comfort terms (reward component) during episode.
        :param ep_term_energy: Here will be stored all energy terms (reward component) during episode.
        :param num_comfort_violation: Number of timesteps in which comfort has been violated.
        :param ep_timesteps: Each timestep during an episode, this value increment 1.
    """

    def __init__(self, logger: SB3Logger, client_id=0, verbose=0):
        """Custom callback for plotting additional values in tensorboard.
        Args:
            sinergym_logger (boolean): Indicate if CSVLogger inner Sinergym will be activated or not.
        """
        super(WeeklyLoggerCallback, self).__init__(verbose)

        self.weekly_logger = logger
        self.client_id = client_id

        self.week_rewards = []
        self.week_powers = []
        self.week_term_energy = []
        self.week_term_comfort = []
        self.week_num_comfort_violation = 0
        self.week_timesteps = 1
        self.total_timesteps = 1

        self.week_metrics = OrderedDict()
        self.week_metrics['week_num'] = 0

        # Training starts at first week of the year 
        self.current_day = 1
        self.day_count = 0

        self.new_week = False

        self.current_episode = 0

        progress_header_list = [
            'month_num',
            'cumulative_reward',
            'mean_reward',
            'cumulative_power_consumption',
            'mean_power_consumption',
            'cumulative_comfort_penalty',
            'mean_comfort_penalty',
            'cumulative_power_penalty',
            'mean_power_penalty',
            'comfort_violation (%)',
            #'mean_comfort_violation',
            #'std_comfort_violation',
            #'cumulative_comfort_violation',
            'length(timesteps)',
            'time_elapsed(seconds)']
        self.progress_header = ''
        for element_header in progress_header_list:
            self.progress_header += element_header + ','
        self.progress_header = self.progress_header[:-1] + '\n'

        #self.log_path = self.training_env.get_attr('simulator')[0]._env_working_dir_parent + '/weekly_progress.csv'

    def _on_training_start(self):
        self.training_env_name = self.training_env.get_attr('simulator')[0].env_name
        self.log_progress_file = self.training_env.get_attr('simulator')[0]._env_working_dir_parent + '/weekly_progress.csv'
        

    def _on_step(self) -> bool:
        info = self.locals['infos'][-1]

        if info['day'] != self.current_day:
            self.current_day = info['day']
            self.day_count += 1
            if (self.day_count + self.current_episode * 365) % 7 == 0:
                self.new_week = True
 
        if self.new_week:   
            # store last week metrics
            self.week_metrics['week_length'] = self.week_timesteps
            self.week_metrics['cumulative_reward'] = np.sum(self.week_rewards)
            self.week_metrics['mean_reward'] = np.mean(self.week_rewards)
            self.week_metrics['mean_power'] = np.mean(self.week_powers)
            self.week_metrics['cumulative_power'] = np.sum(self.week_powers)
            self.week_metrics['mean_comfort_penalty'] = np.mean(self.week_term_comfort)
            self.week_metrics['cumulative_comfort_penalty'] = np.sum(self.week_term_comfort)
            self.week_metrics['mean_power_penalty'] = np.mean(self.week_term_energy)
            self.week_metrics['cumulative_power_penalty'] = np.sum(self.week_term_energy)
            try:
                self.week_metrics['comfort_violation_time(%)'] = self.week_num_comfort_violation / \
                    self.week_timesteps * 100
            except ZeroDivisionError:
                self.week_metrics['comfort_violation_time(%)'] = np.nan
            
            for key, metric in self.week_metrics.items():
                #self.weekly_logger.record(key, metric)
                self.weekly_logger.record('week/' + key, metric)
            
            if self.verbose > 0:
                message = f"CLIENT {self.client_id}, "
                message += f"Training environment: {self.training_env_name} \n"
                message += f"Week: {self.week_metrics['week_num'] + 1} \n"
                message += f"Mean reward: {self.week_metrics['mean_reward']} \n"
                message += f"Mean power: {self.week_metrics['mean_power']} \n"
                message += f"Mean comfort penalty: {self.week_metrics['mean_comfort_penalty']} \n"
                message += f"Mean power penalty: {self.week_metrics['mean_power_penalty']} \n"
                message += f"comfort_violation_time(%): {self.week_metrics['comfort_violation_time(%)']} \n"
                print(message, flush=True)

            # Dump metrics to output formats
            self.weekly_logger.dump(step=self.total_timesteps)

            # Create CSV file with header if it's required for progress.csv
            if not os.path.isfile(self.log_progress_file):
                with open(self.log_progress_file, 'a', newline='\n') as file_obj:
                    file_obj.write(self.progress_header)

            # building episode row
            row_contents = [
                self.week_metrics['week_num'],
                self.week_metrics['cumulative_reward'],
                self.week_metrics['mean_reward'],
                self.week_metrics['cumulative_power'],
                self.week_metrics['mean_power'],
                self.week_metrics['cumulative_comfort_penalty'],
                self.week_metrics['mean_comfort_penalty'],
                self.week_metrics['cumulative_power_penalty'],
                self.week_metrics['mean_power_penalty'],
                self.week_metrics['comfort_violation_time(%)'],
                #ep_mean_abs_comfort,
                #ep_std_abs_comfort,
                #ep_cumulative_abs_comfort,
                self.total_timesteps,
                info['time_elapsed']]

            with open(self.log_progress_file, 'a+', newline='') as file_obj:
                # Create a writer object from csv module
                csv_writer = csv.writer(file_obj)
                # Add contents of list as last row in the csv file
                csv_writer.writerow(row_contents)

            self.week_metrics['week_num'] += 1
            self.week_rewards = []
            self.week_powers = []
            self.week_term_energy = []
            self.week_term_comfort = []
            self.week_num_comfort_violation = 0
            self.week_timesteps = 0

            self.new_week = False 

        # Store weekly data
        if is_vecenv_wrapped(self.training_env, VecNormalize):
            reward = self.training_env.get_original_reward()[0]
            self.week_rewards.append(reward)
        else:
            try:
                self.week_rewards.append(self.locals['rewards'][-1])
            except KeyError:
                try:
                    self.week_rewards.append(self.locals['reward'][-1])
                except KeyError:
                    print('Algorithm reward key in locals dict unknown')

        self.week_powers.append(info['total_power'])
        self.week_term_comfort.append(info['comfort_penalty'])
        self.week_term_energy.append(info['total_power_no_units'])

        self.week_num_comfort_violation += info['comfort_violation']        
        self.week_timesteps += 1
        self.total_timesteps += 1

        #self.current_day += 1

        try:
            done = self.locals['dones'][-1]
        except KeyError:
            try:
                done = self.locals['done'][-1]
            except KeyError:
                print('Algorithm done key in locals dict unknown')
        if done:
            self.current_episode += 1

        return True

        
    def on_training_end(self):
        pass


# TODO: Make LoggeEvalCalback compatible with VecNormalization
class LoggerEvalCallback(EvalCallback):
    """Callback for evaluating an agent.
        :param eval_env: The environment used for initialization
        :param callback_on_new_best: Callback to trigger when there is a new best model according to the mean reward
        :param n_eval_episodes: The number of episodes to test the agent
        :param eval_freq: Evaluate the agent every eval freq call of the callback.
        :param log_path: Path to a folder where the evaluations (evaluations.npz) will be saved. It will be updated at each evaluation.
        :param best_model_save_path: Path to a folder where the best model according to performance on the eval env will be saved.
        :param deterministic: Whether the evaluation should use a stochastic or deterministic actions.
        :param render: Whether to render or not the environment during evaluation
        :param verbose:
        :param warn: Passed to evaluate policy (warns if eval env has not been wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super(
            LoggerEvalCallback,
            self).__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn)
        self.evaluations_power_consumption = []
        self.evaluations_comfort_violation = []
        self.evaluations_comfort_penalty = []
        self.evaluations_power_penalty = []
        self.evaluation_metrics = {}

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            # Reset success rate buffer
            self._is_success_buffer = []
            #episodes_rewards, episodes_lengths, episodes_powers, episodes_comfort_violations, episodes_comfort_penalties, episodes_power_penalties
            episodes_data = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                callback=None,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(
                    episodes_data['episodes_rewards'])
                self.evaluations_length.append(
                    episodes_data['episodes_lengths'])
                self.evaluations_power_consumption.append(
                    episodes_data['episodes_powers'])
                self.evaluations_comfort_violation.append(
                    episodes_data['episodes_comfort_violations'])
                self.evaluations_comfort_penalty.append(
                    episodes_data['episodes_comfort_penalties'])
                self.evaluations_power_penalty.append(
                    episodes_data['episodes_power_penalties'])

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    ep_powers=self.evaluations_power_consumption,
                    ep_comfort_violations=self.evaluations_comfort_violation,
                    episodes_comfort_penalties=self.evaluations_comfort_penalty,
                    episodes_power_penalties=self.evaluations_power_penalty,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(
                episodes_data['episodes_rewards']), np.std(
                episodes_data['episodes_rewards'])
            mean_ep_length, std_ep_length = np.mean(
                episodes_data['episodes_lengths']), np.std(
                episodes_data['episodes_lengths'])

            self.evaluation_metrics['mean_rewards'] = mean_reward
            self.evaluation_metrics['std_rewards'] = std_reward
            self.evaluation_metrics['mean_ep_length'] = mean_ep_length
            self.evaluation_metrics['mean_power_consumption'] = np.mean(
                episodes_data['episodes_powers'])
            self.evaluation_metrics['comfort_violation(%)'] = np.mean(
                episodes_data['episodes_comfort_violations'])
            self.evaluation_metrics['comfort_penalty'] = np.mean(
                episodes_data['episodes_comfort_penalties'])
            self.evaluation_metrics['power_penalty'] = np.mean(
                episodes_data['episodes_power_penalties'])

            if self.verbose > 0:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(
                    f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            for key, metric in self.evaluation_metrics.items():
                self.logger.record('eval/' + key, metric)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(
                        os.path.join(
                            self.best_model_save_path,
                            'model.zip'))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True