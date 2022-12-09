"""Implementation of custom Gym environments."""

from collections import deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import gym
import numpy as np

from sinergym.utils.common import is_wrapped
from sinergym.utils.logger import CSVLogger


class NormalizeObservation(gym.ObservationWrapper):

    def __init__(self,
                 env: Any,
                 ranges: Dict[str, Sequence[Any]]):
        """Observations normalized to range [0, 1].

        Args:
            env (Any): Original Sinergym environment.
            ranges (Dict[str, Sequence[Any]]): Observation variables ranges to apply normalization (rely on environment).
        """
        super(NormalizeObservation, self).__init__(env)
        self.unwrapped_observation = None
        self.ranges = ranges

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Applies normalization to observation.

        Args:
            obs (np.ndarray): Original observation.

        Returns:
            np.ndarray: Normalized observation.
        """
        # Save original obs in class attribute
        self.unwrapped_observation = obs.copy()

        # NOTE: If you want to record day, month and hour, you should add that
        # variables as keys
        for i, variable in enumerate(self.env.variables['observation']):
            # Check if we are using forecasted values. Forecasted values have same range
            # as their counterpart at the current timestep, so we can just fetch the range
            # using the respective current step key
            if variable.startswith('Forecasted Outdoor Air Drybulb'):
                variable = 'Site Outdoor Air Drybulb Temperature(Environment)'
            elif variable.startswith('Forecasted Outdoor Air Relative Humidity'):
                variable = 'Site Outdoor Air Relative Humidity(Environment)'

            # normalization (handle DivisionbyZero Error)
            if(self.ranges[variable][1] - self.ranges[variable][0] == 0):
                obs[i] = max(
                    self.ranges[variable][0], min(
                        obs[i], self.ranges[variable][1]))
            else:
                obs[i] = (obs[i] - self.ranges[variable][0]) / \
                    (self.ranges[variable][1] - self.ranges[variable][0])

            # If value is out
            if np.isnan(obs[i]):
                obs[i] = 0
            elif obs[i] > 1:
                obs[i] = 1
            elif obs[i] < 0:
                obs[i] = 0
        # Return obs values in the SAME ORDER than obs argument.
        return np.array(obs)

    def get_unwrapped_obs(self) -> Optional[np.ndarray]:
        """Get last environment observation without normalization.

        Returns:
            Optional[np.ndarray]: Last original observation. If it is the first observation, this value is None.
        """
        return self.unwrapped_observation

    get_original_obs = get_unwrapped_obs


class MultiObsWrapper(gym.Wrapper):

    def __init__(self, env: Any, n: int = 5, flatten: bool = True) -> None:
        """Stack of observations.

        Args:
            env (Any): Original Gym environment.
            n (int, optional): Number of observations to be stacked. Defaults to 5.
            flatten (bool, optional): Whether or not flat the observation vector. Defaults to True.
        """
        super(MultiObsWrapper, self).__init__(env)
        self.n = n
        self.ind_flat = flatten
        self.history = deque([], maxlen=n)
        shape = env.observation_space.shape
        new_shape = (shape[0] * n,) if flatten else ((n,) + shape)
        self.observation_space = gym.spaces.Box(
            low=-5e6, high=5e6, shape=new_shape, dtype=np.float32)

    def reset(self) -> np.ndarray:
        """Resets the environment.

        Returns:
            np.ndarray: Stacked previous observations.
        """
        obs = self.env.reset()
        for _ in range(self.n):
            self.history.append(obs)
        return self._get_obs()

    def step(self, action: Union[int, np.ndarray]
             ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Performs the action in the new environment.

        Args:
            action (Union[int, np.ndarray]): Action to be executed in environment.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: Tuple with next observation, reward, bool for terminated episode and dict with extra information.
        """

        observation, reward, done, info = self.env.step(action)
        self.history.append(observation)
        return self._get_obs(), reward, done, info

    def _get_obs(self) -> np.ndarray:
        """Get observation history.

        Returns:
            np.array: Array of previous observations.
        """
        if self.ind_flat:
            return np.array(self.history).reshape(-1,)
        else:
            return np.array(self.history)


class LoggerWrapper(gym.Wrapper):

    def __init__(
        self,
        env: Any,
        logger_class: Callable = CSVLogger,
        monitor_header: Optional[List[str]] = None,
        progress_header: Optional[List[str]] = None,
        flag: bool = True,
    ):
        """CSVLogger to log interactions with environment.

        Args:
            env (Any): Original Gym environment.
            logger_class (CSVLogger): CSV Logger class to use to log all information.
            monitor_header: Header for monitor.csv in each episode. Default is None (default format).
            progress_header: Header for progress.csv in whole simulation. Default is None (default format).
            flag (bool, optional): State of logger (activate or deactivate). Defaults to True.
        """
        gym.Wrapper.__init__(self, env)
        # Headers for csv logger
        monitor_header_list = monitor_header if monitor_header is not None else [
            'timestep'] + env.variables['observation'] + env.variables['action'] + ['time (seconds)', 'reward', 'power_penalty', 'comfort_penalty', 'abs_comfort', 'done']
        self.monitor_header = ''
        for element_header in monitor_header_list:
            self.monitor_header += element_header + ','
        self.monitor_header = self.monitor_header[:-1]
        progress_header_list = progress_header if progress_header is not None else [
            'episode_num',
            'cumulative_reward',
            'mean_reward',
            'cumulative_power_consumption',
            'mean_power_consumption',
            'cumulative_comfort_penalty',
            'mean_comfort_penalty',
            'cumulative_power_penalty',
            'mean_power_penalty',
            'comfort_violation (%)',
            'mean_comfort_violation',
            'std_comfort_violation',
            'cumulative_comfort_violation',
            'length(timesteps)',
            'time_elapsed(seconds)']
        self.progress_header = ''
        for element_header in progress_header_list:
            self.progress_header += element_header + ','
        self.progress_header = self.progress_header[:-1]

        # Create simulation logger, by default is active (flag=True)
        self.logger = logger_class(
            monitor_header=self.monitor_header,
            progress_header=self.progress_header,
            log_progress_file=env.simulator._env_working_dir_parent +
            '/progress.csv',
            flag=flag)

    def step(self, action: Union[int, np.ndarray]
             ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Step the environment. Logging new information

        Args:
            action (Union[int, np.ndarray]): Action executed in step

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: Tuple with next observation, reward, bool for terminated episode and dict with extra information.
        """
        obs, reward, done, info = self.env.step(action)
        # We added some extra values (month,day,hour) manually in env, so we
        # need to delete them.
        if is_wrapped(self, NormalizeObservation):
            # Record action and new observation in simulator's csv
            self.logger.log_step_normalize(
                obs=obs,
                action=info['action_'],
                reward=reward,
                done=done,
                info=info)
            # Record original observation too
            self.logger.log_step(
                obs=self.env.get_unwrapped_obs(),
                action=info['action_'],
                reward=reward,
                done=done,
                info=info)
        else:
            # Only record observation without normalization
            self.logger.log_step(
                obs=obs,
                action=info['action_'],
                reward=reward,
                done=done,
                info=info)

        return obs, reward, done, info

    def reset(self) -> np.ndarray:
        """Resets the environment. Recording episode summary in logger

        Returns:
            np.ndarray: First observation given
        """
        # It isn't first episode simulation, so we can logger last episode
        if self.env.simulator._episode_existed:
            self.env.simulator.logger_main.debug(
                'End of episode, recording summary (progress.csv) if logger is active')
            self.logger.log_episode(episode=self.env.simulator._epi_num)

        # Then, reset environment
        obs = self.env.reset()

        # Create monitor.csv for information of this episode
        self.env.simulator.logger_main.debug(
            'Creating monitor.csv for current episode (episode ' + str(
                self.env.simulator._epi_num) + ') if logger is active')
        self.logger.set_log_file(
            self.env.simulator._eplus_working_dir + '/monitor.csv')

        if is_wrapped(self, NormalizeObservation):
            # Store initial state of simulation (normalized)
            self.logger.log_step_normalize(obs=obs,
                                           action=[None for _ in range(
                                               len(self.env.variables['action']))],
                                           reward=None,
                                           done=False,
                                           info=None)
            # And store original obs
            self.logger.log_step(obs=self.env.get_unwrapped_obs(),
                                 action=[None for _ in range(
                                     len(self.env.variables['action']))],
                                 reward=None,
                                 done=False,
                                 info=None)
        else:
            # Only store original step
            self.logger.log_step(obs=obs,
                                 action=[None for _ in range(
                                     len(self.env.variables['action']))],
                                 reward=None,
                                 done=False,
                                 info=None)

        return obs

    def close(self) -> None:
        """Recording last episode summary and close env.
        """
        # Record last episode summary before end simulation
        self.env.simulator.logger_main.debug(
            'End of episode, recording summary (progress.csv) if logger is active')
        self.logger.log_episode(episode=self.env.simulator._epi_num)

        # Then, close env
        self.env.close()

    def activate_logger(self) -> None:
        """Activate logger if its flag False.
        """
        self.logger.activate_flag()

    def deactivate_logger(self) -> None:
        """Deactivate logger if its flag True.
        """
        self.logger.deactivate_flag()
