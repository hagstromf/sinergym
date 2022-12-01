"""Implementation of reward functions."""


from datetime import datetime
from math import exp
from typing import Any, Dict, List, Tuple, Union

from gym import Env


class BaseReward(object):

    def __init__(self, env):
        """
        Base reward class.

        All reward functions should inherit from this class.

        Args:
            env (Env): Gym environment.
        """
        self.env = env

    def __call__(self):
        """Method for calculating the reward function."""
        raise NotImplementedError(
            "Reward class must have a `__call__` method.")


class LinearReward(BaseReward):

    def __init__(
        self,
        env: Env,
        temperature_variable: Union[str, list],
        energy_variable: str,
        range_comfort_winter: Tuple[int, int],
        range_comfort_summer: Tuple[int, int],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.5,
        lambda_energy: float = 1e-4,
        lambda_temperature: float = 1.0
    ):
        """
        Linear reward function.

        It considers the energy consumption and the absolute difference to temperature comfort.

        .. math::
            R = - W * lambda_E * power - (1 - W) * lambda_T * (max(T - T_{low}, 0) + max(T_{up} - T, 0))

        Args:
            env (Env): Gym environment.
            temperature_variable (Union[str, list]): Name(s) of the temperature variable(s).
            energy_variable (str): Name of the energy/power variable.
            range_comfort_winter (Tuple[int,int]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[int,int]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
        """

        super(LinearReward, self).__init__(env)

        # Name of the variables
        self.temp_name = temperature_variable
        self.energy_name = energy_variable

        # Reward parameters
        self.range_comfort_winter = range_comfort_winter
        self.range_comfort_summer = range_comfort_summer
        self.W_energy = energy_weight
        self.lambda_energy = lambda_energy
        self.lambda_temp = lambda_temperature

        # Summer period
        self.summer_start = summer_start  # (month,day)
        self.summer_final = summer_final  # (month,day)

        self.temp_range = range_comfort_winter

    def __call__(self) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate the reward function.

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """
        # Current observation
        obs_dict = self.env.obs_dict.copy()

        # Energy term
        reward_energy = - self.lambda_energy * obs_dict[self.energy_name]

        # Comfort
        comfort, temps, comfort_violation = self._get_comfort(obs_dict)
        reward_comfort = - self.lambda_temp * comfort

        # Weighted sum of both terms
        reward = self.W_energy * reward_energy + \
            (1.0 - self.W_energy) * reward_comfort

        reward_terms = {
            'reward_energy': reward_energy,
            'total_energy': obs_dict[self.energy_name],
            'reward_comfort': reward_comfort,
            'abs_comfort': comfort,
            'temperatures': temps, 
            'comfort_violation': comfort_violation
        }

        return reward, reward_terms

    def _get_comfort(self,
                     obs_dict: Dict[str,
                                    Any]) -> Tuple[float,
                                                   List[float], bool]:
        """Calculate the comfort term of the reward.

        Returns:
            Tuple[float, List[float], bool]: comfort penalty, List with temperatures used, and comfort violation.
        """

        month = obs_dict['month']
        day = obs_dict['day']
        year = obs_dict['year']
        current_dt = datetime(year, month, day)

        # Periods
        summer_start_date = datetime(
            year,
            self.summer_start[0],
            self.summer_start[1])
        summer_final_date = datetime(
            year,
            self.summer_final[0],
            self.summer_final[1])

        if current_dt >= summer_start_date and current_dt <= summer_final_date:
            self.temp_range = self.range_comfort_summer
        else:
            self.temp_range = self.range_comfort_winter

        temps = [v for k, v in obs_dict.items() if k in self.temp_name]
        comfort = 0.0
        for T in temps:
            if T < self.temp_range[0] or T > self.temp_range[1]:
                comfort += min(abs(self.temp_range[0] - T), abs(T - self.temp_range[1]))

        # Check if temperature comfort has been violated
        comfort_violation = True if comfort != 0 else False    

        return comfort, temps, comfort_violation


class ExpReward(LinearReward):

    def __init__(
        self,
        env: Env,
        temperature_variable: Union[str, list],
        energy_variable: str,
        range_comfort_winter: Tuple[int, int],
        range_comfort_summer: Tuple[int, int],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.5,
        lambda_energy: float = 1e-4,
        lambda_temperature: float = 1.0
    ):
        """
        Reward considering exponential absolute difference to temperature comfort.

        .. math::
            R = - W * lambda_E * power - (1 - W) * lambda_T * exp( (max(T - T_{low}, 0) + max(T_{up} - T, 0)) )

        Args:
            env (Env): Gym environment.
            temperature_variable (Union[str, list]): Name(s) of the temperature variable(s).
            energy_variable (str): Name of the energy/power variable.
            range_comfort_winter (Tuple[int,int]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[int,int]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
        """

        super(ExpReward, self).__init__(
            env,
            temperature_variable,
            energy_variable,
            range_comfort_winter,
            range_comfort_summer,
            summer_start,
            summer_final,
            energy_weight,
            lambda_energy,
            lambda_temperature
        )

        self.temp_range = range_comfort_winter

    def _get_comfort(self,
                     obs_dict: Dict[str,
                                    Any]) -> Tuple[float,
                                                   List[float], bool]:
        """Calculate the comfort term of the reward.

        Returns:
            Tuple[float, List[float], bool]: comfort penalty, List with temperatures used, and comfort violation.
        """

        month = obs_dict['month']
        day = obs_dict['day']
        year = obs_dict['year']
        current_dt = datetime(year, month, day)

        # Periods
        summer_start_date = datetime(
            year,
            self.summer_start[0],
            self.summer_start[1])
        summer_final_date = datetime(
            year,
            self.summer_final[0],
            self.summer_final[1])

        if current_dt >= summer_start_date and current_dt <= summer_final_date:
            self.temp_range = self.range_comfort_summer
        else:
            self.temp_range = self.range_comfort_winter

        temps = [v for k, v in obs_dict.items() if k in self.temp_name]
        comfort = 0.0
        for T in temps:
            if T < self.temp_range[0] or T > self.temp_range[1]:
                comfort += exp(min(abs(self.temp_range[0] - T),
                                   abs(T - self.temp_range[1])))

        # Check if temperature comfort has been violated
        comfort_violation = True if comfort != 0 else False    

        return comfort, temps, comfort_violation


class GausTrapReward(BaseReward):

    def __init__(
        self,
        env: Env,
        temperature_variable: Union[str, list],
        energy_variable: Union[str, list],
        range_comfort_winter: Tuple[int, int],
        range_comfort_summer: Tuple[int, int],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        T_targets: Tuple[float, float] = (None, None),
        lambda_energy: float = 1e-5,
        lambda_1: float = 0.2,
        lambda_2: float = 0.1
    ):
        """
        The reward function proposed in article "Experimental evaluation of model-free reinforcement learning algorithms for 
        continuous HVAC control" by Biemann et al.. This reward considers a mix of a guassian and trapezoid function
        for the temperature comfort. 

        .. math::
            R_tot = sum(R_i) - lambda_energy * sum(P_j), 
            where R_i is the temperature reward of zone i and P_j is the value of energy/power variable j.

            R_i is given by the gaussian-trapezoid mixed function:

                R_i = exp(-lambda_1 * (T_i - T_target)²) - lambda_2 * (max(T_{low} - T_i, 0) + max(T_i - T_{up}, 0))

        Args:
            env (Env): Gym environment.
            temperature_variable (Union[str, list]): Name(s) of the temperature variable(s).
            energy_variable (Union[str, list]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[int,int]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[int,int]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            T_targets (Tuple[float, float]): Target temperatures of winter and summer periods respectively. Defaults to None, in which case
                                             the midpoint of the comfort range is chosen.
            lambda_energy(float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-5.
            lambda_1 (float, optional): Constant for weighting the gaussian part of the comfort expression. Defaults to 0.2.
            lambda_2 (float, optional): Constant for weighting the trapezoid part of the comfort expression. Defaults to 0.1.

        """

        super(GausTrapReward, self).__init__(env)
        
        # Name of the variables
        self.temp_name = temperature_variable
        self.energy_name = energy_variable

        # Reward parameters
        self.range_comfort_winter = range_comfort_winter
        self.range_comfort_summer = range_comfort_summer

        if T_targets[0] is None:
            self.T_target_winter = sum(range_comfort_winter) / 2
        else:
            self.T_target_winter = T_targets[0]

        if T_targets[1] is None:
            self.T_target_summer = sum(range_comfort_summer) / 2
        else:
            self.T_target_summer = T_targets[1]

        print(f"Target temperatures:\n   Winter: {self.T_target_winter}\n   Summer: {self.T_target_summer}")

        self.lambda_energy = lambda_energy
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        # Summer period
        self.summer_start = summer_start  # (month,day)
        self.summer_final = summer_final  # (month,day)

        self.temp_range = range_comfort_winter

    def __call__(self) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate the reward function.

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """
        # Current observation
        obs_dict = self.env.obs_dict.copy()

        # Energy term
        energies = [v for k, v in obs_dict.items() if k in self.energy_name]
        reward_energy = - self.lambda_energy * sum(energies)

        # Comfort term
        reward_comfort, temps, comfort_violation = self._get_comfort(obs_dict)

        # Sum of both terms
        reward = reward_comfort + reward_energy

        reward_terms = {
            'reward_energy': reward_energy,
            'total_energy': sum(energies),
            'reward_comfort': reward_comfort,
            'abs_comfort': reward_comfort,
            'temperatures': temps,
            'comfort_violation': comfort_violation
        }

        return reward, reward_terms

    def _get_comfort(self, obs_dict: Dict[str, Any]) -> Tuple[float, List[float], bool]:
        """Calculate the comfort term of the reward.

        Returns:
            Tuple[float, List[float], bool]: comfort penalty, List with temperatures used, and comfort violation.
        """

        month = obs_dict['month']
        day = obs_dict['day']
        year = obs_dict['year']
        current_dt = datetime(year, month, day)

        # Periods
        summer_start_date = datetime(
            year,
            self.summer_start[0],
            self.summer_start[1])
        summer_final_date = datetime(
            year,
            self.summer_final[0],
            self.summer_final[1])

        if current_dt >= summer_start_date and current_dt <= summer_final_date:
            self.temp_range = self.range_comfort_summer
            T_target = self.T_target_summer
        else:
            self.temp_range = self.range_comfort_winter
            T_target = self.T_target_winter

        # Calculate comfort term
        temps = [v for k, v in obs_dict.items() if k in self.temp_name]
        comfort = 0.0
        for T in temps:
            comfort += exp(-self.lambda_1 * (T - T_target)**2) - self.lambda_2 * (max(self.temp_range[0] - T, 0) + max(T - self.temp_range[1], 0))

        # Check if comfort range has been violated
        comfort_violation = self._check_comfort_violation(temps, self.temp_range)

        return comfort, temps, comfort_violation

    def _check_comfort_violation(self, temps: List[float], temp_range: List[float]) -> bool:
        """Check if the temperature comfort has been violated in any zone.
        
        Returns:
            bool: Boolean value indicating if comfort has been vioolated or not
        """
        comfort_violation = False

        for T in temps:
            if temp_range[0] >= T or T >= temp_range[1]:
                comfort_violation = True
                break
        
        return comfort_violation


def test_GausTrapReward():
    import gym
    from pytest import approx

    reward_kwargs = {
        'temperature_variable': [
            'Zone Air Temperature(West Zone)',
            'Zone Air Temperature(East Zone)'
            ],
        'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
        'range_comfort_winter': (22, 25),
        'range_comfort_summer': (22, 25)
        }

    env_name = 'Eplus-datacenter-hot-continuous-strict-v1'

    env = gym.make(env_name, reward=GausTrapReward, reward_kwargs=reward_kwargs)
 
    rew_func = env.reward_fn
    obs_dict = {'month': 1, 'day': 1, 'year':1991, 'Zone Air Temperature(West Zone)': 0, 'Zone Air Temperature(East Zone)': 0}
    test_obs = [[23.5, 23.5], [16.0, 16.0], [30.0, 30.0]]
    test_comforts = []
    test_violations = []

    for i in range(3):
        obs_dict['Zone Air Temperature(West Zone)'] = test_obs[i][0]
        obs_dict['Zone Air Temperature(East Zone)'] = test_obs[i][1]

        comfort, _, comfort_violation = rew_func._get_comfort(obs_dict)
        test_comforts.append(comfort)
        test_violations.append(comfort_violation)

    env.close()

    assert test_comforts[0] == approx(2.0)
    assert test_comforts[1] == approx(-1.2, 0.005)
    assert test_comforts[2] == approx(-1.0, 0.005)

    assert test_violations[0] == False
    assert test_violations[1] == True
    assert test_violations[2] == True


class HourlyLinearReward(LinearReward):

    def __init__(
        self,
        env: Env,
        temperature_variable: Union[str, list],
        energy_variable: str,
        range_comfort_winter: Tuple[int, int],
        range_comfort_summer: Tuple[int, int],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        min_energy_weight: float = 0.5,
        lambda_energy: float = 1e-4,
        lambda_temperature: float = 1.0,
        range_comfort_hours: tuple = (9, 19),
    ):
        """
        Linear reward function with a time-dependent weight for consumption and energy terms.

        Args:
            env (Env): Gym environment.
            temperature_variable (Union[str, list]): Name(s) of the temperature variable(s).
            energy_variable (str): Name of the energy/power variable.
            range_comfort_winter (Tuple[int,int]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[int,int]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            min_energy_weight (float, optional): Minimum weight given to the energy term. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
            range_comfort_hours (tuple, optional): Hours where thermal comfort is considered. Defaults to (9, 19).
        """

        super(HourlyLinearReward, self).__init__(
            env,
            temperature_variable,
            energy_variable,
            range_comfort_winter,
            range_comfort_summer,
            summer_start,
            summer_final,
            min_energy_weight,
            lambda_energy,
            lambda_temperature
        )

        # Reward parameters
        self.range_comfort_hours = range_comfort_hours

    def __call__(self) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function.

        Returns:
            Tuple[float, Dict[str, Any]]: Reward and dict with reward terms.
            """
        # Current observation
        obs_dict = self.env.obs_dict.copy()

        # Energy term
        reward_energy = - self.lambda_energy * obs_dict[self.energy_name]

        # Comfort
        comfort, temps = self._get_comfort(obs_dict)
        reward_comfort = - self.lambda_temp * comfort

        # Determine energy weight depending on the hour
        hour = obs_dict['hour']
        if hour >= self.range_comfort_hours[0] and hour <= self.range_comfort_hours[1]:
            weight = self.W_energy
        else:
            weight = 1.0

        # Weighted sum of both terms
        reward = weight * reward_energy + (1.0 - weight) * reward_comfort

        reward_terms = {
            'reward_energy': reward_energy,
            'total_energy': obs_dict[self.energy_name],
            'reward_comfort': reward_comfort,
            'temperatures': temps
        }

        return reward, reward_terms
