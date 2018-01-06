import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

from bath.world import World




class DonEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, my_world):
        self.bath = my_world
        self.max_saccade_step = 20
        self.action_space = spaces.Box(low=-self.max_saccade_step, high=self.max_saccade_step, shape=(2,))
        self.observation_space = self._observation_space()


    def _observation_space(self):
        len_of_state = self.bath.get_state_len()
        len_of_extra = 2
        state_low = np.full(len_of_state, fill_value=-0.1)
        state_high = np.full(len_of_state, fill_value=1.1)
        extra_vector_low = np.full(len_of_extra, fill_value=-self.max_saccade_step)
        extra_vector_high = np.full(len_of_extra, fill_value=self.max_saccade_step)

        state_low = np.concatenate((state_low, extra_vector_low))
        state_high = np.concatenate((state_high, extra_vector_high))
        return spaces.Box(low=state_low, high=state_high)

    def _step(self, action):
        """Run one timestep of the environment's dynamics. When end of
               episode is reached, you are responsible for calling `reset()`
               to reset this environment's state.
               Accepts an action and returns a tuple (observation, reward, done, info).
               Args:
                   action (object): an action provided by the environment
               Returns:
                   observation (object): agent's observation of the current environment
                   reward (float) : amount of reward returned after previous action
                   done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
                   info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        assert self.action_space.contains(action)
        observation, reward, done = self.bath.step(action)
        return observation, reward, done, None

    def _reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
                space.
        """
        return self.bath.reset()

    def _render(self, mode='human', close=False):
       print("render...")

