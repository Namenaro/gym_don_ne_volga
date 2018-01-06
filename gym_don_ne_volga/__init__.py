from gym.envs.registration import register

register(
    id='don-v0',
    entry_point='gym_don_ne_volga.envs:DonEnv',
)
