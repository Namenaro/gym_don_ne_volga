import numpy as np

state_vector_low = np.full(5, fill_value=-0.1)
action_vector = np.array([-100., 100.])
state_vector_low = np.concatenate((state_vector_low, action_vector))

print (state_vector_low)