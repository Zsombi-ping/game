# ROCK, PAPER, SCRISS

import nashpy as nash
import numpy as np
A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
rps = nash.Game(A)
print(rps)
B = - A
rps = nash.Game(A, B)
print(rps)

print("\n\n")
sigma_r = [0, 0, 1]
sigma_c = [0, 1, 0]
print(rps[sigma_r, sigma_c])

print("\n\n")

sigma_c = [1 / 2, 1 / 2, 0]
print(rps[sigma_r, sigma_c])

print("\n\n")

sigma_r = [0, 1 / 2, 1 / 2]
print(rps[sigma_r, sigma_c])
print("\n\n")


eqs = rps.support_enumeration()
print(list(eqs))
