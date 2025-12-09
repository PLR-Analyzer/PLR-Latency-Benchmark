"""
Script to fit the coefficients for Eq 18 and Eq 19 from Pamplona et al. (2009).
Based on the data from Moon and Spencer (1944).
"""

import matplotlib.pyplot as plt
import numpy as np

lum = [-5, -3, -1, 1, 3, 5]

C_b = [6.6, 6.5, 5.0, 3.5, 2.2, 2.0]

C_m = [7.7, 7.4, 6.5, 4.5, 2.8, 2.0]

C_t = [8.8, 8.7, 8.0, 6.5, 3.0, 2.1]


C_tD = np.polyfit(C_m, C_t, 5)
C_bD = np.polyfit(C_m, C_b, 5)
print("C_tD coefficients:", C_tD)
print("C_bD coefficients:", C_bD)

plt.plot(lum, C_b, label="C_b")
plt.plot(lum, C_m, label="C_m")
plt.plot(lum, C_t, label="C_t")
plt.plot(lum, C_tD, label="C_tD")
plt.plot(lum, C_bD, label="C_bD")
plt.xlabel("Log Luminosity")
plt.ylabel("Pupil Diameter (mm)")

plt.legend()
plt.show()
