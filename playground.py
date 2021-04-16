import sys
import numpy as np
sys.path.append('./src/')

co_dataset = np.loadtxt('./data/gas_sensor/ethylene_CO.csv', dtype=np.float16, delimiter=',', skiprows=1)

print(co_dataset[:6])
