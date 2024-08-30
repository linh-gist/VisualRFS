import time
import cpputils

pyfuzzy = cpputils.FuzzyLite('./compute_pd.fis')
start = time.time()
miss, ioi = 0.5, 0.6
for i in range(1000):
    output = pyfuzzy.compute(miss, ioi)
print('Execution duration for 1000 times', time.time() - start)
print('Input', miss, ioi, 'Output', output)
