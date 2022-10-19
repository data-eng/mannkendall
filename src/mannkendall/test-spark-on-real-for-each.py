import sys

import numpy as np
from pyspark.sql import SparkSession
import time

obs = np.loadtxt(sys.argv[1])

obs = obs.T

(rows, cols) = obs.shape

# The num of slopes is Sum rows-1, rows-2, ... 1 = rows*(rows-1)/2
l = rows * (rows - 1) / 2
if l % 2 == 1:
    slope_idx_1 = int((l - 1) // 2)
    slope_idx_2 = int((l - 1) // 2)
    # these m_1, m_2 defaults will be overriden below,
    # unless k_var is very low
    m_1 = (l - 1) // 2 - 1
    m_2 = (l - 1) // 2 + 1
else:
    slope_idx_1 = int(l // 2 - 1)
    slope_idx_2 = int(l // 2)
    # these m_1, m_2 defaults will be overriden below,
    # unless k_var is very low
    m_1 = l // 2 - 2
    m_2 = l // 2 + 1

if l % 2 == 1:
    median_pos = l // 2
    is_even = False
else:
    median_pos = (l - 1) // 2
    median_pos_2 = median_pos + 1
    is_even = True

m_1_pos = int(m_1)
m_2_pos = int(m_2)

#spark = SparkSession.builder.appName('slopes').master('local[*]').getOrCreate()
spark = SparkSession.builder.appName('slopes').master('spark://spark-master:7077').config('spark.executor.memory', '124g').getOrCreate()

sc = spark.sparkContext

start_time = time.time()

obs_rdd = sc.parallelize(obs)

positions = [m_1_pos, m_2_pos, median_pos]
if is_even:
    positions.append(median_pos_2)

res = obs_rdd.cartesian(obs_rdd) \
    .filter(lambda x: x[0][0] < x[1][0]) \
    .map(lambda x: (x[1][1] - x[0][1]) / (x[1][0] - x[0][0])) \
    .sortBy(lambda x: x) \
    .zipWithIndex() \
    .filter(lambda x: x[1] in positions)\
    .map(lambda x: (x[1], x[0]))

print(res.toDebugString())

# transform results in dict {index: float_value}
percentiles_dict = dict(res.collect())

lcl = percentiles_dict.get(m_1_pos)
slope = percentiles_dict.get(median_pos)
if is_even:
    slope = (slope + percentiles_dict.get(median_pos_2)) / 2
ucl = percentiles_dict.get(m_2_pos)

print(f'lcl={lcl}, slope={slope}, ucl={ucl}')
print("--- %s seconds ---" % (time.time() - start_time))

spark.stop()
