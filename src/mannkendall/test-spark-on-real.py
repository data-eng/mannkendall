import numpy as np
from pyspark.sql import SparkSession
import time

obs = np.loadtxt('../../datasets/obs_daily')

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
spark = SparkSession.builder.appName('slopes').master('spark://spark-master:7077').getOrCreate()

spark.conf.set("spark.executor.memory", '124g')

sc = spark.sparkContext

start_time = time.time()

obs_rdd = sc.parallelize(obs)

res = obs_rdd.cartesian(obs_rdd) \
    .filter(lambda x: x[0][0] < x[1][0]) \
    .map(lambda x: (x[1][1] - x[0][1]) / (x[1][0] - x[0][0])) \
    .sortBy(lambda x: x) \
    .zipWithIndex() \
    .map(lambda x: (x[1], x[0])) \
    .sortByKey() \
    .cache()

lcl = res.lookup(m_1_pos).pop()
slope = res.lookup(median_pos).pop()
if is_even:
    slope = (slope + res.lookup(median_pos_2).pop()) / 2
ucl = res.lookup(m_2_pos)

print(f'lcl={lcl}, slope={slope}, ucl={ucl}')
print("--- %s seconds ---" % (time.time() - start_time))

spark.stop()
