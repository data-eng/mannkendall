import sys

import numpy as np
import pandas as pd
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

pd_df = pd.DataFrame(obs, columns=['time', 'obs'])

#spark = SparkSession.builder.appName('slopes').master('local[*]').getOrCreate()
spark = SparkSession.builder.appName('slopes').master('spark://spark-master:7077').config('spark.executor.memory', '124g').getOrCreate()

sc = spark.sparkContext

start_time = time.time()

spark_df = spark.createDataFrame(pd_df)

spark_df.createOrReplaceTempView("arr")

res = spark.sql('select percentile(slope, array(0.25, 0.5, 0.75)) from ('
                'select ((b.obs - a.obs) / (b.time - a.time)) as slope from arr as a, arr as b where a.time < b.time'
                ') as tab(slope)')

res.explain()
res.show()

print("--- %s seconds ---" % (time.time() - start_time))

spark.stop()
