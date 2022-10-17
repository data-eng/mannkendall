import numpy as np
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('slopes').master('local[*]').getOrCreate()
sc = spark.sparkContext
#sc.setLogLevel('DEBUG')

obs = np.genfromtxt('/home/gmouchakis/tmp/numpy-array')

(cols, rows) = obs.shape

(cols,rows) = obs.shape
if cols != 2:
    raise Exception( "There must be two columns in obs" )

# Besides the median, we will also need the confidence limits.
# Here we calculate the indexes for the lcl and the ucl

# The num of slopes is Sum rows-1, rows-2, ... 1 = rows*(rows-1)/2
l = rows*(rows-1)/2
if l % 2 == 1:
    slope_idx_1 = int( (l-1)//2 )
    slope_idx_2 = int( (l-1)//2 )
    # these m_1, m_2 defaults will be overriden below,
    # unless k_var is very low
    m_1 = (l-1)//2 - 1
    m_2 = (l-1)//2 + 1
else:
    slope_idx_1 = int( l//2-1 )
    slope_idx_2 = int( l//2 )
    # these m_1, m_2 defaults will be overriden below,
    # unless k_var is very low
    m_1 = l//2 - 2
    m_2 = l//2 + 1

tmp_array_l = 400000000
tmp_array_last_index = tmp_array_l - 1

tmp_array = np.empty(tmp_array_l)

rdds = []

tmp_array_c = 0

# compute the slopes
for i in range(0, rows-1):
    for j in range(i + 1, rows):

        val = (obs[1, j] - obs[1, i]) / (obs[0, j] - obs[0, i])

        tmp_array[tmp_array_c] = val

        if tmp_array_c == tmp_array_last_index:
            rdds.append(sc.parallelize(tmp_array))
            tmp_array_c = 0
        else:
            tmp_array_c += 1

if tmp_array_c != 0:  # write remaining values
    tmp_array = tmp_array[:tmp_array_c]
    rdds.append(sc.parallelize(tmp_array))

big_rdd = sc.union(rdds)

sorted_rdd = big_rdd.sortBy(lambda x: x).zipWithIndex()

print(sorted_rdd.first())

if l % 2 == 1:
    median_pos = l // 2
    is_even = False
else:
    median_pos = (l - 1) // 2
    median_pos_2 = median_pos + 1
    is_even = True

m_1_pos = int(m_1)
m_2_pos = int(m_2)

positions = [m_1_pos, m_2_pos, median_pos]
if is_even:
    positions.append(median_pos_2)

percentiles = sorted_rdd.filter(lambda x: x[1] in positions).map(lambda x: (x[1], x[0])).collect()

# transform results in dict {index: float_value}
#percentiles_dict = dict(map(lambda x: (x[1], float(x[0])), percentiles))
percentiles_dict = dict(percentiles)

lcl = percentiles_dict.get(m_1_pos)
slope = percentiles_dict.get(median_pos)
if is_even:
    slope = (slope + percentiles_dict.get(median_pos_2)) / 2
ucl = percentiles_dict.get(m_2_pos)

print(lcl, slope, ucl)

spark.stop()
