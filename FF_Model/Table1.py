import pandas as pd
from scipy import stats


FACTOR = 'data/5_Factor_2x3.csv'
df = pd.read_csv(FACTOR)
df.set_index("Time", inplace=True)

# replica
# range = df[0:618].drop(columns=['RF'])
range = df[0:618]

# print(range)
# mean_MktRF = range['Mkt-RF'].mean()
# SD_MktRF = range['Mkt-RF'].std()
# t_stat_MktRF, p_MktRF = stats.ttest_1samp(range['Mkt-RF'],0)
# print(mean_MktRF, SD_MktRF, t_stat_MktRF)   # 0.5075404530744336 4.4565585601609365 2.8311656663395537

mean = range.mean()
SD = range.std()
t_stat, p = stats.ttest_1samp(range,[0,0,0,0,0,0])
print('Replicate:')
print('mean:')
print(mean)
# Mkt-RF    0.507540
# SMB       0.268026
# HML       0.370405
# RMW       0.264693
# CMA       0.320971
# MOM       0.687184
print('SD:')
print(SD)
# Mkt-RF    4.456559
# SMB       3.052500
# HML       2.819011
# RMW       2.219373
# CMA       2.020899
# MOM       4.220473
print('t-statistic:')
print(t_stat)  # [2.83116567 2.18280664 3.26643249 2.96486959 3.94834739 4.04768218]



# update
range1 = df[0:666]
mean_new = range1.mean()
SD_new = range1.std()
t_stat_new, p_new = stats.ttest_1samp(range1,[0,0,0,0,0,0])
print('------------------------')
print('Update:')
print('mean:')
print(mean_new)
# Mkt-RF    0.512613
# SMB       0.239234
# HML       0.324565
# RMW       0.257207
# CMA       0.281772
# MOM       0.662898
print('SD:')
print(SD_new)
# Mkt-RF    4.389855
# SMB       3.021648
# HML       2.799864
# RMW       2.171005
# CMA       1.998596
# MOM       4.171569
print('t-statistic:')
print(t_stat_new)   # [3.01353522 2.04322645 2.99158421 3.05745087 3.63839223 4.10094825]