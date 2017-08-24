
"""
Created on Thu Aug 17 21:15:06 2017

@author: d29905p
"""

import pandas as pd
import numpy as np

test_1 = np.array([0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,1,1,1,1,0,0,0,0,0,1,1,0,0,0,0])
#with window 5, stay requirement 3, ...should be entry on index 15
outt_1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
pre_window = 5
post_window = 3
mavg = pd.rolling_mean(pd.Series(test_1),pre_window)
entry_possible = mavg.shift(1)
#should have entry 15 and 3rd to last
entry_pre_check = np.where((entry_possible==0) & (mavg>0),1,0)
# to check if airline remains at least one quarter...convert 1s to zeros, use moving average again
presence_reverse = 1-test_1
mavg_reverse = pd.rolling_mean(pd.Series(presence_reverse),post_window)
out = np.where((mavg_reverse.shift(-(post_window-1))==0) & (entry_pre_check==1),1,0)