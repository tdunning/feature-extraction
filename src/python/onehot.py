import numpy as np
import pandas as pd

from sklearn import preprocessing

enc = preprocessing.OrdinalEncoder()

X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)
enc.transform([['female', 'from US', 'uses Safari']])
