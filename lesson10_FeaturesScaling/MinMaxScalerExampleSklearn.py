from sklearn import preprocessing
import numpy as np

weights = np.array([[115.], [140.], [175.]])
scaler = preprocessing.MinMaxScaler()
rescaled_weights = scaler.fit_transform(weights)
print(rescaled_weights)