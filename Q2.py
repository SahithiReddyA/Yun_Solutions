import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import fclusterdata

# Create a sample dataframe with x and y coordinates
data = {
    "x": np.random.uniform(0, 10, 20),  # 20 random x-coordinates
    "y": np.random.uniform(0, 10, 20),  # 20 random y-coordinates
}
df = pd.DataFrame(data)

# Group points within 0.5 Euclidean distance
distance_threshold = 0.5
df['group'] = fclusterdata(df[['x', 'y']].values, t=distance_threshold, criterion='distance')

# Output the grouped dataframe
print(df)
