import pandas as pd
from scipy.io import arff

data, meta = arff.loadarff('jm1.arff')
df = pd.DataFrame(data)

# Show all columns (attributes)
pd.set_option('display.max_columns', None)

# Display the first few rows of the DataFrame
# Insert the number of rows in ()
print(df.head(10))
