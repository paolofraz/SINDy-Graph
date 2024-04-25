import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv ('./data/WHO-COVID-19-global-data.csv',parse_dates=True)
# get onlny italy data
it_data = df.loc[df['Country'] == 'Italy']
print(it_data.head())
it_data["New_cases"].plot()
plt.show()