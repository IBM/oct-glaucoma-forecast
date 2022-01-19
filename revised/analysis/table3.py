import pandas as pd

df_estimation = pd.read_csv('results_pooled/strucfunc_estimation.csv')
df_forecast = pd.read_csv('results_pooled/strucfunc_forecast.csv')

def make_column(df):
    res = []
    for i in ['1', '2', '3', '4', '5']:
        m = df[i].mean()
        std = df[i].std()
        str = f"{m:.2f}({std:.2f})"
        res.append(str)
    return res

res_estimation = make_column(df_estimation)
res_forecast = make_column(df_forecast)

for i in range(5):
    str = res_estimation[i] + "\t" + res_forecast[i]
    print(str)

