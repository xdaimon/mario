import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from TrainingInfo import TrainingInfo

trainingInfo = TrainingInfo('data/trainingInfo.pkl')

df_sections = []
for i,f in enumerate(sorted(os.listdir('data/'))):
    try:
        name,extension = os.path.splitext(f)
    except:
        continue
    if extension != '.pkl' or 'agent_stats.' not in name:
        continue
    df_sections.append(pd.read_pickle('data/' + f,compression='gzip'))

df = pd.concat(df_sections, ignore_index=True)





# ---- Visualize stats ----
grouped = df.groupby(['episode','agent'])
returns = grouped['reward'].sum()
# Plot returns
plot_rows = []
for name,group in returns.groupby('episode'):
    plot_rows.append(np.sort(group.tolist()))
ax = plt.subplot(1,1,1)
ax.matshow(plot_rows)
ax.set_xlabel('Function Samples (Sorted)')
ax.set_ylabel('Param Updates')
ax.set_yticklabels([])
ax.set_xticklabels([])
#plot mean, min, max, std for returns
stats = returns.groupby('episode').describe().rolling(window=10).mean().dropna()
del stats['count']
stats.plot()
plt.show()





# ---- In each episode, which agents performed better than the mean ----
# grouped = df.groupby(['episode','agent'])
# returns = pd.DataFrame(grouped['reward'].sum()) # returns a MultiIndex Series with index=[episode, agent]
# # returns_in_episode = returns.groupby('episode')
# # mean_return_per_episode = returns_in_episode.mean()
# returns.reset_index()
# print(returns.index)
# # print(returns.index)
# # print(returns_in_episode.index)
# # print(mean_return_per_episode.index)
# # print(returns_in_episode > mean_return_per_episode)





# # ---- Get rewards for a specific agent during certain episode ----
# # TODO episodes are one off
# df=df[df['episode']==67]
# df.set_index(['episode','agent'],inplace=True)
# rewards = df['reward'][(362,1)].tolist()
# df.reset_index(inplace=True)





# ---- Which agent had the greatest return? ----
grouped = df.groupby(['episode','agent'])
returns = grouped['reward'].sum() # returns a MultiIndex Series with index [episode, agent]
episode,agent = returns.idxmax()
print(episode,agent)






# # ---- Which agent had the greatest x_pos? ----
# row = df.loc[df['x_pos'].idxmax()]
# episode,agent = row[['episode','agent']]






# # ---- Which agents were above the 75th percentile ----
# df.set_index(['episode','agent'],inplace=True)
# percentile = returns.describe()['75%']+200 # what was the 75th percentile?
# agents_above = df.drop(returns[returns <= percentile].index) # drop those below it
# df.reset_index(inplace=True)





# # ---- What is the number of steps taken by best agent ----
# df.set_index(['episode','agent'],inplace=True)
# print(df.loc[episode,agent]['step'].max())
# df.reset_index(inplace=True)






# # ---- Does return maximize x_pos? ----
# max_xpos = df.groupby(['episode','agent'])['x_pos'].max()
# df.set_index(['episode','agent'],inplace=True)
# bools = returns < max_xpos # FAILS IF EMPTY
# agents_above = df.drop(returns[bools].index) # drop those below it
# # print(agents_above)
# df.reset_index(inplace=True)
