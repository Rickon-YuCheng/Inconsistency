import numpy as np
DB=np.load('DistilBert.npz')
WV=np.load('Wav2Vec2.npz')

# START=300
# END=302
START=300
END=493
totSplit=j=0

for i in range(START,END):
    totSplit=DB['a'][j]+DB['b'][j]+DB['c'][j] # patient300 totSplit=87
    t_pos, t_neg, t_neu=round(DB['a'][j]/totSplit, 1), round(DB['b'][j]/totSplit, 1), round(DB['c'][j]/totSplit, 1)
    a_pos, a_neg, a_neu=round(WV['a'][j]/totSplit, 1), round(WV['b'][j]/totSplit, 1), round(WV['c'][j]/totSplit, 1)

    print(f"=== patient{i} -> T:A <=> {t_pos:<3.1f} : {t_neg:<3.1f} : {t_neu:<3.1f} = {a_pos:<3.1f} : {a_neg:<3.1f} : {a_neu:<3.1f}")
    j+=1

# 箱型圖
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# DB = np.load('DistilBert.npz')
# WV = np.load('Wav2Vec2.npz')

# labels = ['Positive', 'Negative', 'Neutral']
# keys = ['a', 'b', 'c']

# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# fig.suptitle('DistilBert vs Wav2Vec2', fontsize=16)

# for i, key in enumerate(keys):
#     sns.boxplot(y=DB[key], ax=axes[0, i], color='skyblue')
#     axes[0, i].set_title(f'DistilBert - {labels[i]}')
#     axes[0, i].set_ylabel('Numbers')
#     sns.boxplot(y=WV[key], ax=axes[1, i], color='salmon')
#     axes[1, i].set_title(f'Wav2Vec2 - {labels[i]}')
#     axes[1, i].set_ylabel('Numbers')

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig('test3.jpg')
