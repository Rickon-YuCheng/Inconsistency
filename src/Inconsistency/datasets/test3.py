import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
DB=np.load('DistilBert.npz')
WV=np.load('Wav2Vec2.npz')

# start=300
# end=302
start=300
end=493
totSplit=j=0

for i in range(start,end):
    totSplit=DB['a'][j]+DB['b'][j]+DB['c'][j]
    t_pos, t_neg, t_neu=round(DB['a'][j]/totSplit, 2), round(DB['b'][j]/totSplit, 2), round(DB['c'][j]/totSplit, 2)
    a_pos, a_neg, a_neu=round(WV['a'][j]/totSplit, 2), round(WV['b'][j]/totSplit, 2), round(WV['c'][j]/totSplit, 2)

    # X=[[t_pos, t_neg, t_neu]]
    # Y=[[a_pos, a_neg, a_neu]]
    # print(f"COSINE SIMILARITY: {cosine_similarity(X,Y)}")
    print(f"=== patient{i} -> totSplit: {totSplit:>5.1f}   T:A <=> {t_pos:<4.2f} : {t_neg:<4.2f} : {t_neu:<4.2f} = {a_pos:<4.2f} : {a_neg:<4.2f} : {a_neu:<4.2f}")
    j+=1
    