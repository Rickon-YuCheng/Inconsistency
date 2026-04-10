"""percent calculate"""

import numpy as np

DB = np.load("DistilBert.npz")
WV = np.load("Wav2Vec2.npz")
HN = np.load("HowNet.npz")
PL = np.load("PseudoLabel.npz")

# START=300
# END=302
START = 300
END = 493

print(f"patient{DB['patientIdx']}")
print(f"patient{WV['patientIdx']}")
print(f"patient{HN['patientIdx']}")
print(f"patient{PL['patientIdx']}")

assert np.array_equal(len(PL["patientIdx"]), len(PL["label"])), " sth error "

# 缺失342,394,398,460
breakpoint()