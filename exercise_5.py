import numpy as np

P_x1 = [0.2, 0.8]
P_x2 = [[0.8, 0.2], [0.2, 0.8]]
P_x3 = [[0.8, 0.2], [0.2, 0.8]]
P_x4 = [[[0.8, 0.2], [0.4, 0.6]], [[0.4, 0.6], [0.2, 0.8]]]
P_x5 = [[0.8, 0.2], [0.2, 0.8]]

# Build Table
print('Joint Probability Distribution Table (JPDT)')
table = dict()
for a in [0, 1]:
    for b in [0, 1]:
        for c in [0, 1]:
            for d in [0, 1]:
                for e in [0, 1]:
                    table[(a, b, c, d, e)] = P_x1[a] * P_x2[a][b] * P_x3[a][c] * P_x4[b][c][d] * P_x5[c][e]

print('P(x1, x2, x3, x4, x5) = P(x1).P(x2|x1).P(x3|x1).P(x4|x2, x3).P(x5|x3)')
for k, v in table.items(): print(f'P{k} = {v:>6.2%}')

# Random Generation
print('\nData Generation:')
n = 100

ks = [k for k in table.keys()]
vs = [v for v in table.values()]

ns = np.random.choice(len(ks), n, p=vs)

for n in ns: print(*ks[n])
