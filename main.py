import numpy as np

# Step 1: Input data
criteria = ['BLEU', 'Perplexity', 'Speed', 'Size', 'Ease']
models = ['GPT-2', 'T5', 'GPT-Neo']
data = np.array([
    [30, 20, 500, 1.5, 4],
    [35, 18, 300, 3.0, 3],
    [28, 22, 450, 2.7, 3]
])
weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # Sum to 1
benefit = [True, False, True, False, True]  # True = Benefit, False = Cost

# Step 2: Normalize
norm_data = data / np.sqrt(np.sum(data**2, axis=0))

# Step 3: Apply weights
weighted = norm_data * weights

# Step 4: Ideal solutions
ideal = []
negative_ideal = []
for i, is_benefit in enumerate(benefit):
    if is_benefit:
        ideal.append(np.max(weighted[:, i]))
        negative_ideal.append(np.min(weighted[:, i]))
    else:
        ideal.append(np.min(weighted[:, i]))
        negative_ideal.append(np.max(weighted[:, i]))

# Step 5: Distances
S_plus = np.sqrt(np.sum((weighted - ideal)**2, axis=1))
S_minus = np.sqrt(np.sum((weighted - negative_ideal)**2, axis=1))

# Step 6: Closeness
C = S_minus / (S_plus + S_minus)

# Step 7: Rank
rank = np.argsort(-C)  # Descending order
for r in rank:
    print(f"{models[r]}: {C[r]:.2f}")
