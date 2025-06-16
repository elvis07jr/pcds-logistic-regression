import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 500

age = np.random.randint(18, 85, size=n_samples)
capillary_refill_time = np.clip(np.random.normal(loc=2 + 0.02*age, scale=0.5), 1, 5)
oxygen_saturation = np.clip(np.random.normal(loc=98 - 0.1*age, scale=2), 85, 100)
heart_rate = np.clip(np.random.normal(loc=75 + 0.2*capillary_refill_time, scale=10), 50, 150)

risk_score = 0.03*capillary_refill_time + 0.04*(100 - oxygen_saturation) + 0.01*heart_rate + 0.02*(age > 60)
prob = 1 / (1 + np.exp(-risk_score))
has_pcds = np.random.binomial(1, prob)

df = pd.DataFrame({
    "capillary_refill_time": capillary_refill_time.round(2),
    "oxygen_saturation": oxygen_saturation.round(1),
    "heart_rate": heart_rate.astype(int),
    "age": age,
    "has_pcds": has_pcds
})

df.to_csv("PCDS_Diagnosis.csv", index=False)
