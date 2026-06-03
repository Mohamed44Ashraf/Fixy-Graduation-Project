import numpy as np
import pandas as pd
from scipy.stats import truncnorm


# Settings
RNG_SEED = 42
np.random.seed(RNG_SEED)

N = 400_000
TECHNICIAN_COUNT = 10_000
CUSTOMER_COUNT = 70_000

TARGET_ANOMALY_RATIO = 0.43

# Helper Function To Generate Numbers with Gaussian Distribution but Truncated to a Range (1,5)
def truncated_normal(mean, sd, low, high, size):
    return truncnorm(
        (low - mean) / sd,
        (high - mean) / sd,
        loc=mean,
        scale=sd
    ).rvs(size)


# IDs
df = pd.DataFrame({
    "CustomerID": np.random.randint(1, CUSTOMER_COUNT + 1, N),
    "TechnicianID": np.random.randint(1, TECHNICIAN_COUNT + 1, N)
})

# User behavior
users = pd.DataFrame({
    "CustomerID": np.arange(1, CUSTOMER_COUNT + 1),
    "UserAvgRating": truncated_normal(3.6, 0.4, 1, 5, CUSTOMER_COUNT),
    "UserRatingStd": truncated_normal(0.45, 0.18, 0.1, 1.2, CUSTOMER_COUNT),
    "FeedbackCount": np.random.poisson(6, CUSTOMER_COUNT) + 1
})

df = df.merge(users, on="CustomerID")



# Technician behavior
tech_avg = truncated_normal(4.1, 0.35, 1, 5, TECHNICIAN_COUNT)
tech_var = (
    0.55 * (5 - tech_avg) / 4 +
    np.random.normal(0, 0.1, TECHNICIAN_COUNT)
).clip(0.05, 1.1)

techs = pd.DataFrame({
    "TechnicianID": np.arange(1, TECHNICIAN_COUNT + 1),
    "WorkerAvgRating": tech_avg,
    "WorkerRatingVar": tech_var
})

df = df.merge(techs, on="TechnicianID")



# User–Technician interaction
df["UserWorkerAvg"] = (
    0.45 * df["UserAvgRating"]
    + 0.45 * df["WorkerAvgRating"]
    - 0.3 * df["UserRatingStd"]
    + np.random.normal(0, 0.2, N)
).clip(1, 5)



# Hidden latent behavior (NOT EXPORTED)
latent_shift = np.random.binomial(
    1,
    0.45 * (df["UserRatingStd"] > 0.6) +
    0.25 * (df["WorkerRatingVar"] > 0.6),
    N
)

latent_intensity = np.random.normal(
    0,
    1.4 + df["UserRatingStd"],
    N
)

hidden_behavior = latent_shift * latent_intensity



# Rating deviation
base_dev = np.random.normal(
    0,
    df["WorkerRatingVar"] * 0.7,
    N
)

df["RatingDeviation"] = base_dev + hidden_behavior
df["RatingDeviation"] = df["RatingDeviation"].clip(-3.5, 3.5)



# predicted rating
df["predictedRating"] = (
    0.5 * df["WorkerAvgRating"]
    + 0.35 * df["UserWorkerAvg"]
    + 0.15 * df["UserAvgRating"]
    + 0.6 * df["RatingDeviation"]
    + np.random.normal(0, 0.12, N)
).clip(1, 5)



# Anomaly logic (uses hidden behavior)
anomaly_score = (
    np.abs(hidden_behavior) * 1.2 +
    np.abs(df["predictedRating"] - df["UserAvgRating"]) * 0.4 +
    np.abs(df["predictedRating"] - df["WorkerAvgRating"]) * 0.35 +
    df["UserRatingStd"] * 0.4 +
    (1 / np.sqrt(df["FeedbackCount"])) * 0.25
)

threshold = np.quantile(anomaly_score, 1 - TARGET_ANOMALY_RATIO)
df["IsAnomalous"] = (anomaly_score > threshold).astype(int)



# Missing values (UNCHANGED)
missing_map = {
    "UserAvgRating": (0.005, 0.015),
    "WorkerAvgRating": (0.003, 0.012),
    "UserRatingStd": (0.006, 0.018),
    "UserWorkerAvg": (0.008, 0.02),
    "FeedbackCount": (0.004, 0.01),
    "WorkerRatingVar": (0.005, 0.015),
    "RatingDeviation": (0.003, 0.01)
}

rng = np.random.default_rng(42)

for col, frac in missing_map.items():
    n_missing = rng.integers(
        int(frac[0] * N),
        int(frac[1] * N)
    )
    idx = rng.choice(df.index, size=n_missing, replace=False)
    df.loc[idx, col] = np.nan


# Final dataset (NO HIDDEN FEATURES)
df = df[
    [
        "CustomerID", "TechnicianID",
        "UserAvgRating", "WorkerAvgRating",
        "UserRatingStd", "UserWorkerAvg",
        "FeedbackCount", "WorkerRatingVar",
        "RatingDeviation", "predictedRating",
        "IsAnomalous"
    ]
]

print(df["IsAnomalous"].value_counts(normalize=True))
df.to_csv("Anomaly_Detection_Dataset_v7_1.csv", index=False)
