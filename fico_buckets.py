
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

def create_fico_buckets(fico_scores, num_buckets):
    """
    Creates buckets for FICO scores by minimizing the Mean Squared Error (MSE) within each bucket.

    Parameters:
    - fico_scores: Array of FICO scores.
    - num_buckets: Number of buckets to create.

    Returns:
    - bucket_boundaries: The boundaries of each bucket.
    """
    fico_scores_sorted = np.sort(fico_scores)
    bucket_size = len(fico_scores_sorted) // num_buckets

    # Initialize buckets
    buckets = []
    start = 0

    for i in range(num_buckets):
        if i == num_buckets - 1:  # Last bucket takes any remaining elements
            end = len(fico_scores_sorted)
        else:
            end = start + bucket_size
        buckets.append(fico_scores_sorted[start:end])
        start = end

    # Calculate bucket boundaries
    bucket_boundaries = [bucket[0] for bucket in buckets]
    bucket_boundaries.append(fico_scores_sorted[-1])

    # Calculate the mean squared error within each bucket
    mse_per_bucket = []
    for bucket in buckets:
        bucket_mean = np.mean(bucket)
        mse = mean_squared_error(bucket, [bucket_mean] * len(bucket))
        mse_per_bucket.append(mse)

    return bucket_boundaries, mse_per_bucket

# Load the dataset for the FICO score task
file_path_fico = 'Task 3 and 4_Loan_Data.csv'
fico_data = pd.read_csv(file_path_fico)

# Extract FICO scores from the dataset
fico_scores = fico_data['fico_score'].values

# Create 5 buckets for FICO scores
num_buckets = 5
bucket_boundaries, mse_per_bucket = create_fico_buckets(fico_scores, num_buckets)

print("Bucket Boundaries:", bucket_boundaries)
print("MSE per Bucket:", mse_per_bucket)
