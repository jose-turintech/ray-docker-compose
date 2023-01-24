import ray
from ray.train.xgboost import XGBoostTrainer
from ray.air.config import ScalingConfig

ray.init(address='ray://localhost:10001')

# Load data.
dataset = ray.data.read_csv("s3://anonymous@air-example-data/breast_cancer.csv").repartition(4)
# Split data into train and validation.
train_dataset, valid_dataset = dataset.train_test_split(test_size=0.3)

trainer = XGBoostTrainer(
    scaling_config=ScalingConfig(
        # Number of workers to use for data parallelism.
        num_workers=2,
        # Whether to use GPU acceleration.
        use_gpu=False,
        resources_per_worker={"xgboost": 1}
    ),
    label_column="target",
    num_boost_round=20,
    params={
        # XGBoost specific params
        "objective": "binary:logistic",
        # "tree_method": "gpu_hist",  # uncomment this to use GPU for training
        "eval_metric": ["logloss", "error"],
    },
    datasets={"train": train_dataset, "valid": valid_dataset},
)
result = trainer.fit()
print(result.metrics)
