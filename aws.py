import os

USE_AWS = bool(int(os.environ.get("USE_AWS", "0")))

import boto3

if USE_AWS:
  AWS_DYNAMO_TABLE = os.environ.get("AWS_DYNAMO_TABLE")
  for env_var in [AWS_DYNAMO_TABLE]:
    if env_var is None:
      USE_AWS = False
else:
  AWS_DYNAMO_TABLE = None
print(f"USE_AWS: {USE_AWS}")

dynamo_client = boto3.client('dynamodb') if USE_AWS else None
