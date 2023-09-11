import os

USE_AWS = bool(int(os.environ.get("USE_AWS", "0")))

import boto3

if USE_AWS:
  AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET")
  if AWS_S3_BUCKET is None:
    USE_AWS = False
else:
  AWS_S3_BUCKET = None
s3_client = boto3.client('s3') if USE_AWS else None