#!/usr/bin/env python3
"""Upload local checkpoints to OSS."""

import os
import sys
from pathlib import Path

import boto3
from tqdm import tqdm

BUCKET = "shengshu-base2-test"
ENDPOINT = "https://oss-cn-wulanchabu-internal.aliyuncs.com"
OSS_PREFIX = "ali-checkpoint/mayuanbo/pi05_xingchen_new_blocks"

LOCAL_CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints" / "pi05_xingchen" / "new_blocks"


def get_s3_client():
    ak = os.environ["AWS_ACCESS_KEY_ID"]
    sk = os.environ["AWS_SECRET_ACCESS_KEY"]
    from botocore.config import Config
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        config=Config(
            s3={"addressing_style": "virtual"},
            request_checksum_calculation="when_required",
        ),
    )


def upload_dir(s3, local_dir: Path, prefix: str):
    files = sorted(f for f in local_dir.rglob("*") if f.is_file())
    for f in tqdm(files, desc=f"Uploading {local_dir.name}"):
        key = f"{prefix}/{f.relative_to(local_dir)}"
        s3.upload_file(str(f), BUCKET, key)


def main():
    if not LOCAL_CKPT_DIR.exists():
        print(f"Checkpoint dir not found: {LOCAL_CKPT_DIR}")
        sys.exit(1)

    s3 = get_s3_client()

    subdirs = sorted(
        [d for d in LOCAL_CKPT_DIR.iterdir() if d.is_dir()],
        key=lambda p: int(p.name) if p.name.isdigit() else 0,
    )

    if not subdirs:
        print("No checkpoint subdirectories found.")
        sys.exit(1)

    print(f"Found {len(subdirs)} checkpoints to upload.")
    print(f"Target: oss://{BUCKET}/{OSS_PREFIX}/")

    for d in subdirs:
        oss_key_prefix = f"{OSS_PREFIX}/{d.name}"
        print(f"\n--- {d.name} ---")
        upload_dir(s3, d, oss_key_prefix)

    print(f"\nDone. All checkpoints uploaded to oss://{BUCKET}/{OSS_PREFIX}/")


if __name__ == "__main__":
    main()
