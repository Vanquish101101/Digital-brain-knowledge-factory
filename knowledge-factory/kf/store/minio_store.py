from pathlib import Path

from minio import Minio
from minio.error import S3Error

from kf.config import Settings

BUCKET = "raw"


def get_client(settings: Settings) -> Minio:
    return Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=False,
    )


def ensure_bucket(client: Minio, bucket: str = BUCKET) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


def upload_file(client: Minio, local_path: Path, object_name: str, bucket: str = BUCKET) -> None:
    client.fput_object(bucket, object_name, str(local_path))


def file_exists(client: Minio, object_name: str, bucket: str = BUCKET) -> bool:
    try:
        client.stat_object(bucket, object_name)
        return True
    except S3Error as e:
        if e.code == "NoSuchKey":
            return False
        raise
