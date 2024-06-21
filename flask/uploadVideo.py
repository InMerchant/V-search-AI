import oci
import os

def upload_to_bucket(folder_name):
    # 설정 파일 경로와 프로파일
    config_file_path = '~/key/config'
    profile_name = 'DEFAULT'
    region = 'ap-chuncheon-1'
    bucket_name = 'bucket-20240503-1000'
    object_name = f"{folder_name}_smr/{folder_name}_smr.mp4"  # 업로드할 객체의 이름
    upload_file_path = os.path.join(folder_name, 'final_video.mp4')  # 업로드할 로컬 파일 경로

    # OCI 설정 파일을 로드하고 인증 제공자 생성
    config = oci.config.from_file(config_file_path, profile_name)
    client = oci.object_storage.ObjectStorageClient(config)
    client.base_client.set_region(region)

    # 네임스페이스 가져오기
    namespace = client.get_namespace().data
    print(f"Using namespace: {namespace}")

    # 파일 업로드
    try:
        with open(upload_file_path, 'rb') as f:
            data = f.read()

        # MIME 타입 설정
        content_type = 'video/mp4'

        client.put_object(
            namespace,
            bucket_name,
            object_name,
            data,
            content_type=content_type
        )

        print(f"File {upload_file_path} uploaded successfully to bucket {bucket_name} as {object_name}")
        return True
    except oci.exceptions.ServiceError as e:
        print(f"Service error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    upload_to_bucket()
