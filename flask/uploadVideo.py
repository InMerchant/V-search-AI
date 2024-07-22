import oci
import os
import tempfile

def upload_to_bucket(folder_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 디렉토리 기준
    config_file_path = os.path.join(base_dir, 'key', 'config')
    # 설정 파일 경로와 프로파일
    original_config_file_path = config_file_path
    profile_name = 'DEFAULT'
    region = 'ap-chuncheon-1'
    bucket_name = 'bucket-20240503-1000'
    object_name = f"{folder_name}_smr/{folder_name}_smr.mp4"  # 업로드할 객체의 이름
    upload_file_path = os.path.join(folder_name, 'final_video.mp4')  # 업로드할 로컬 파일 경로

    # 원래 설정 파일 읽기
    with open(original_config_file_path, 'r') as file:
        config_lines = file.readlines()

    # pem 파일 경로 생성 및 설정 파일 업데이트
    config_dir = os.path.dirname(original_config_file_path)
    pem_file_path = os.path.join(config_dir, 'oci_api_key.pem')

    # pem 파일 존재 여부 확인
    if not os.path.exists(pem_file_path):
        raise Exception(f"PEM 파일을 찾을 수 없음: {pem_file_path}")

    # 임시 파일에 설정 파일 작성
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_config_file:
        for line in config_lines:
            if line.startswith('key_file='):
                temp_config_file.write(f"key_file={pem_file_path}\n")
            else:
                temp_config_file.write(line)
        temp_config_file_path = temp_config_file.name

    try:
        # OCI 설정 파일을 로드하고 인증 제공자 생성
        config = oci.config.from_file(temp_config_file_path, profile_name)
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
    finally:
        # 임시 파일 삭제
        os.remove(temp_config_file_path)

if __name__ == "__main__":
    folder_name = 'test'  # 폴더 이름 설정
    upload_to_bucket(folder_name)
