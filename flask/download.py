import oci
import os
import tempfile

def download_from_bucket(folder_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 디렉토리 기준
    config_file_path = os.path.join(base_dir, 'key', 'config')
    # 설정 파일 경로와 프로파일
    original_config_file_path = config_file_path
    profile_name = 'DEFAULT'
    region = 'ap-chuncheon-1'
    bucket_name = 'bucket-20240503-1000'
    object_name = f"{folder_name}/{folder_name}.mp4"
    download_file_path = os.path.join(folder_name, 'process.mp4')

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

        # 객체 다운로드
        try:
            get_obj = client.get_object(namespace, bucket_name, object_name)
            with open(download_file_path, 'wb') as f:
                for chunk in get_obj.data.raw.stream(1024 * 1024, decode_content=False):
                    f.write(chunk)
            print(f"File {object_name} downloaded successfully to {download_file_path}")
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
    # 예시 폴더 이름으로 함수 호출
    folder_name = 'test'
    download_from_bucket(folder_name)
