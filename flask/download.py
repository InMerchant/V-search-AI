import oci
import os

def download_from_bucket(folder_name):
    # 설정 파일 경로와 프로파일
    config_file_path = '~/key/config'
    profile_name = 'DEFAULT'
    region = 'ap-chuncheon-1'
    bucket_name = 'bucket-20240503-1000'
    object_name = f"{folder_name}/{folder_name}.mp4"
    download_file_path = os.path.join(folder_name, 'process.mp4')

    # OCI 설정 파일을 로드하고 인증 제공자 생성
    config = oci.config.from_file(config_file_path, profile_name)
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

if __name__ == "__main__":
    # 예시 폴더 이름으로 함수 호출
    folder_name = 'test'
    download_from_bucket(folder_name)
