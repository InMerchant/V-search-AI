import shutil

from flask import Flask, request
from download import download_from_bucket
from uploadVideo import upload_to_bucket
from uploadCsv import upload_to_DB
from ai_caption import run_ai_code1
from ai_translate import run_ai_code2
from ai_video import run_ai_code3
from ai_videoInfo import run_ai_code4

import os

app = Flask(__name__)

#동영상 불러오기 ~ 캡션 작업
@app.route('/aiload', methods=['POST'])
def execute_files():
    if request.method == 'POST':

        # 요청에서 받은 메시지 추출
        data = request.json
        folder_name = data.get('folder_name')  # 예시: {'folder_name': 'example_folder'}
        # 폴더 생성
        folder_path = os.path.join(os.getcwd(), folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # download_from_bucket 함수 실행
        if not download_from_bucket(folder_name):
            return "Error occurred while downloading the file", 500

        # AI 프로세스 작동 함수 실행
        if not run_ai_code1(folder_path):
            return "Error occurred while running tests", 500

        # 모든 함수가 성공적으로 실행되었을 경우 응답 반환
        return "All executed successfully", 200

#ai코드 비디오 생성 작업
@app.route('/generate', methods=['POST'])
def execute_files3():
    if request.method == 'POST':

        # 요청에서 받은 메시지 추출
        data = request.json
        folder_name = data.get('folder_name')  # 예시: {'folder_name': 'example_folder'}
        # 폴더 경로 저장
        folder_path = os.path.join(os.getcwd(), folder_name)

        # AI 프로세스 작동 함수 실행
        if not run_ai_code3(folder_path):
            return "Error occurred while running tests", 500

        # 모든 함수가 성공적으로 실행되었을 경우 응답 반환
        return "All executed successfully", 200

#비디오 생성 이후 남은 ai 작업 ~ 요약 영상 업로드까지
@app.route('/savevideo', methods=['POST'])
def execute_files4():
    if request.method == 'POST':

        # 요청에서 받은 메시지 추출
        data = request.json
        folder_name = data.get('folder_name')  # 예시: {'folder_name': 'example_folder'}
        # 폴더 경로 저장
        folder_path = os.path.join(os.getcwd(), folder_name)

        # AI 프로세스 작동 함수 실행
        if not run_ai_code4(folder_path):
            return "Error occurred while running tests", 500

        # upload_to_bucket 함수 실행
        if not upload_to_bucket(folder_name):
            return "Error occurred while uploading the file", 500

        # 모든 함수가 성공적으로 실행되었을 경우 응답 반환
        return "All executed successfully", 200

#번역 코드
@app.route('/translate', methods=['POST'])
def execute_files2():
    if request.method == 'POST':

        # 요청에서 받은 메시지 추출
        data = request.json
        folder_name = data.get('folder_name')  # 예시: {'folder_name': 'example_folder'}
        # 폴더 경로 저장
        folder_path = os.path.join(os.getcwd(), folder_name)

        # AI 프로세스 작동 함수 실행
        if not run_ai_code2(folder_path):
            return "Error occurred while running tests", 500

        # 모든 함수가 성공적으로 실행되었을 경우 응답 반환
        return "All executed successfully", 200

#csv파일 DB 저장
@app.route('/saveinfo', methods=['POST'])
def excute_info_files():
    if request.method == 'POST':

        # 요청에서 받은 메시지 추출
        data = request.json
        folder_name = data.get('folder_name')
        video_no = data.get('video_no')
        print(folder_name, video_no)

        # CSV파일 DB 저장
        if not upload_to_DB(folder_name, video_no):
            return "Error occurred while uploading the file", 500

        # 현재 스크립트 파일의 경로를 기준으로 폴더 삭제
        script_dir = os.path.dirname(__file__)
        folder_path = os.path.join(script_dir, folder_name)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            print(f"Folder {folder_name} deleted successfully")
        else:
            print(f"Folder {folder_name} does not exist")

        # 모든 함수가 성공적으로 실행되었을 경우 응답 반환
        return "All executed successfully", 200

if __name__ == '__main__':
    app.run()
