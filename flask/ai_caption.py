import os
import cv2
import csv
import glob
import torch
from tqdm import tqdm
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


def run_ai_code1(folder_path):
    # 경로 설정
    input_video_path = os.path.join(folder_path, 'process.mp4')
    output_video_path = os.path.join(folder_path, 'output_video_downfps.mp4')
    output_folder = os.path.join(folder_path, 'output_frames')
    csv_path = os.path.join(folder_path, 'viddata.csv')

    # 1 다운프레임
    downsample_to_10fps(input_video_path, output_video_path)

    # 2 이미지 추출
    save_video_frames(output_video_path, output_folder, csv_path)

    # 3 이미지 캡션 생성
    generate_image_captions(output_folder, csv_path)

    return True



#1.다운프레임
def downsample_to_10fps(input_file, output_file):


    # 비디오를 읽어 들입니다.
    cap = cv2.VideoCapture(input_file)

    if not cap.isOpened():
        raise ValueError("Error opening video file")

    # 원본 비디오의 속성을 가져옵니다.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 새 비디오 FPS를 설정합니다. 여기서는 10FPS를 사용합니다.
    new_fps = 1
    # 프레임을 선택하기 위한 카운터와 비율을 초기화합니다.
    frame_counter = 0
    ratio = original_fps / new_fps

    # 비디오 코덱과 출력 파일 설정
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, new_fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ratio에 따라 프레임을 선택합니다.
        if int(frame_counter % ratio) == 0:
            out.write(frame)
            progress = (frame_counter / total_frames) * 100
            print(f"Progress: {progress:.2f}%", end='\\r')

        frame_counter += 1

    # 완료 메시지 출력
    print("Downsampling complete.")

    # 사용한 자원을 해제합니다.
    cap.release()
    out.release()




# 2 이미지 추출
def format_time_for_csv(frame_number, fps):
    # 프레임 수와 초당 프레임 수(FPS)를 기반으로 시간대를 계산하고, 00:00:00 형식으로 형식화합니다.
    total_seconds = frame_number / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def format_time_for_filename(frame_number, fps):
    # 프레임 수와 초당 프레임 수(FPS)를 기반으로 시간대를 계산하고 형식화합니다.
    total_seconds = frame_number / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)  # 여기서 정수형으로 변환
    return f"{hours:02d}h{minutes:02d}m{seconds:02d}s"

def save_video_frames(video_path, output_folder, csv_path):
    # 비디오 파일을 불러옵니다.
    cap = cv2.VideoCapture(video_path)


    # 비디오의 전체 프레임 수와 초당 프레임 수(FPS)를 가져옵니다.
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 출력 디렉터리가 존재하는지 확인하고, 없다면 생성합니다.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        # 폴더가 이미 존재하면, 해당 폴더 내의 모든 .png 파일을 삭제합니다.
        files = glob.glob(os.path.join(output_folder, '*.png'))
        for f in files:
            os.remove(f)

    # CSV 파일을 열고, 헤더를 작성합니다.
    csv_filename = os.path.join(output_folder, csv_path)
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['frame_number', 'timeline']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 프레임을 하나씩 읽어 이미지로 저장하고, CSV 파일에 기록합니다.
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 이미지 파일명에 포함될 시간대 문자열을 생성합니다.
            time_label_for_filename = format_time_for_filename(frame_count, fps)
            time_label_for_csv = format_time_for_csv(frame_count, fps)

            # 이미지 파일명을 지정하고 저장합니다.
            filename = f"frame_{frame_count:04d}_{time_label_for_filename}.png"
            cv2.imwrite(os.path.join(output_folder, filename), frame)

            # CSV 파일에 프레임 번호와 시간을 기록합니다.
            writer.writerow({'frame_number': f"{frame_count:04d}", 'timeline': time_label_for_csv})

            # 진행 상황을 표시합니다.
            progress = (frame_count + 1) / total_frames * 100
            print(f"Saving frame {frame_count + 1}/{total_frames} ({progress:.2f}%) with timestamp {time_label_for_csv}")

            frame_count += 1

    # 비디오 파일을 닫습니다.
    cap.release()
    print("All frames have been saved successfully.")




#3이미지 캡션생성
def generate_image_captions(image_dir, viddata_csv_path):
    """
    지정된 디렉토리의 이미지들에 대해 캡션을 생성하고 이를 CSV 파일에 저장합니다.

    :param image_dir: 이미지가 있는 디렉토리 경로
    :param viddata_csv_path: CSV 파일을 저장할 디렉토리 경로
    """
    # GPU 또는 CPU 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Blip 모델과 프로세서 불러오기
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

    # viddata.csv 파일에서 프레임 번호와 시간 정보를 읽어옵니다.
    frame_data = []
    with open(viddata_csv_path, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # 헤더 스킵
        for row in csv_reader:
            frame_data.append(row)  # 각 행의 데이터(프레임 번호와 시간)를 저장

    # 새로운 CSV 파일 경로 설정 및 헤더 추가
    new_csv_path = viddata_csv_path
    with open(new_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['frame_number', 'timeline', 'caption'])

        # 이미지 목록 가져오기 및 진행 상황 로딩 바 설정
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        total_images = len(image_files)

        with tqdm(total=total_images, desc="이미지 처리 중", ncols=50) as pbar:
            for i, filename in enumerate(image_files):
                img_path = os.path.join(image_dir, filename)

                # 이미지 열기 및 캡션 생성
                raw_image = Image.open(img_path).convert('RGB')
                inputs = processor(raw_image, return_tensors="pt").to(device)  # 입력 텐서를 GPU로 이동
                out = model.generate(**inputs)
                caption = processor.decode(out[0], skip_special_tokens=True)

                # 프레임 데이터와 캡션을 결합하여 새로운 CSV 파일에 저장
                frame_number, time_info = frame_data[i]  # i번째 이미지에 해당하는 프레임 번호와 시간 정보
                csv_writer.writerow([frame_number, time_info, caption])

                # 진행 상황 업데이트
                pbar.update(1)

    # 작업 완료 메시지 표시
    print("이미지 캡션 생성 및 CSV 파일 저장 완료.")