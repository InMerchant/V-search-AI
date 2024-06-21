import os
import csv
import pandas as pd
from collections import Counter
from konlpy.tag import Okt
from moviepy.editor import VideoFileClip, concatenate_videoclips
import whisper
import torch

#영어 불용어 정제 관련
from nltk.corpus import stopwords
import nltk

def run_ai_code3(folder_path):

    # 경로 설정
    input_video_path = os.path.join(folder_path, 'process.mp4')
    csv_path = os.path.join(folder_path, 'viddata.csv')
    time_csv_path = os.path.join(folder_path, 'time.csv')
    transcript_csv_path = os.path.join(folder_path, 'transcript.csv')
    viddata_csv_path = os.path.join(folder_path, 'viddata.csv')

    nltk.download('stopwords')

    # 5 불용어 정제
    clean_csv(csv_path, csv_path)

    # 6 주요 타임라인 찾기
    find_major_timeline(csv_path, time_csv_path)

    # 7 요약편집
    #final_video경로는 함수안에
    process_video(time_csv_path, input_video_path, folder_path)

    # 8 음성전사
    extract_and_transcribe(input_video_path, transcript_csv_path, folder_path)
    merge_csv_files(viddata_csv_path, transcript_csv_path, viddata_csv_path)

    return True



# 5 불용어 정제
def remove_stopwords(text, stop_words):
    """
    주어진 텍스트에서 불용어를 제거하는 함수

    :param text: 원본 텍스트
    :param stop_words: 불용어 목록
    :return: 불용어가 제거된 텍스트
    """
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def clean_csv(input_csv, output_csv):
    """
    CSV 파일을 정제하고 새로운 CSV 파일로 저장하는 함수

    :param input_csv: 입력 CSV 파일 경로
    :param output_csv: 출력 CSV 파일 경로
    """
    # CSV 파일 읽기
    df = pd.read_csv(input_csv, encoding='utf-8')

    # NLTK 불용어 목록 가져오기
    stop_words = set(stopwords.words('english'))

    # 사용자 정의 불용어 추가
    custom_stop_words = {'arafed', 'image', 'araffes', 'araffed', 'araffe'}
    stop_words.update(custom_stop_words)

    # caption 열에 대해 불용어 제거
    if 'caption' in df.columns:
        df['caption'] = df['caption'].apply(lambda x: remove_stopwords(x, stop_words) if isinstance(x, str) else x)

    # 변경된 데이터를 CSV 파일에 저장
    df.to_csv(output_csv, index=False, encoding='utf-8')

    print('파일 정제가 완료되었습니다.')


# 6주요 타임라인 찾기
def find_major_timeline(file_path, output_file_path):
    """
    주요 단어를 포함하는 타임라인을 찾아 CSV 파일로 저장하는 함수.

    :param file_path: 입력 CSV 파일 경로
    :param output_file_path: 출력 CSV 파일 경로
    """
    # CSV 파일 불러오기
    df = pd.read_csv(file_path)

    # 모든 캡션을 공백으로 분리하여 단어 리스트 생성
    words = " ".join(df['caption']).split()

    # 단어의 빈도 계산
    word_counts = Counter(words)

    # 가장 많이 나온 단어 찾기
    # most_common_word, _ = word_counts.most_common(1)[0]
    most_common_word = [word for word, _ in word_counts.most_common(5)]

    # Okt 형태소 분석기 인스턴스 생성
    okt = Okt()

    # 주요 단어들 중에서 타임라인을 찾음
    for word in most_common_word:

        # 형태소 분석을 통해 단어와 품사 태그 추출
        morphs = okt.pos(word, norm=True, stem=True)

        # 조사를 제외한 단어만 추출
        filtered_word = [word for word, tag in morphs if tag not in ('Josa')]

        # 해당 단어를 포함하는 캡션의 타임라인 찾기
        matching_rows = df[df['caption'].str.contains("|".join(filtered_word))]

        # 타임라인이 존재하면 결과를 CSV 파일로 저장하고 함수 종료
        if not matching_rows.empty:
            result_df = pd.DataFrame({
                'timeline': matching_rows['timeline']
            })

            # 결과를 CSV 파일로 저장
            result_df.to_csv(output_file_path, index=False)

            print(f"주요 단어 '{filtered_word}'를 포함하는 타임라인:")
            print(result_df)
            print(f"결과가 {output_file_path}에 저장되었습니다.")
            return

    # 주요 단어가 포함된 타임라인을 찾지 못한 경우
    print("주요 단어를 포함하는 타임라인을 찾지 못했습니다.")


# 7요약편집

def process_video(csv_file_path, original_video_path, folder_name):
    # CSV 파일에서 잘라낼 시간 정보 읽기
    cut_points = []
    with open(csv_file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)  # 첫 번째 행(헤더) 건너뛰기
        for row in csvreader:
            time_str = row[0]
            hours, minutes, seconds = map(int, time_str.split(':'))
            time_in_seconds = hours * 3600 + minutes * 60 + seconds
            cut_points.append(time_in_seconds)

    # 구간 결정 로직
    cut_segments = []
    for i in range(len(cut_points)):
        if i == 0 or (cut_points[i] - cut_points[i - 1] > 1):
            cut_segments.append([cut_points[i], cut_points[i]])
        else:
            cut_segments[-1][1] = cut_points[i]

        # 원본 영상 파일 로드
        original_clip = VideoFileClip(original_video_path)

    # 잘라낸 영상과 음원들을 저장할 리스트
    cut_clips = []

    # 잘라낼 영상과 음원 추출 및 리스트에 추가
    for start, end in cut_segments:
        if (end - start) > 1:
            clip = original_clip.subclip(start, end)
            cut_clips.append(clip)

    # 잘라낸 영상들을 합치기
    if cut_clips:
        final_clip = concatenate_videoclips(cut_clips)
        # 최종 영상 저장
        final_video_path = os.path.join(folder_name, "final_video.mp4")
        temp_audiofile_path = os.path.join(folder_name, "final_videoTEMP_MPY_wvf_snd.mp4")
        final_clip.write_videofile(final_video_path, codec="libx264", audio_codec="aac",
                                   temp_audiofile=temp_audiofile_path)
    else:
        print("잘라낼 영상 구간이 없거나 모두 1초 미만입니다.")

    os.remove(csv_file_path)  # CSV 파일 삭제

# 8음성전사
def extract_and_transcribe(video_file, output_csv_path, folder_name):
    video = VideoFileClip(video_file)
    segments = []

    if video.audio is None:
        print("이 비디오는 오디오 트랙이 없습니다.")
        segments = [{"Start Time": "0.00", "End Time": f"{video.duration:.2f}", "Text": "(None)"}]
        if not os.path.exists(output_csv_path):
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Start Time', 'End Time', 'Text']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(segments[0])
    else:
        # audio_path = "temp_audio.mp3"
        audio_path = os.path.join(folder_name, "temp_audio.mp3")
        video.audio.write_audiofile(audio_path)
        video.close()  # 메모리 해제
        print("오디오 추출 및 저장 완료.")

        # Whisper 모델 로드 및 오디오 전사 (GPU 사용)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("large").to(device)
        result = model.transcribe(audio_path)

        # 전사 결과가 비어 있는 경우 처리
        if not result["segments"]:
            segments = [{"Start Time": "0.00", "End Time": f"{video.duration:.2f}", "Text": "(None)"}]
        else:
            # 전사 결과를 사용하여 세그먼트 리스트 생성
            segments = [
                {"Start Time": f"{seg['start']:.2f}", "End Time": f"{seg['end']:.2f}", "Text": seg["text"]}
                for seg in result["segments"]
            ]

        # 'transcript.csv' 파일이 없으면 생성 및 결과 기록
        with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Start Time', 'End Time', 'Text']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if os.stat(output_csv_path).st_size == 0:  # 파일이 비어있으면 헤더 추가
                writer.writeheader()
            for segment in segments:
                writer.writerow(segment)

        # temp_audio.mp3 파일 삭제
        os.remove(audio_path)  # 파일 삭제
        print("임시 오디오 파일이 삭제되었습니다.")


# 두 번째 CSV의 Start Time을 00:00:00 형식으로 변환

def seconds_to_hms(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def merge_csv_files(viddata_csv_path, transcript_csv_path, output_csv_path):
    # CSV 파일 로드
    df_viddata = pd.read_csv(viddata_csv_path)
    df_transcript = pd.read_csv(transcript_csv_path)

    df_transcript['Start Time'] = df_transcript['Start Time'].apply(lambda x: seconds_to_hms(float(x)))
    # 엔드 타임 컬럼 삭제
    df_transcript = df_transcript.drop('End Time', axis=1)
    # 필요한 컬럼 변경 및 병합
    df_transcript.rename(columns={'Start Time': 'timeline', 'Text': 'script'}, inplace=True)
    df_merged = pd.merge(df_viddata, df_transcript, on='timeline', how='left').fillna('(None)')

    # 결과 저장
    df_merged.to_csv(output_csv_path, index=False)
    print('합쳐진 파일이 생성되었습니다.')
