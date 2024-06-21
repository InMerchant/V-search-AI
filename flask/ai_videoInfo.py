import os
import pandas as pd
from collections import Counter
from konlpy.tag import Okt
import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from tqdm import tqdm
import time
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from nudenet import NudeDetector
import re

def run_ai_code4(folder_path):

    # 경로 설정
    output_folder = os.path.join(folder_path, 'output_frames')
    viddata_csv_path = os.path.join(folder_path, 'viddata.csv')
    output_csv_path = os.path.join(folder_path, 'viddata2.csv')

    # 9 텍스트 요약
    summarize_text(viddata_csv_path, output_csv_path)

    # 10 6순위 단어 카운트
    count_top_words(viddata_csv_path, output_csv_path)

    # 11 윤리검증
    # epoch_4_evalAcc_64.pth경로는 함수안에
    process_csv(viddata_csv_path, viddata_csv_path)

    # 12 선정성라벨
    nude_detector = NudeDetector()
    interested_labels1 = ['FEMALE_GENITALIA_COVERED', 'ANUS_COVERED', 'FEMALE_BREAST_COVERED', 'BUTTOCKS_COVERED',
                          'ARMPITS_COVERED']
    interested_labels2 = ['FEMALE_BREAST_EXPOSED', 'MALE_GENITALIA_EXPOSED', 'FEMALE_GENITALIA_EXPOSED',
                          'BUTTOCKS_EXPOSED', 'ANUS_EXPOSED']
    detect_nudity_and_update_csv(output_folder, nude_detector, interested_labels1, interested_labels2, viddata_csv_path)

    return True


# 9텍스트요약
def summarize_text(csv_file_path, output_csv_file_path):
    # BART 모델 및 토크나이저 로드
    tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
    model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')

    try:
        # CSV 파일 로드
        df = pd.read_csv(csv_file_path)

        # 'script' 열에서 '(None)' 문자열을 포함하지 않는 행만 필터링
        df_filtered = df[~df['script'].str.contains(r'\(None\)', na=False)]

        # 필터링된 'script' 열만 추출
        texts = df_filtered['script'].tolist()
        full_text = ' '.join(texts).replace('\\n', ' ')

        # 변환된 문자열을 요약
        raw_input_ids = tokenizer.encode(full_text)
        chunks = [raw_input_ids[i:i + 512] for i in range(0, len(raw_input_ids), 512)]

        summaries = []
        for chunk in chunks:
            input_ids = [tokenizer.bos_token_id] + chunk + [tokenizer.eos_token_id]
            summary_ids = model.generate(torch.tensor([input_ids]), num_beams=4, max_length=512, eos_token_id=1)
            summary = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)

            # 마침표, 물음표, 느낌표 이후의 문자열 제거
            summary = re.sub(r'[.!?].*', '', summary)

            summaries.append(summary)

        full_summary = ' '.join(summaries)

        # 요약된 값을 'scriptsum' 열로 가진 새로운 데이터프레임 생성
        summary_df = pd.DataFrame({'scriptsum': [full_summary]})

        # 새로운 데이터프레임을 output_csv_file_path 파일로 저장
        summary_df.to_csv(output_csv_file_path, index=False)

    except Exception as e:
        print(f"실패했습니다: {e}")


# 10. 6순위 단어 카운트
def count_top_words(csv_file_path, output_csv_file_path, top_n=10):
    # CSV 파일 로드
    df = pd.read_csv(csv_file_path)

    # 'script'와 'caption' 열의 텍스트를 결합
    df['combined_text'] = df['script'].fillna('') + ' ' + df['caption'].fillna('')

    # Okt 형태소 분석기 인스턴스 생성
    okt = Okt()

    # 모든 텍스트를 하나의 문자열로 결합
    combined_text = ' '.join(df['combined_text'])

    # 형태소 분석을 통해 단어와 품사 태그 추출
    words = okt.pos(combined_text, norm=True, stem=True)

    # 조사와 동사를 제외한 단어만 추출
    extracted_words = [word for word, tag in words if tag not in ('Josa', 'Verb')]

    # 제외할 단어 목록
    excluded_words = ['(', 'None', ')', '?', '있다', '.', '!', '클로즈업']  # 제외하고 싶은 단어들을 이곳에 추가하세요.

    # 제외할 단어들을 필터링
    filtered_words = [word for word in extracted_words if word not in excluded_words]

    # 단어의 빈도수 계산
    word_counts = Counter(filtered_words)

    # 가장 많이 나타난 단어 top_n개 출력
    most_common_words = word_counts.most_common(top_n)

    words_only = [word for word, count in most_common_words]

    #viddata2.csv 읽기
    vd2f = pd.read_csv(output_csv_file_path)

    #csv에 새로운 열 데이터 추가
    new_row_values = ', '.join(words_only)
    print(new_row_values)
    vd2f['word'] = new_row_values


    os.remove(output_csv_file_path)
    vd2f.to_csv(output_csv_file_path, index = False)

    print("새로운 행이 추가되었습니다.")

# 11윤리검증
def process_csv(input_csv_path, output_csv_path):
    # 모델 및 토크나이저 로드
    model_path = "epoch_4_evalAcc_64.pth"
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    config = BertConfig.from_pretrained('bert-base-multilingual-cased', num_labels=8)
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', config=config)

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드 (GPU로 설정)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 모델을 평가 모드로 전환

    def classify_sentence(sentence):
        # 입력 데이터 토큰화
        inputs = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 예측
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

        return predicted_class

    # CSV 파일 불러오기
    df = pd.read_csv(input_csv_path)

    # 열 이름 수동 지정
    df.columns = ['frame_number', 'timeline', 'caption', 'script']

    # 새로운 데이터프레임 생성 및 열 이름 지정
    new_df = pd.DataFrame(columns=['frame_number', 'timeline', 'caption', 'script', 'scriptimo'])

    start = time.time()

    # tqdm을 사용하여 진행 상황을 로딩바로 표시 (한 줄에 표시되도록 설정)
    rows = []
    for i in tqdm(range(len(df)), ncols=50, dynamic_ncols=True, leave=False):
        scriptimo = classify_sentence(df.loc[i, 'script'])
        rows.append({
            'frame_number': df.loc[i, 'frame_number'],
            'timeline': df.loc[i, 'timeline'],
            'caption': df.loc[i, 'caption'],
            'script': df.loc[i, 'script'],
            'scriptimo': scriptimo
        })

    new_df = pd.concat([new_df, pd.DataFrame(rows)], ignore_index=True)
    end = time.time()

    # 새로운 CSV 파일로 저장
    new_df.to_csv(output_csv_path, index=False, header=True)

    print(f"경과 시간: {end - start}초")


# 12선정성 라벨
def detect_nudity_and_update_csv(folder_path, nude_detector, interested_labels1, interested_labels2, viddata_csv_path):
    # CSV 파일 로드
    df = pd.read_csv(viddata_csv_path)

    # 'nude' 열이 없다면 추가
    if 'nude' not in df.columns:
        df['nude'] = 0

    # 폴더 내 모든 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 이미지 파일을 순서대로 처리
    for i, image_file in enumerate(image_files):
        file_path = os.path.join(folder_path, image_file)

        # 이미지에서 음란 내용 탐지
        detections = nude_detector.detect(file_path)

        # 초기 설정: 음란성이 없는 것으로 가정
        nude_value = 0

        # 탐지된 각 내용에 대해
        for detection in detections:
            # 관심 있는 라벨1 또는 라벨2에 속하고, 점수가 0.7 이상인 경우
            if (detection['class'] in interested_labels1 or detection['class'] in interested_labels2) and detection[
                'score'] >= 0.7:
                nude_value = 1 if detection['class'] in interested_labels1 else 2
                break  # 관심 있는 라벨에 해당하는 콘텐츠를 발견하면 루프 종료

        # 'nude' 열 업데이트
        if i < len(df):
            df.at[i, 'nude'] = nude_value
        else:
            # 이미지가 더 많은 경우, 새로운 행 추가
            df = df.append({'nude': nude_value}, ignore_index=True)

    # 변경된 데이터프레임을 CSV 파일에 저장
    df.to_csv(viddata_csv_path, index=False)
    print("완료")