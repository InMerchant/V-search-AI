import os
import csv
from deep_translator import GoogleTranslator

def run_ai_code2(folder_path):

    # 경로 설정
    csv_path = os.path.join(folder_path, 'viddata.csv')
    csv_path2 = os.path.join(folder_path, 'viddata2.csv')

    # 4 번역
    translate_captions_google(csv_path, csv_path)
    translate_captions_google2(csv_path2, csv_path2)

    return True


#4번역

def translate_captions_google(input_file, output_file):
    translator = GoogleTranslator(source='en', target='ko')

    with open(input_file, mode='r', encoding='utf-8', newline='') as csvfile:
        reader = csv.reader(csvfile)
        translated_lines = []

        # 첫 번째 행(헤더) 읽기
        header = next(reader)
        translated_lines.append(header)  # 헤더를 결과 리스트에 추가

        total_lines = sum(1 for row in reader)  # 총 행 수를 얻기 위해 파일을 읽습니다.
        # 파일 포인터를 다시 시작 위치로 이동
        csvfile.seek(0)
        next(reader)  # 헤더 건너뛰기

        print(f"Total rows to translate: {total_lines + 1}")  # 헤더 포함하여 출력

        for index, row in enumerate(reader, start=1):
            original_text = row[2]  # 3열의 영어 캡션
            translated_text = translator.translate(original_text)  # 한글로 번역
            row[2] = translated_text  # 번역된 텍스트를 3열에 다시 저장
            translated_lines.append(row)  # 번역된 행 저장

            print(f"Translated {index}/{total_lines}: [{row[0]}] {translated_text}")

    with open(output_file, mode='w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(translated_lines)  # 새로운 CSV 파일에 쓰기
        print(f"Translation completed! {total_lines + 1} rows have been translated and saved to '{output_file}'")

def translate_captions_google2(input_file, output_file):
    translator = GoogleTranslator(source='en', target='ko')

    with open(input_file, mode='r', encoding='utf-8', newline='') as csvfile:
        reader = csv.reader(csvfile)
        translated_lines = []

        # 첫 번째 행(헤더) 읽기
        header = next(reader)
        translated_lines.append(header)  # 헤더를 결과 리스트에 추가

        print("Translating rows...")

        for index, row in enumerate(reader, start=1):
            original_text = row[1]  # 2열의 영어 캡션
            translated_text = translator.translate(original_text)  # 한글로 번역
            row[1] = translated_text  # 번역된 텍스트를 2열에 다시 저장
            translated_lines.append(row)  # 번역된 행 저장

            print(f"Translated {index}: [{row[0]}] {translated_text}")

    with open(output_file, mode='w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(translated_lines)  # 새로운 CSV 파일에 쓰기
        print(f"Translation completed! Rows have been translated and saved to '{output_file}'")
