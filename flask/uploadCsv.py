import csv

import cx_Oracle
import os
import logging

# Oracle 클라우드 설정
cx_Oracle.init_oracle_client(lib_dir=r"C:\oracle\instantclient-basic-windows.x64-19.23.0.0.0dbru\instantclient_19_23")

connection = cx_Oracle.connect(user='ADMIN', password='Baesunmoon2024', dsn='vsearch_high')

#업로드 실행 함수
def upload_to_DB(folder_name, video_no):

    infoCSV_upload(folder_name, video_no)
    SMY_OBJ_DATA_upload(folder_name, video_no)






#csv1업로드
def infoCSV_upload(folder_name, video_no):
    # 로거 설정
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # CSV 파일 읽기
        logger.info("CSV 파일 읽는 중...")
        csv_file_path = os.path.join(folder_name, 'viddata.csv')
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            lines = [row for row in reader if row and row['frame_number']]  # frame_number가 있는 줄만 처리

        logger.info("CSV 파일 DB에 삽입 중...")
        # Oracle 데이터베이스에 데이터 삽입
        cursor = connection.cursor()
        query = """
            INSERT INTO video_info (frame_number, timeline, caption, script, scriptimo, nude, video_no)
            VALUES (:1, TO_TIMESTAMP(:2, 'HH24:MI:SS'), :3, :4, :5, :6, :7)
        """
        for line_num, row in enumerate(lines, start=2):  # 헤더를 감안하여 줄 번호 2부터 시작
            try:
                # 데이터 삽입
                cursor.execute(query, [
                    row['frame_number'],
                    row['timeline'],
                    row['caption'],
                    row['script'],
                    row['scriptimo'],
                    row['nude'],
                    int(video_no)
                ])
            except Exception as e:
                # 예외 처리
                logger.error(f"데이터베이스 삽입 오류 - 줄 번호: {line_num}, 오류: {e}")
                return False

        connection.commit()  # 커밋
        cursor.close()  # 커서 닫기

        logger.info("데이터베이스에 데이터 삽입이 완료되었습니다.")
        return True
    except cx_Oracle.Error as error:
        logger.error(f"데이터베이스 오류 발생: {error}")
        return False
    except Exception as e:
        logger.error(f"예외 발생: {e}")
        return False

#csv2 업로드
def SMY_OBJ_DATA_upload(folder_name, video_no):
    # 로거 설정
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # CSV 파일 읽기
        logger.info("CSV 파일 읽는 중...")
        csv_file_path = os.path.join(folder_name, 'viddata2.csv')

        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            lines = [row for row in reader]  # 모든 줄을 읽기

        # Oracle 데이터베이스에 데이터 업데이트
        cursor = connection.cursor()
        update_query = """
            UPDATE video
            SET smyscript = :1, video_obj = :2
            WHERE video_no = :3
        """

        # 각 줄 처리
        for line_num, row in enumerate(lines, start=2):  # 헤더를 감안하여 줄 번호 2부터 시작
            try:
                smyscript = row['scriptsum'].strip() if row['scriptsum'] else None  # 'scriptsum'이 비어 있으면 NULL
                words = row['word'].split(', ') if row['word'] else []  # 'word' 열을 쉼표로 구분하여 분리

                if smyscript is None:  # smyscript가 None인 경우, 빈 문자열로 처리
                    smyscript = ''

                video_obj = ', '.join(words)  # 모든 word를 쉼표로 구분하여 하나의 문자열로 만듦
                logger.info(f"smyscript: {smyscript}, video_obj: {video_obj}")
                # 데이터 업데이트
                cursor.execute(update_query, [smyscript, video_obj, int(video_no)])

            except Exception as e:
                # 예외 처리
                logger.error(f"데이터베이스 업데이트 오류 발생 - 줄 번호: {line_num}, 오류: {e}")
                return False

        connection.commit()  # 커밋
        cursor.close()  # 커서 닫기
        connection.close()  # 연결 종료

        logger.info("데이터베이스에 데이터 업데이트가 완료되었습니다.")
        return True
    except cx_Oracle.Error as error:
        logger.error(f"데이터베이스 오류 발생: {error}")
        return False
    except Exception as e:
        logger.error(f"예외 발생: {e}")
        return False

if __name__ == "__main__":
    folder_name = "test"
    video_no = "500"
    result = upload_to_DB(folder_name, video_no)
    print(result)
