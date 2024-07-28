# V-search-AI
 이것만보고 프로젝트 AI서버
---
# 실행방법
+ 프로젝트 위치로 이동 후 분할압축된 [epoch_4_evalAcc_64.zip] 압축해제
> 폴더 안이 아닌 여기에 풀기로 파일 바로 생성하기
+ 아나콘다 설치
> 파이참보다 우선 설치 권장
+ 파이참 설치
+ Anaconda Prompt 실행
> conda create -n search python=3.10.14
> conda activate search
> cd [프로젝트 경로(app.py가 있는 곳으로)]
> pip install -r requirements.txt

+ ffmpeg 설치
> <https://ffmpeg.org/> 사이트로 이동
>> [Windows builds from gyan.dev] 이동
>> [ffmpeg-git-full.7z] 클릭하여 다운로드 진행
> 7zip 프로그램을 이용해 압축 해제 후 C:\ffmpeg 경로로 ffmpeg폴더 생성 후 압축 해제한 파일 이동
> 시스템 환경 변수 편집
>> 시스템 변수 - Path - 새로만들기 - C:\ffmpeg\bin 등록 -> cmd 창에서 [ffmpeg -version] 으로 확인

+ jdk 17 설치
> <https://www.oracle.com/java/technologies/downloads/#jdk17-windows> 사이트로 이동
>> JDK 17 - Windwos - [x64 Installer] 다운로드
> 시스템 환경 변수 편집
>> 시스템 변수 - 새로만들기 - (변수 이름=JAVA_HOME, 변수 값=C:\Program Files\Java\jdk-17 (설치경로로 변경))
>> 시스템 변수 - Path - 새로만들기 - %JAVA_HOME%\bin -> 이후 맨 위로 이동

+ 오라클 클라이언트 19c버전 설치
> <https://www.oracle.com/database/technologies/oracle-database-software-downloads.html> 사이트로 이동
>> 밑으로 스크롤 한 후 - Oracle Database Enterprise Edition - Oracle Database 19c for Microsoft Windows x64 (64-bit) - Realated Resources 라인에 [Individual Component Downloads] 로 이동 -  WINDOWS.X64_193000_client_home.zip 파일 다운로드
> 다운로드한 압축파일을 C:\ 경로에 oracle 폴더 생성 후 oracle 폴더 안에 압축 풀기
>> (예시 : C:\oracle\WINDOWS.X64_193000_db_home)
> WINDOWS.X64_193000_client_home 폴더 안에 있는 setup.exe파일을 통해 설치 진행
>> Windows 내장 계정 사용(L)
>> C:\oracle 폴더를 Oracle Base로 만들어서 진행 후 계속 설치
> <https://www.oracle.com/kr/database/technologies/instant-client/winx64-64-downloads.html> 사이트로 이동
>> Version 19.23.~~~ 클릭 후 Basic Package 다운로드
>> 폴더 이름\ 안에 풀기로 푼 후 C:\oracle 경로 안에 폴더 옮기기
>>> (예시 : C:\oracle\instantclient-basic-windows.x64-19.23.0.0.0dbru)
>>> [필수] C:\oracle\instantclient-basic-windows.x64-19.23.0.0.0dbru\instantclient_19_23 경로가 존재해야됨
+ 이후 재시작
