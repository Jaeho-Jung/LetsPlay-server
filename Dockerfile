# 베이스 이미지 설정
FROM python:3.10-buster

# 작업 디렉토리 생성 및 이동
WORKDIR /app

# 필요한 파일 복사
COPY requirements.txt .

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg

RUN pip install --upgrade pip
# 종속성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 앱 코드 복사
COPY . .

# FastAPI 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]