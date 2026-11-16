
import os
from dotenv import load_dotenv
from openai import OpenAI

# 환경변수 불러오기
load_dotenv()

# API 클라이언트 생성
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def analyze_safety(risk_level, robot_data):

    prompt = f"""
    당신은 로봇 안전 분석 전문가입니다.

    다음 로봇 작업 환경을 분석해 주세요.

    위험 등급: {risk_level}
    작업 환경 데이터: {robot_data}

    다음 내용을 한국어로 설명해 주세요.
    1. 위험 원인
    2. 안전 개선 방안
    3. 추가 주의사항

    교육용 분석 결과임을 고려해 주세요.
    """

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    return response.output_text
