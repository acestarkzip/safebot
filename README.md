# safebot
SafeBot — 로봇 작업 안전 위험도 사전 점검 AI


# SafeBot 🤖
### AI 기반 로봇 안전 위험도 분석 시스템

## 📌 프로젝트 소개
SafeBot은 머신러닝과 AI Agent를 활용하여 로봇 작업 환경의 위험 요소를 분석하고 안전 개선 방안을 제안하는 프로젝트입니다.

## 🎯 개발 목적
- 로봇 작업 환경의 잠재적 위험 요소 분석
- 머신러닝을 활용한 위험도 분류
- AI Agent를 통한 안전 개선 방안 제안
- AI 기술을 활용한 로봇 안전 관리 가능성 탐색

## ✨ 주요 기능
- 작업 환경 및 로봇 상태 입력
- 머신러닝 기반 위험도 분석
- 위험 수준 시각화
- AI 기반 안전 개선 방안 제안

## 🛠️ 기술 스택

| 구분 | 기술 |
|---|---|
| 언어 | Python |
| 머신러닝 | Scikit-learn |
| 딥러닝 | PyTorch |
| 데이터 분석 | Pandas, NumPy |
| AI Agent | OpenAI API |
| 웹 인터페이스 | Streamlit |
| 개발 환경 | GitHub Codespaces |

## 🏗️ 프로젝트 구조

```text
SafeBot/
├── data/              # 데이터 저장
├── models/            # 학습 모델 저장
├── app.py             # 메인 프로그램
├── generate_data.py   # 데이터 생성
├── train_model.py    # 모델 학습
├── agent.py           # AI Agent 연동
├── README.md
├── requirements.txt
└── .env.example
```

## 🚀 실행 방법

```bash
pip install -r requirements.txt

python generate_data.py
python train_model.py

streamlit run app.py
```

## 🔮 향후 발전 방향
- NVIDIA Isaac Sim 기반 로봇 시뮬레이션 연동
- ROS 2 기반 로봇 데이터 활용
- 컴퓨터 비전을 통한 사람 및 장애물 감지
- 실시간 로봇 안전 모니터링 시스템 개발

## ⚠️ 참고 사항
본 프로젝트는 가상 데이터를 활용한 교육 및 실험용 프로토타입이며, 실제 산업 현장의 안전 인증이나 위험성 평가를 대체하지 않습니다.

<!-- 

generate_data.py — 가상 안전 데이터 생성
train_model.py — 머신러닝 모델 학습
models/ — 학습된 모델 저장
agent.py — AI 안전 분석
app.py — Streamlit 대시보드와 분석 이력 관리
safebot.db — 분석 기록 저장
README.md, requirements.txt — 프로젝트 설명 및 설치 환경

>