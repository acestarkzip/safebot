# 🤖 SafeBot
### AI 기반 로봇 작업 환경 안전 위험도 분석 시스템

**SafeBot은 머신러닝과 AI를 활용하여 로봇 작업 환경의 잠재적 위험 요소를 분석하고, 안전 개선 방안을 제안하는 교육용 프로토타입입니다.**

---

## 📌 프로젝트 소개

로봇과 사람이 같은 공간에서 작업하는 환경에서는 충돌, 장애물, 작업 공간 혼잡 등 다양한 안전 위험이 발생할 수 있습니다.

SafeBot은 이러한 작업 환경의 입력 데이터를 바탕으로 머신러닝 모델이 위험도를 분류하고, OpenAI API를 활용해 위험 원인과 안전 개선 방안을 제안하는 시스템입니다.

## 🎯 개발 목적

- 로봇 작업 환경의 잠재적 위험 요소 분석
- 머신러닝을 활용한 위험도 분류
- AI를 통한 안전 개선 방안 제안
- 데이터 기반 로봇 안전 관리 가능성 탐색

## ✨ 주요 기능

| 기능 | 설명 |
|---|---|
| 작업 환경 입력 | 로봇 속도, 작업자 거리, 장애물 감지 등 입력 |
| 위험도 예측 | Random Forest 기반 Low / Medium / High 분류 |
| AI 안전 분석 | OpenAI API를 활용한 위험 원인 및 개선 방안 제안 |
| 대시보드 | 위험 등급별 분석 현황 및 통계 시각화 |
| 분석 이력 관리 | SQLite를 활용한 분석 결과 저장 및 조회 |
| 데이터 다운로드 | CSV 및 PDF 분석 보고서 다운로드 |

## 🧠 머신러닝 모델

- 알고리즘: Random Forest Classifier
- 학습 데이터: 가상 로봇 작업 환경 데이터 1,000건
- 입력 변수:
  - 로봇 이동 속도
  - 작업자와의 거리
  - 장애물 감지 여부
  - 작업 공간 혼잡도
  - 비상 정지 기능 상태
- 데이터 분할: 학습 80%, 테스트 20%
- 모델 평가: Accuracy 및 Classification Report

※ 현재 학습 데이터는 실제 로봇 센서 데이터가 아닌 가상 생성 데이터이며, 사전에 정의한 교육용 규칙으로 위험 등급을 부여합니다.

## 🛠️ 기술 스택

| 구분 | 기술 |
|---|---|
| 프로그래밍 언어 | Python |
| 머신러닝 | Scikit-learn |
| 데이터 분석 | Pandas, NumPy |
| AI 분석 | OpenAI API |
| 웹 인터페이스 | Streamlit |
| 데이터베이스 | SQLite |
| PDF 보고서 | ReportLab |
| 모델 저장 | Joblib |
| 개발 환경 | GitHub Codespaces |

## 🏗️ 프로젝트 구조

```text
SafeBot/
├── data/
│   └── safety_data.csv       # 가상 안전 데이터
├── models/
│   └── risk_model.pkl        # 학습된 머신러닝 모델
├── app.py                    # Streamlit 메인 애플리케이션
├── agent.py                  # AI 안전 분석
├── generate_data.py          # 가상 데이터 생성
├── train_model.py            # 머신러닝 모델 학습
├── requirements.txt          # 라이브러리 목록
├── .env.example              # 환경변수 설정 예시
├── .gitignore                # Git 제외 파일 설정
└── README.md                 # 프로젝트 설명
```

## 🚀 실행 방법

### 1. 저장소 복제

```bash
git clone https://github.com/본인계정/SafeBot.git
cd SafeBot
```

### 2. 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 3. OpenAI API 키 설정

프로젝트 최상위 폴더에 `.env` 파일을 생성하고 API 키를 입력합니다.

```env
OPENAI_API_KEY=본인의_API_KEY
```

### 4. 가상 데이터 생성

```bash
python generate_data.py
```

### 5. 머신러닝 모델 학습

```bash
python train_model.py
```

### 6. 애플리케이션 실행

```bash
streamlit run app.py
```

## 🔮 향후 발전 방향

- NVIDIA Isaac Sim 기반 로봇 시뮬레이션 연동
- ROS 2 기반 로봇 상태 데이터 활용
- 컴퓨터 비전을 통한 작업자 및 장애물 감지
- 실제 로봇 데이터를 활용한 모델 학습
- 실시간 로봇 작업 환경 안전 모니터링

## ⚠️ 참고 사항

본 프로젝트는 가상 데이터를 활용한 교육 및 실험용 프로토타입입니다.

현재 위험도 예측 결과는 사전에 정의한 가상 데이터 생성 규칙을 학습한 모델의 분류 결과이며, 실제 산업 현장의 위험도를 검증한 결과가 아닙니다.

따라서 실제 산업 현장의 안전 인증, 전문적인 위험성 평가 및 안전 관리 시스템을 대체할 수 없습니다.

---

**SafeBot | S급 로봇 연구소 (SSS Robotics)**  
*Exploring AI-powered Safety for Robotics*
