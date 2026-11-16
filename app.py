
import streamlit as st
import pandas as pd
import joblib

from agent import analyze_safety

# 페이지 설정
st.set_page_config(
    page_title="SafeBot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 SafeBot")
st.subheader("AI 기반 로봇 안전 위험도 분석")

# 머신러닝 모델 불러오기
model = joblib.load("models/risk_model.pkl")

# 작업 환경 입력
st.header("작업 환경 입력")

speed = st.slider("로봇 이동 속도", 0.1, 2.0, 1.0)
distance = st.slider("작업자와의 거리 (m)", 0.2, 5.0, 2.0)
obstacle = st.selectbox("장애물 감지", [0, 1])
congestion = st.slider("작업 공간 혼잡도", 1, 5, 3)
emergency = st.selectbox("비상 정지 기능", [1, 0])

# 분석 실행
if st.button("위험도 분석하기"):

    input_data = pd.DataFrame([{
        "robot_speed": speed,
        "human_distance": distance,
        "obstacle_detected": obstacle,
        "workspace_congestion": congestion,
        "emergency_stop": emergency
    }])

    # 머신러닝 예측
    result = model.predict(input_data)[0]

    st.header("📊 머신러닝 분석 결과")

    if result == "Low":
        st.success("낮은 위험도")
    elif result == "Medium":
        st.warning("중간 위험도")
    else:
        st.error("높은 위험도")

    st.write(f"예측 등급: **{result}**")

    # AI Agent 분석
    st.header("🤖 AI 안전 분석")

    with st.spinner("AI가 분석하고 있습니다..."):
        try:
            advice = analyze_safety(
                result,
                input_data.iloc[0].to_dict()
            )

            st.markdown(advice)

        except Exception as e:
            st.error(f"AI 분석 오류: {e}")

st.divider()

st.caption(
    "SafeBot | S급 로봇 연구소 | 교육용 프로토타입"
)
