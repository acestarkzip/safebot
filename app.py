
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

# 분석 이력 초기화
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

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

    # 분석 이력 저장
    st.session_state.analysis_history.append({
        "robot_speed": speed,
        "human_distance": distance,
        "risk_level": result
    })

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

# 위험도 통계 대시보드
st.divider()
st.header("📈 위험도 분석 통계")

history = st.session_state.analysis_history

if history:

    history_df = pd.DataFrame(history)

    # 위험 등급별 건수
    risk_counts = (
        history_df["risk_level"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
    )

    total = len(history_df)

    # 전체 분석 건수
    st.metric("총 분석 건수", f"{total}건")

    # 위험 등급별 통계
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("낮은 위험도", f"{risk_counts['Low']}건")

    with col2:
        st.metric("중간 위험도", f"{risk_counts['Medium']}건")

    with col3:
        st.metric("높은 위험도", f"{risk_counts['High']}건")

    # 위험도 차트
    st.subheader("위험 등급별 분석 현황")

    st.bar_chart(risk_counts)

    # 위험도 비율
    st.subheader("위험 등급별 비율")

    risk_percent = (risk_counts / total * 100).round(1)

    st.dataframe(
        pd.DataFrame({
            "분석 건수": risk_counts,
            "비율(%)": risk_percent
        }),
        use_container_width=True
    )

    # 분석 이력
    st.subheader("최근 분석 이력")

    st.dataframe(
        history_df.tail(10).iloc[::-1],
        use_container_width=True
    )
    
    # CSV 다운로드
    st.subheader("📥 분석 기록 다운로드")

    csv = history_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="분석 이력 CSV 다운로드",
        data=csv,
        file_name="safebot_analysis_history.csv",
        mime="text/csv"
    )

else:
    st.info("아직 분석 기록이 없습니다. 위험도 분석을 실행해주세요.")

st.divider()

st.caption(
    "SafeBot | S급 로봇 연구소 | 교육용 프로토타입"
)
