
import streamlit as st
import pandas as pd
import joblib
import sqlite3

from agent import analyze_safety

# ==========================================
# 1. 페이지 설정
# ==========================================

st.set_page_config(
    page_title="SafeBot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 SafeBot")
st.subheader("AI 기반 로봇 안전 위험도 분석")


# ==========================================
# 2. SQLite 데이터베이스 설정
# ==========================================

DB_PATH = "safebot.db"


def init_db():
    """데이터베이스 및 분석 기록 테이블 생성"""

    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_time TEXT NOT NULL DEFAULT (
                datetime('now', 'localtime')
            ),
            robot_speed REAL NOT NULL,
            human_distance REAL NOT NULL,
            obstacle_detected INTEGER NOT NULL,
            workspace_congestion INTEGER NOT NULL,
            emergency_stop INTEGER NOT NULL,
            risk_level TEXT NOT NULL,
            ai_advice TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_analysis(input_data, risk_level):
    """분석 결과 데이터베이스 저장"""

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.execute("""
        INSERT INTO analysis_history (
            robot_speed,
            human_distance,
            obstacle_detected,
            workspace_congestion,
            emergency_stop,
            risk_level
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        input_data["robot_speed"],
        input_data["human_distance"],
        input_data["obstacle_detected"],
        input_data["workspace_congestion"],
        input_data["emergency_stop"],
        risk_level
    ))

    record_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return record_id


def update_advice(record_id, advice):
    """AI 분석 결과 업데이트"""

    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        UPDATE analysis_history
        SET ai_advice = ?
        WHERE id = ?
    """, (advice, record_id))

    conn.commit()
    conn.close()


def load_history():
    """전체 분석 기록 불러오기"""

    conn = sqlite3.connect(DB_PATH)

    history_df = pd.read_sql_query("""
        SELECT *
        FROM analysis_history
        ORDER BY id DESC
    """, conn)

    conn.close()

    return history_df


# 데이터베이스 초기화
init_db()


# ==========================================
# 3. 머신러닝 모델 불러오기
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("models/risk_model.pkl")


model = load_model()


# ==========================================
# 4. 작업 환경 입력
# ==========================================

st.header("🦾 작업 환경 입력")

col1, col2 = st.columns(2)

with col1:
    speed = st.slider(
        "로봇 이동 속도 (m/s)",
        0.1, 2.0, 1.0
    )

    distance = st.slider(
        "작업자와의 거리 (m)",
        0.2, 5.0, 2.0
    )

    obstacle = st.selectbox(
        "장애물 감지",
        [0, 1],
        format_func=lambda x: "감지 안 됨" if x == 0 else "감지됨"
    )

with col2:
    congestion = st.slider(
        "작업 공간 혼잡도",
        1, 5, 3
    )

    emergency = st.selectbox(
        "비상 정지 기능",
        [1, 0],
        format_func=lambda x: "활성화" if x == 1 else "비활성화"
    )


# ==========================================
# 5. 위험도 분석 실행
# ==========================================

if st.button("🔍 위험도 분석하기", type="primary"):

    input_data = pd.DataFrame([{
        "robot_speed": speed,
        "human_distance": distance,
        "obstacle_detected": obstacle,
        "workspace_congestion": congestion,
        "emergency_stop": emergency
    }])

    try:
        # 머신러닝 예측
        result = model.predict(input_data)[0]

        # SQLite에 분석 기록 저장
        record_id = save_analysis(
            input_data.iloc[0].to_dict(),
            result
        )

        st.session_state["last_record_id"] = record_id
        st.session_state["last_risk_result"] = result

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

                # AI 분석 결과 데이터베이스 업데이트
                update_advice(record_id, advice)

            except Exception as e:

                error_message = f"AI 분석 실패: {e}"

                st.error(error_message)

                update_advice(
                    record_id,
                    error_message
                )

        st.success("분석 기록이 데이터베이스에 저장되었습니다.")

    except Exception as e:
        st.error(f"분석 실행 오류: {e}")


# ==========================================
# 6. 위험도 통계 대시보드
# ==========================================

st.divider()

st.header("📈 위험도 분석 통계")

history_df = load_history()

if not history_df.empty:

    # 위험 등급별 건수
    risk_counts = (
        history_df["risk_level"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
    )

    total = len(history_df)

    # 전체 분석 건수
    st.metric("총 누적 분석 건수", f"{total}건")

    # 위험 등급별 통계
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "낮은 위험도",
            f"{risk_counts['Low']}건"
        )

    with col2:
        st.metric(
            "중간 위험도",
            f"{risk_counts['Medium']}건"
        )

    with col3:
        st.metric(
            "높은 위험도",
            f"{risk_counts['High']}건"
        )

    # 위험도 차트
    st.subheader("위험 등급별 분석 현황")

    st.bar_chart(risk_counts)

    # 위험도 비율
    st.subheader("위험 등급별 비율")

    risk_percent = (
        risk_counts / total * 100
    ).round(1)

    st.dataframe(
        pd.DataFrame({
            "분석 건수": risk_counts,
            "비율(%)": risk_percent
        }),
        width="stretch"
    )

    # ======================================
    # 7. 최근 분석 이력
    # ======================================

    st.subheader("📋 최근 분석 이력")

    display_df = history_df.head(10).copy()

    display_df = display_df.rename(columns={
        "id": "분석 ID",
        "analysis_time": "분석 시간",
        "robot_speed": "로봇 속도",
        "human_distance": "작업자 거리",
        "obstacle_detected": "장애물 감지",
        "workspace_congestion": "공간 혼잡도",
        "emergency_stop": "비상 정지",
        "risk_level": "위험 등급",
        "ai_advice": "AI 분석 결과"
    })

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True
    )

    # ======================================
    # 8. 분석 기록 CSV 다운로드
    # ======================================

    st.subheader("📥 분석 기록 다운로드")

    csv = history_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="전체 분석 이력 CSV 다운로드",
        data=csv,
        file_name="safebot_analysis_history.csv",
        mime="text/csv"
    )

else:

    st.info(
        "아직 분석 기록이 없습니다. "
        "위험도 분석을 실행해주세요."
    )


# ==========================================
# 9. 데이터베이스 정보
# ==========================================

st.divider()

st.subheader("🗄️ 데이터베이스 정보")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "저장된 데이터",
        f"{len(history_df)}건"
    )

with col2:
    st.caption(
        "저장 위치: safebot.db"
    )


# ==========================================
# 10. 하단 정보
# ==========================================

st.divider()

st.caption(
    "SafeBot | S급 로봇 연구소 | 교육용 프로토타입"
)
