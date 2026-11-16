
import streamlit as st
import pandas as pd
import joblib
import sqlite3

from datetime import date

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


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    """기존 데이터베이스 유지 및 테이블 확인"""

    with get_connection() as conn:

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


def save_analysis(input_data, risk_level):
    """분석 결과 저장"""

    with get_connection() as conn:

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

        return cursor.lastrowid


def update_advice(record_id, advice):
    """AI 분석 결과 업데이트"""

    with get_connection() as conn:

        conn.execute("""
            UPDATE analysis_history
            SET ai_advice = ?
            WHERE id = ?
        """, (advice, record_id))


def load_history():
    """전체 분석 기록 조회"""

    with get_connection() as conn:

        return pd.read_sql_query("""
            SELECT *
            FROM analysis_history
            ORDER BY id DESC
        """, conn)


def delete_analysis(record_id):
    """선택한 분석 기록 삭제"""

    with get_connection() as conn:

        conn.execute("""
            DELETE FROM analysis_history
            WHERE id = ?
        """, (record_id,))


def delete_all_history():
    """전체 분석 기록 삭제"""

    with get_connection() as conn:

        conn.execute("""
            DELETE FROM analysis_history
        """)


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

st.divider()

st.header("🦾 작업 환경 입력")

col1, col2 = st.columns(2)

with col1:

    speed = st.slider(
        "로봇 이동 속도 (m/s)",
        0.1, 2.0, 1.0,
        key="robot_speed"
    )

    distance = st.slider(
        "작업자와의 거리 (m)",
        0.2, 5.0, 2.0,
        key="human_distance"
    )

    obstacle = st.selectbox(
        "장애물 감지",
        [0, 1],
        format_func=lambda x: (
            "감지 안 됨" if x == 0 else "감지됨"
        ),
        key="obstacle_detected"
    )


with col2:

    congestion = st.slider(
        "작업 공간 혼잡도",
        1, 5, 3,
        key="workspace_congestion"
    )

    emergency = st.selectbox(
        "비상 정지 기능",
        [1, 0],
        format_func=lambda x: (
            "활성화" if x == 1 else "비활성화"
        ),
        key="emergency_stop"
    )


# ==========================================
# 5. 위험도 분석 실행
# ==========================================

if st.button(
    "🔍 위험도 분석하기",
    type="primary"
):

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

        # SQLite 저장
        record_id = save_analysis(
            input_data.iloc[0].to_dict(),
            result
        )

        st.session_state["last_record_id"] = record_id

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

                update_advice(
                    record_id,
                    advice
                )

            except Exception as e:

                error_message = f"AI 분석 실패: {e}"

                st.error(error_message)

                update_advice(
                    record_id,
                    error_message
                )

        st.success(
            f"분석 ID {record_id}의 기록이 저장되었습니다."
        )

    except Exception as e:

        st.error(f"분석 실행 오류: {e}")


# ==========================================
# 6. 위험도 통계 대시보드
# ==========================================

st.divider()

st.header("📈 위험도 분석 통계")

history_df = load_history()

if not history_df.empty:

    risk_counts = (
        history_df["risk_level"]
        .value_counts()
        .reindex(
            ["Low", "Medium", "High"],
            fill_value=0
        )
    )

    total = len(history_df)

    # 전체 분석 건수
    st.metric(
        "총 누적 분석 건수",
        f"{total}건"
    )

    # 위험 등급별 통계
    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "🟢 낮은 위험도",
            f"{risk_counts['Low']}건"
        )

    with col2:

        st.metric(
            "🟡 중간 위험도",
            f"{risk_counts['Medium']}건"
        )

    with col3:

        st.metric(
            "🔴 높은 위험도",
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

else:

    st.info("아직 분석 기록이 없습니다.")


# ==========================================
# 7. 분석 이력 검색 및 필터
# ==========================================

st.divider()

st.header("🔎 분석 이력 관리")

history_df = load_history()

if not history_df.empty:

    # 날짜 변환
    history_df["analysis_time"] = pd.to_datetime(
        history_df["analysis_time"],
        errors="coerce"
    )

    min_date = history_df["analysis_time"].min().date()
    max_date = history_df["analysis_time"].max().date()

    # 검색 영역
    st.subheader("검색 조건")

    col1, col2 = st.columns(2)

    with col1:

        search_id = st.text_input(
            "분석 ID 검색",
            placeholder="예: 5",
            key="search_id"
        )

    with col2:

        selected_risk = st.selectbox(
            "위험 등급 필터",
            ["전체", "Low", "Medium", "High"],
            key="selected_risk"
        )

    # 날짜 필터
    st.write("분석 기간")

    date_range = st.date_input(
        "조회 기간 선택",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="date_range"
    )

    # 필터 적용
    filtered_df = history_df.copy()

    # ID 검색
    if search_id.strip():

        if not search_id.strip().isdigit():

            st.warning("분석 ID는 숫자로 입력해주세요.")

            filtered_df = filtered_df.iloc[0:0]

        else:

            filtered_df = filtered_df[
                filtered_df["id"].astype(str).str.contains(
                    search_id.strip()
                )
            ]

    # 위험 등급 검색
    if selected_risk != "전체":

        filtered_df = filtered_df[
            filtered_df["risk_level"] == selected_risk
        ]

    # 날짜 검색
    if isinstance(date_range, (tuple, list)):

        if len(date_range) == 2:

            start_date, end_date = date_range

            filtered_df = filtered_df[
                (
                    filtered_df["analysis_time"].dt.date
                    >= start_date
                )
                &
                (
                    filtered_df["analysis_time"].dt.date
                    <= end_date
                )
            ]

    # 검색 결과
    st.subheader("검색 결과")

    st.metric(
        "검색된 분석 건수",
        f"{len(filtered_df)}건"
    )

    # 검색 결과 표
    if not filtered_df.empty:

        display_df = filtered_df[[
            "id",
            "analysis_time",
            "robot_speed",
            "human_distance",
            "obstacle_detected",
            "workspace_congestion",
            "emergency_stop",
            "risk_level"
        ]].copy()

        display_df = display_df.rename(columns={

            "id": "분석 ID",
            "analysis_time": "분석 시간",
            "robot_speed": "로봇 속도",
            "human_distance": "작업자 거리",
            "obstacle_detected": "장애물 감지",
            "workspace_congestion": "공간 혼잡도",
            "emergency_stop": "비상 정지",
            "risk_level": "위험 등급"

        })

        display_df["분석 시간"] = (
            display_df["분석 시간"].dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True
        )

    else:

        st.warning("검색 조건에 해당하는 기록이 없습니다.")


    # ======================================
    # 8. 분석 상세 보기
    # ======================================

    st.divider()

    st.subheader("📋 분석 상세 보기")

    if not filtered_df.empty:

        available_ids = filtered_df["id"].astype(int).tolist()

        selected_id = st.selectbox(
            "상세 내용을 확인할 분석 ID",
            available_ids,
            format_func=lambda x: f"분석 ID #{x}",
            key="detail_record_id"
        )

        selected_record = filtered_df[
            filtered_df["id"] == selected_id
        ].iloc[0]

        with st.expander(
            f"📄 분석 ID #{selected_id} 상세 정보",
            expanded=True
        ):

            st.write(
                f"**분석 시간:** {selected_record['analysis_time']}"
            )

            st.write(
                f"**위험 등급:** {selected_record['risk_level']}"
            )

            st.divider()

            st.write("**작업 환경 정보**")

            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:

                st.metric(
                    "로봇 속도",
                    f"{selected_record['robot_speed']} m/s"
                )

                st.metric(
                    "작업자 거리",
                    f"{selected_record['human_distance']} m"
                )

                st.metric(
                    "장애물 감지",
                    "감지됨" if selected_record[
                        "obstacle_detected"
                    ] else "감지 안 됨"
                )

            with detail_col2:

                st.metric(
                    "공간 혼잡도",
                    f"{selected_record['workspace_congestion']}/5"
                )

                st.metric(
                    "비상 정지",
                    "활성화" if selected_record[
                        "emergency_stop"
                    ] else "비활성화"
                )

            st.divider()

            st.write("**🤖 AI 안전 분석 결과**")

            advice = selected_record["ai_advice"]

            if advice:

                st.markdown(advice)

            else:

                st.warning("저장된 AI 분석 결과가 없습니다.")

    else:

        st.info("상세 내용을 확인할 기록이 없습니다.")


    # ======================================
    # 9. 분석 기록 삭제
    # ======================================

    st.divider()

    st.subheader("🗑️ 분석 기록 관리")

    if not filtered_df.empty:

        delete_id = st.selectbox(
            "삭제할 분석 ID",
            filtered_df["id"].astype(int).tolist(),
            format_func=lambda x: f"분석 ID #{x}",
            key="delete_record_id"
        )

        confirm_delete = st.checkbox(
            f"분석 ID #{delete_id} 삭제에 동의합니다.",
            key="confirm_delete"
        )

        if st.button(
            "선택한 분석 기록 삭제",
            type="secondary",
            disabled=not confirm_delete
        ):

            delete_analysis(delete_id)

            st.session_state["confirm_delete"] = False

            st.success(
                f"분석 ID #{delete_id} 기록을 삭제했습니다."
            )

            st.rerun()

    else:

        st.info("삭제할 분석 기록이 없습니다.")


    # ======================================
    # 10. CSV 다운로드
    # ======================================

    st.divider()

    st.subheader("📥 분석 기록 다운로드")

    # 검색된 결과 다운로드
    csv = filtered_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="🔍 검색 결과 CSV 다운로드",
        data=csv,
        file_name="safebot_filtered_history.csv",
        mime="text/csv"
    )

    # 전체 기록 다운로드
    all_csv = load_history().to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="📁 전체 분석 이력 CSV 다운로드",
        data=all_csv,
        file_name="safebot_analysis_history.csv",
        mime="text/csv"
    )

else:

    st.info("관리할 분석 기록이 없습니다.")


# ==========================================
# 11. 데이터베이스 정보
# ==========================================

st.divider()

st.subheader("🗄️ 데이터베이스 정보")

current_history = load_history()

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "저장된 데이터",
        f"{len(current_history)}건"
    )

with col2:

    st.caption(
        f"저장 위치: {DB_PATH}"
    )


# ==========================================
# 12. 하단 정보
# ==========================================

st.divider()

st.caption(
    "SafeBot | S급 로봇 연구소 | 교육용 프로토타입"
)
