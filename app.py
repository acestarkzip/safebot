import io
import sqlite3
from datetime import date

import joblib
import pandas as pd
import streamlit as st

from agent import analyze_safety


# ==========================================
# 1. 페이지 설정
# ==========================================

st.set_page_config(
    page_title="SafeBot",
    page_icon="🤖",
    layout="wide"
)

DB_PATH = "safebot.db"
MODEL_PATH = "models/risk_model.pkl"
RISK_ORDER = ["Low", "Medium", "High"]
RISK_LABELS = {
    "Low": "낮은 위험도",
    "Medium": "중간 위험도",
    "High": "높은 위험도",
}


# ==========================================
# 2. 데이터베이스
# ==========================================

def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    """기존 데이터베이스를 유지하면서 테이블이 없을 때만 생성."""
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
    with get_connection() as conn:
        conn.execute("""
            UPDATE analysis_history
            SET ai_advice = ?
            WHERE id = ?
        """, (advice, record_id))


def load_history():
    with get_connection() as conn:
        return pd.read_sql_query("""
            SELECT *
            FROM analysis_history
            ORDER BY id DESC
        """, conn)


def delete_analysis(record_id):
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM analysis_history WHERE id = ?",
            (record_id,)
        )


def delete_all_history():
    with get_connection() as conn:
        conn.execute("DELETE FROM analysis_history")


init_db()


# ==========================================
# 3. 모델 및 보고서 함수
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def make_pdf_report(record):
    """선택한 분석 기록을 한글 PDF 보고서로 생성."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    )
    from xml.sax.saxutils import escape

    buffer = io.BytesIO()

    # ReportLab 내장 한국어 CID 글꼴
    font_name = "HYSMyeongJo-Medium"
    pdfmetrics.registerFont(UnicodeCIDFont(font_name))

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=42,
        leftMargin=42,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "KoreanTitle",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=19,
        leading=26,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#20252B"),
        spaceAfter=16
    )
    heading_style = ParagraphStyle(
        "KoreanHeading",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=12,
        leading=18,
        textColor=colors.HexColor("#263746"),
        spaceBefore=12,
        spaceAfter=7
    )
    body_style = ParagraphStyle(
        "KoreanBody",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=9,
        leading=14,
        alignment=TA_LEFT
    )

    risk_text = RISK_LABELS.get(
        str(record["risk_level"]),
        str(record["risk_level"])
    )

    story = [
        Paragraph("SafeBot 로봇 안전 분석 보고서", title_style),
        Paragraph(
            f"분석 ID: {int(record['id'])}　|　"
            f"분석 시간: {escape(str(record['analysis_time']))}",
            body_style
        ),
        Spacer(1, 12),
        Paragraph("1. 머신러닝 위험도 예측", heading_style),
        Paragraph(f"예측 결과: <b>{escape(risk_text)}</b>", body_style),
        Paragraph("2. 작업 환경 입력값", heading_style),
    ]

    rows = [
        ["항목", "분석값"],
        ["로봇 이동 속도", f"{float(record['robot_speed']):.2f} m/s"],
        ["작업자와의 거리", f"{float(record['human_distance']):.2f} m"],
        ["장애물 감지", "감지됨" if int(record["obstacle_detected"]) else "감지 안 됨"],
        ["작업 공간 혼잡도", f"{int(record['workspace_congestion'])} / 5"],
        ["비상 정지 기능", "활성화" if int(record["emergency_stop"]) else "비활성화"],
    ]

    table = Table(rows, colWidths=[150, 300], repeatRows=1)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EDF2")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17212B")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C8D0D8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.extend([table, Paragraph("3. AI 안전 분석 결과", heading_style)])

    advice = record.get("ai_advice")
    if pd.isna(advice) or not str(advice).strip():
        advice_text = "저장된 AI 분석 결과가 없습니다."
    else:
        # PDF에서는 Markdown 기호를 제거하지 않고 텍스트로 안전하게 표시
        advice_text = escape(str(advice)).replace("\n", "<br/>")

    story.append(Paragraph(advice_text, body_style))
    story.extend([
        Spacer(1, 18),
        Paragraph(
            "본 보고서는 SafeBot 교육용 프로토타입의 분석 결과입니다. "
            "실제 산업 현장의 안전 인증이나 위험성 평가를 대체하지 않습니다.",
            body_style
        )
    ])

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ==========================================
# 4. 화면 제목 및 작업 환경 입력
# ==========================================

st.title("🤖 SafeBot")
st.subheader("AI 기반 로봇 안전 위험도 분석")
st.caption("로봇 작업 환경의 위험 요소를 분석하고 안전 개선 방향을 제안합니다.")

st.divider()
st.header("🦾 작업 환경 입력")

input_col1, input_col2 = st.columns(2)

with input_col1:
    speed = st.slider(
        "로봇 이동 속도 (m/s)",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        key="robot_speed"
    )
    distance = st.slider(
        "작업자와의 거리 (m)",
        min_value=0.2,
        max_value=5.0,
        value=2.0,
        step=0.1,
        key="human_distance"
    )
    obstacle = st.selectbox(
        "장애물 감지",
        [0, 1],
        format_func=lambda x: "감지 안 됨" if x == 0 else "감지됨",
        key="obstacle_detected"
    )

with input_col2:
    congestion = st.slider(
        "작업 공간 혼잡도",
        min_value=1,
        max_value=5,
        value=3,
        key="workspace_congestion"
    )
    emergency = st.selectbox(
        "비상 정지 기능",
        [1, 0],
        format_func=lambda x: "활성화" if x == 1 else "비활성화",
        key="emergency_stop"
    )

if st.button("🔍 위험도 분석하기", type="primary"):
    input_data = pd.DataFrame([{
        "robot_speed": speed,
        "human_distance": distance,
        "obstacle_detected": obstacle,
        "workspace_congestion": congestion,
        "emergency_stop": emergency
    }])

    try:
        model = load_model()
        result = model.predict(input_data)[0]

        record_id = save_analysis(
            input_data.iloc[0].to_dict(),
            result
        )

        st.session_state["last_record_id"] = record_id
        st.session_state["last_risk_result"] = result

        st.subheader("📊 머신러닝 분석 결과")
        if result == "Low":
            st.success("낮은 위험도")
        elif result == "Medium":
            st.warning("중간 위험도")
        else:
            st.error("높은 위험도")

        st.write(f"예측 등급: **{result}**")

        st.subheader("🤖 AI 안전 분석")
        with st.spinner("AI가 분석하고 있습니다..."):
            try:
                advice = analyze_safety(
                    result,
                    input_data.iloc[0].to_dict()
                )
                st.markdown(advice)
                update_advice(record_id, advice)
            except Exception as exc:
                error_message = f"AI 분석 실패: {exc}"
                st.error(error_message)
                update_advice(record_id, error_message)

        st.success(f"분석 ID #{record_id}의 기록이 저장되었습니다.")

    except Exception as exc:
        st.error(f"분석 실행 오류: {exc}")


# ==========================================
# 5. 데이터 불러오기 및 공통 전처리
# ==========================================

history_df = load_history()

st.divider()
st.header("📈 위험도 분석 대시보드")

if history_df.empty:
    st.info("아직 분석 기록이 없습니다. 작업 환경을 입력하고 위험도 분석을 실행해주세요.")
else:
    history_df["analysis_time"] = pd.to_datetime(
        history_df["analysis_time"],
        errors="coerce"
    )
    history_df["analysis_date"] = history_df["analysis_time"].dt.date

    # --------------------------------------
    # 5-1. 핵심 지표
    # --------------------------------------

    risk_counts = (
        history_df["risk_level"]
        .value_counts()
        .reindex(RISK_ORDER, fill_value=0)
    )
    total = len(history_df)
    high_count = int(risk_counts["High"])
    high_rate = high_count / total * 100 if total else 0

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("총 누적 분석", f"{total}건")
    metric2.metric("낮은 위험도", f"{int(risk_counts['Low'])}건")
    metric3.metric("중간 위험도", f"{int(risk_counts['Medium'])}건")
    metric4.metric("높은 위험도 비율", f"{high_rate:.1f}%")

    # --------------------------------------
    # 5-2. 위험 등급 분포
    # --------------------------------------

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("위험 등급별 분석 건수")
        st.bar_chart(risk_counts.rename(index=RISK_LABELS))

    with chart_col2:
        st.subheader("위험 등급별 비율")
        risk_percent = (risk_counts / total * 100).round(1)
        percent_df = pd.DataFrame({
            "분석 건수": risk_counts,
            "비율(%)": risk_percent
        }).rename(index=RISK_LABELS)
        st.dataframe(percent_df, width="stretch")

    # --------------------------------------
    # 5-3. 날짜별 위험도 변화
    # --------------------------------------

    st.subheader("📅 날짜별 위험도 변화")

    daily_df = (
        history_df.dropna(subset=["analysis_date"])
        .groupby(["analysis_date", "risk_level"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=RISK_ORDER, fill_value=0)
        .sort_index()
    )

    if not daily_df.empty:
        daily_df.index = daily_df.index.astype(str)
        st.line_chart(daily_df.rename(columns=RISK_LABELS))
    else:
        st.info("날짜별 추이를 표시할 데이터가 없습니다.")

    # --------------------------------------
    # 5-4. 위험 요소별 기초 통계
    # --------------------------------------

    st.subheader("🧭 위험 요소별 분석")

    factor_col1, factor_col2 = st.columns(2)

    with factor_col1:
        speed_summary = (
            history_df.groupby("risk_level")["robot_speed"]
            .agg(["count", "mean"])
            .reindex(RISK_ORDER)
        )
        speed_summary.columns = ["분석 건수", "평균 속도(m/s)"]
        st.write("위험 등급별 평균 로봇 속도")
        st.dataframe(speed_summary, width="stretch")

        distance_summary = (
            history_df.groupby("risk_level")["human_distance"]
            .agg(["count", "mean"])
            .reindex(RISK_ORDER)
        )
        distance_summary.columns = ["분석 건수", "평균 작업자 거리(m)"]
        st.write("위험 등급별 평균 작업자 거리")
        st.dataframe(distance_summary, width="stretch")

    with factor_col2:
        congestion_summary = (
            history_df.groupby("workspace_congestion")["risk_level"]
            .apply(lambda values: (values == "High").mean() * 100)
            .rename("높은 위험도 비율(%)")
            .to_frame()
        )
        st.write("공간 혼잡도별 높은 위험도 비율")
        st.dataframe(congestion_summary.round(1), width="stretch")

        obstacle_summary = (
            history_df.groupby("obstacle_detected")["risk_level"]
            .apply(lambda values: (values == "High").mean() * 100)
            .rename("높은 위험도 비율(%)")
            .to_frame()
        )
        obstacle_summary.index = obstacle_summary.index.map(
            {0: "장애물 감지 안 됨", 1: "장애물 감지됨"}
        )
        st.write("장애물 감지 여부별 높은 위험도 비율")
        st.dataframe(obstacle_summary.round(1), width="stretch")

    st.caption(
        "참고: 위 통계는 저장된 분석 사례의 단순 집계입니다. "
        "표본 수가 적거나 입력 조건이 편향되어 있으면 일반적인 인과관계로 해석할 수 없습니다."
    )


# ==========================================
# 6. 검색 및 필터
# ==========================================

st.divider()
st.header("🔎 분석 이력 관리")

history_df = load_history()

if not history_df.empty:
    history_df["analysis_time"] = pd.to_datetime(
        history_df["analysis_time"],
        errors="coerce"
    )

    valid_times = history_df["analysis_time"].dropna()
    min_date = valid_times.min().date()
    max_date = valid_times.max().date()

    search_col1, search_col2 = st.columns(2)

    with search_col1:
        search_id = st.text_input(
            "분석 ID 검색",
            placeholder="예: 5",
            key="search_id"
        )

    with search_col2:
        selected_risk = st.selectbox(
            "위험 등급 필터",
            ["전체", "Low", "Medium", "High"],
            key="selected_risk"
        )

    date_range = st.date_input(
        "조회 기간 선택",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="date_range"
    )

    filtered_df = history_df.copy()

    if search_id.strip():
        if not search_id.strip().isdigit():
            st.warning("분석 ID는 숫자로 입력해주세요.")
            filtered_df = filtered_df.iloc[0:0]
        else:
            filtered_df = filtered_df[
                filtered_df["id"].astype(str).str.contains(
                    search_id.strip(),
                    regex=False
                )
            ]

    if selected_risk != "전체":
        filtered_df = filtered_df[
            filtered_df["risk_level"] == selected_risk
        ]

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            filtered_df["analysis_time"].dt.date.between(
                start_date, end_date
            )
        ]

    st.subheader("검색 결과")
    st.metric("검색된 분석 건수", f"{len(filtered_df)}건")

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
            "robot_speed": "로봇 속도(m/s)",
            "human_distance": "작업자 거리(m)",
            "obstacle_detected": "장애물 감지",
            "workspace_congestion": "공간 혼잡도",
            "emergency_stop": "비상 정지",
            "risk_level": "위험 등급"
        })

        display_df["분석 시간"] = display_df["분석 시간"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        display_df["장애물 감지"] = display_df["장애물 감지"].map(
            {0: "감지 안 됨", 1: "감지됨"}
        )
        display_df["비상 정지"] = display_df["비상 정지"].map(
            {0: "비활성화", 1: "활성화"}
        )
        st.dataframe(display_df, width="stretch", hide_index=True)
    else:
        st.warning("검색 조건에 해당하는 기록이 없습니다.")

    # --------------------------------------
    # 6-1. 상세 조회 및 PDF 보고서
    # --------------------------------------

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

        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            st.metric("위험 등급", RISK_LABELS.get(
                selected_record["risk_level"],
                selected_record["risk_level"]
            ))
            st.metric("로봇 속도", f"{selected_record['robot_speed']} m/s")
            st.metric("작업자 거리", f"{selected_record['human_distance']} m")

        with detail_col2:
            st.metric(
                "장애물 감지",
                "감지됨" if selected_record["obstacle_detected"] else "감지 안 됨"
            )
            st.metric(
                "공간 혼잡도",
                f"{selected_record['workspace_congestion']} / 5"
            )
            st.metric(
                "비상 정지",
                "활성화" if selected_record["emergency_stop"] else "비활성화"
            )

        st.markdown("**🤖 AI 안전 분석 결과**")
        advice = selected_record.get("ai_advice")
        if pd.notna(advice) and str(advice).strip():
            st.markdown(str(advice))
        else:
            st.info("저장된 AI 분석 결과가 없습니다.")

        try:
            pdf_bytes = make_pdf_report(selected_record)
            st.download_button(
                "📄 선택한 분석 보고서 PDF 다운로드",
                data=pdf_bytes,
                file_name=f"safebot_report_{selected_id}.pdf",
                mime="application/pdf"
            )
        except ImportError:
            st.warning(
                "PDF 기능을 사용하려면 터미널에서 "
                "`pip install reportlab`을 실행해주세요."
            )
        except Exception as exc:
            st.warning(f"PDF 보고서 생성 중 오류가 발생했습니다: {exc}")
    else:
        st.info("상세 내용을 확인할 기록이 없습니다.")

    # --------------------------------------
    # 6-2. 기록 삭제
    # --------------------------------------

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
            st.success(f"분석 ID #{delete_id} 기록을 삭제했습니다.")
            st.rerun()
    else:
        st.info("삭제할 분석 기록이 없습니다.")

    # --------------------------------------
    # 6-3. CSV 다운로드
    # --------------------------------------

    st.divider()
    st.subheader("📥 분석 기록 다운로드")

    filtered_csv = filtered_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "🔍 검색 결과 CSV 다운로드",
        data=filtered_csv,
        file_name="safebot_filtered_history.csv",
        mime="text/csv"
    )

    all_csv = history_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📁 전체 분석 이력 CSV 다운로드",
        data=all_csv,
        file_name="safebot_analysis_history.csv",
        mime="text/csv"
    )

else:
    st.info("관리할 분석 기록이 없습니다.")


# ==========================================
# 7. 데이터베이스 정보 및 하단
# ==========================================

st.divider()
st.subheader("🗄️ 데이터베이스 정보")

current_history = load_history()
db_col1, db_col2 = st.columns(2)

with db_col1:
    st.metric("저장된 데이터", f"{len(current_history)}건")

with db_col2:
    st.caption(f"저장 위치: {DB_PATH}")

st.divider()
st.caption("SafeBot | S급 로봇 연구소 | 교육용 프로토타입")
