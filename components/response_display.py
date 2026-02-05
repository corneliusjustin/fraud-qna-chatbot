import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from models.schemas import AgentResponse
from models.enums import QueryType


def render_response(response: AgentResponse, skip_answer: bool = False):
    if not skip_answer:
        st.markdown(response.answer)

    # Render charts for SQL results
    if response.sql_result and not response.sql_result.error and response.sql_result.rows:
        _render_sql_visualization(response)

    # Render sources
    if response.sources:
        with st.expander("📎 Sources", expanded=False):
            for src in response.sources:
                st.markdown(f"- {src}")

    # Render quality details
    if response.quality_score:
        _render_quality_details(response)


def _render_sql_visualization(response: AgentResponse):
    sql = response.sql_result
    if not sql or not sql.columns or not sql.rows:
        return

    df = pd.DataFrame(sql.rows, columns=sql.columns)

    # Detect time-series data for line chart
    time_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["month", "date", "year", "day", "time", "period"])]
    numeric_cols = df.select_dtypes(include=["number", "float", "int"]).columns.tolist()

    if time_cols and numeric_cols:
        with st.expander("📈 Chart", expanded=True):
            fig = px.line(
                df,
                x=time_cols[0],
                y=numeric_cols[0],
                title=f"{numeric_cols[0]} over {time_cols[0]}",
                markers=True,
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Detect categorical data for bar chart
    elif len(df.columns) >= 2 and len(df) <= 30:
        cat_cols = [c for c in df.columns if df[c].dtype == "object" or c.lower() in ["category", "merchant", "state", "city", "job", "gender"]]
        if cat_cols and numeric_cols:
            with st.expander("📊 Chart", expanded=True):
                fig = px.bar(
                    df,
                    x=cat_cols[0],
                    y=numeric_cols[0],
                    title=f"{numeric_cols[0]} by {cat_cols[0]}",
                    color=numeric_cols[0],
                    color_continuous_scale="Reds",
                )
                fig.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig, use_container_width=True)

    # Always show data table
    with st.expander("🗂️ Raw Data", expanded=False):
        st.dataframe(df, use_container_width=True)


def _render_quality_details(response: AgentResponse):
    qs = response.quality_score
    with st.expander("🔍 Quality Details", expanded=False):
        st.markdown(f"**Score:** {qs.score}/5")
        st.markdown(f"**Reasoning:** {qs.reasoning}")
        if qs.has_hallucination:
            st.warning("⚠️ Potential hallucination detected")
        if qs.missing_information:
            st.markdown("**Missing info:**")
            for item in qs.missing_information:
                st.markdown(f"- {item}")
        if response.retry_count > 0:
            st.info(f"Response was refined {response.retry_count} time(s)")
