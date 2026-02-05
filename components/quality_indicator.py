import streamlit as st


def render_quality_badge(score: int, query_type: str):
    if score >= 4:
        color, label = "green", "High Quality"
    elif score >= 3:
        color, label = "orange", "Acceptable"
    else:
        color, label = "red", "Low Quality"

    type_icon = {"sql": "📊", "rag": "📄", "hybrid": "📊📄"}.get(query_type, "❓")

    st.markdown(
        f"""<div style="display: flex; gap: 12px; align-items: center; padding: 4px 0;">
            <span style="background: {color}; color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.85em;">
                {label} ({score}/5)
            </span>
            <span style="font-size: 0.85em;">{type_icon} {query_type.upper()}</span>
        </div>""",
        unsafe_allow_html=True,
    )
