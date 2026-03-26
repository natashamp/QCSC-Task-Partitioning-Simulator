"""Renders concept explanation panels triggered by simulation events."""

import streamlit as st
from streamlit_app.concepts.definitions import CONCEPTS, get_triggered_concepts


def render_concept_panels(sub_events, step_number, triggered_concepts):
    """Check for new concepts to show and render them. Returns updated triggered set."""
    new_concepts = get_triggered_concepts(sub_events, step_number, triggered_concepts)

    for key in new_concepts:
        concept = CONCEPTS[key]
        icon = concept.get("icon", "info")

        if icon == "warning":
            st.warning(f"**{concept['title']}**\n\n{concept['body']}")
        elif icon == "success":
            st.success(f"**{concept['title']}**\n\n{concept['body']}")
        else:
            st.info(f"**{concept['title']}**\n\n{concept['body']}")

        triggered_concepts.add(key)

    return triggered_concepts
