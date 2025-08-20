import streamlit as st
import pandas as pd
import sys
import os

# --- Path Setup ---
# This ensures the app can find your 'src' module
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- Import your recommender function ---
# This line imports the "brain" of your new retrieval-based project
from src.service.recommender import recommend_for_user, RecommendConfig, FusionWeights

# --- Page Configuration ---
st.set_page_config(
    page_title="Multimodal Recommender",
    page_icon="🛍️",
    layout="wide"
)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Controls & Inputs")
    dataset_name = st.selectbox("Select a Dataset:", options=["beauty"], key="dataset_selector")
    
    # --- NEW IDEA: Add sample users for easy testing ---
    st.subheader("Select a User Profile")
    sample_users = ["A3H7T87S984REU", "A3J034YH7UG4KT", "A1S3C5OFU508P3", "A1T1PC2F2ZA82A"]
    user_selection = st.selectbox("Choose a sample existing user:", sample_users, key="user_selector")
    
    st.markdown("---")
    st.subheader("Or Try Another User")
    custom_user = st.text_input("Enter any existing or new User ID (e.g., 'new-user-123'):", key="custom_user_input")

    # Prioritize custom user input if the user types something
    final_user_id = custom_user if custom_user else user_selection

    st.markdown("---")
    st.button(
        "Get Recommendations", 
        key="get_recs_button",
        use_container_width=True,
    )

# --- Main Page ---
st.title("🛍️ Multimodal Recommender System")
st.markdown("Select a sample user or enter a new/existing user ID in the sidebar, then click 'Get Recommendations'.")
st.markdown("---")

# Only run if the button was pressed
if st.session_state.get("get_recs_button"):
    if not final_user_id.strip():
        st.error("Please select or enter a User ID.")
    else:
        st.header(f"✨ Recommendations for: {final_user_id}")

        with st.spinner('Calling the recommendation engine...'):
            try:
                # --- INTEGRATION: Build the config for the new backend ---
                cfg = RecommendConfig(
                    dataset="beauty",
                    user_id=final_user_id,
                    k=10,
                    fusion="concat",
                    weights=FusionWeights(text=1.0, image=1.0, meta=0.4),
                    use_faiss=True,
                    faiss_name="beauty_concat_best",
                    exclude_seen=True
                )
                
                # --- INTEGRATION: Call the new backend function directly ---
                result = recommend_for_user(cfg)
                recommendations = result.get("recommendations", [])

                if not recommendations:
                    st.warning("No recommendations could be found.")
                else:
                    # --- NEW FEATURE: Smart UI for new vs. existing users ---
                    if result.get("fusion") == "popularity":
                        st.success(f"Welcome, new user! Here are our Top 10 most popular products:")
                    else:
                        st.info(f"Showing personalized recommendations for existing user.")

                    # --- INTEGRATION: Display results in a single, richer format ---
                    for rec in recommendations:
                        with st.container(border=True):
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                # Placeholder image as this dataset has no valid image URLs
                                st.image("https://storage.googleapis.com/proudcity/mebanenc/uploads/2021/03/placeholder-image.png")
                            with col2:
                                st.subheader(rec.get('brand', 'Unknown Brand'))
                                st.caption(f"Product ID: {rec.get('item_id')}")
                                st.metric(label="Recommendation Score", value=f"{rec.get('score', 0.0):.4f}")
                                if rec.get('price'):
                                    st.write(f"**Price:** ${rec.get('price')}")
                                if rec.get('categories'):
                                    st.write(f"**Category:** {rec.get('categories')[0]}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
