import streamlit as st
import urllib.parse

st.set_page_config(page_title="AI Video Face Recognition", layout="centered")

st.title("🎬 YouTube Video Face Recognition")
st.subheader("Powered by InsightFace & CUDA Acceleration")

# 1. URL Input Field
yt_url = st.text_input(
    "Enter YouTube Video Link:", 
    placeholder="https://www.youtube.com/watch?v=..."
)

# Initialize execution state
if "stream_active" not in st.session_state:
    st.session_state.stream_active = False

# 2. Control Layout Buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Analyze Video", use_container_width=True):
        if yt_url:
            st.session_state.stream_active = True
        else:
            st.warning("Please paste a valid YouTube link first!")

with col2:
    if st.button("🛑 Stop Video", use_container_width=True):
        st.session_state.stream_active = False

st.write("---")

# 3. Stream logic
if st.session_state.stream_active and yt_url:
    # URL encoding ensures characters like & and ? in video links don't mess up the API routing
    encoded_url = urllib.parse.quote_plus(yt_url)
    api_endpoint = f"http://127.0.0.1:8000/video_feed?url={encoded_url}"

    st.info("Extracting live stream and tracking faces...")
    st.image(api_endpoint, use_container_width=True)
else:
    st.info("Ready. Paste a link and click 'Analyze Video' to run live tracking.")