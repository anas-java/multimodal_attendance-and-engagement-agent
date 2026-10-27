from datetime import datetime
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from attendance_core import (
    analyze_video, analyze_image, analyze_audio, load_attendance, save_attendance,
    build_session_insights, make_report, safe_filename
)

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
ATTENDANCE_PATH = DATA / "attendance.csv"

st.set_page_config(page_title="Multimodal Attendance & Engagement Agent", page_icon="🎓", layout="wide")
st.title("🎓 Multimodal Attendance & Engagement Agent")
st.caption("Local-first prototype: attendance estimates, head-pose cues, visual activity, audio activity, and teaching suggestions.")

with st.sidebar:
    st.header("Session configuration")
    session_name = st.text_input("Course / session", "Demo Classroom")
    instructor = st.text_input("Instructor (optional)", "")
    minimum_presence = st.slider("Minimum presence to mark a track present (seconds)", 1, 120, 10)
    sample_every = st.slider("Process every Nth video frame", 1, 12, 3)
    use_mediapipe = st.checkbox("Use MediaPipe head-pose estimation when available", True)
    blur_faces = st.checkbox("Blur faces in preview", True)
    st.divider()
    st.subheader("Privacy")
    st.write("Media is processed locally. Nothing is sent to an external API.")
    st.caption("Track IDs are temporary and do not identify real-world students.")

video_tab, audio_tab, fusion_tab, image_tab, records_tab, about_tab = st.tabs([
    "Video & attendance", "Audio analysis", "Multimodal insights", "Snapshot test", "Records", "Method & privacy"
])

if "video_result" not in st.session_state:
    st.session_state.video_result = None
if "audio_result" not in st.session_state:
    st.session_state.audio_result = None

with video_tab:
    st.subheader("Analyze a recorded classroom video")
    st.write("Use a short, consented test clip first. Supported formats depend on your local OpenCV codecs.")
    vf = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"], key="video_upload")
    if vf and st.button("Analyze video", type="primary"):
        temp = DATA / safe_filename(vf.name)
        temp.write_bytes(vf.getvalue())
        try:
            with st.spinner("Processing video on this device…"):
                st.session_state.video_result = analyze_video(
                    str(temp), min_presence_seconds=minimum_presence,
                    sample_every=sample_every, use_mediapipe=use_mediapipe,
                    blur_preview=blur_faces
                )
            st.success("Video analysis completed.")
        except Exception as exc:
            st.error(f"Video processing failed: {exc}")
        finally:
            temp.unlink(missing_ok=True)

    vr = st.session_state.video_result
    if vr:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Video duration", f'{vr["duration_seconds"]:.1f} s')
        c2.metric("Anonymous tracks", vr["unique_tracks"])
        c3.metric("Mean motion index", f'{vr["mean_motion"]:.3f}')
        c4.metric("Head-pose coverage", f'{vr["head_pose_coverage_percent"]:.0f}%')
        if vr.get("preview") is not None:
            st.image(cv2.cvtColor(vr["preview"], cv2.COLOR_BGR2RGB), caption="Sample preview (face regions blurred when enabled)")
        st.info(vr["tracking_note"])
        table = pd.DataFrame(vr["tracks"])
        if not table.empty:
            st.subheader("Per-track estimates")
            st.dataframe(table, use_container_width=True)
        with st.expander("Visual timeline (session-level)"):
            if vr.get("timeline"):
                tl = pd.DataFrame(vr["timeline"]).set_index("time_seconds")
                st.line_chart(tl)
        if st.button("Save attendance records locally"):
            stamp = datetime.now().isoformat(timespec="seconds")
            rows = [{
                "timestamp": stamp, "session": session_name, "instructor": instructor,
                "participant_id": f'Track {t["track_id"]}',
                "present_seconds": round(t["present_seconds"], 2),
                "attendance_status": "Present" if t["present_seconds"] >= minimum_presence else "Review",
                "source": "video", "review_required": True
            } for t in vr["tracks"]]
            old = load_attendance(ATTENDANCE_PATH)
            save_attendance(pd.concat([old, pd.DataFrame(rows)], ignore_index=True), ATTENDANCE_PATH)
            st.success(f"Saved {len(rows)} records. Review before official use.")

with audio_tab:
    st.subheader("Audio activity analysis")
    st.write("Audio is analyzed locally for sound-energy and simple speech-like acoustic cues. No transcript or speaker identity is produced.")
    af = st.file_uploader("Upload audio", type=["wav", "flac", "ogg", "mp3", "m4a"], key="audio_upload")
    if af and st.button("Analyze audio", type="primary"):
        try:
            with st.spinner("Analyzing audio locally…"):
                st.session_state.audio_result = analyze_audio(af.getvalue(), af.name)
            st.success("Audio analysis completed.")
        except Exception as exc:
            st.error(f"Audio processing failed: {exc}. Try a PCM WAV file.")
    ar = st.session_state.audio_result
    if ar:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duration", f'{ar["duration_seconds"]:.1f} s')
        c2.metric("Active sound windows", f'{ar["active_percent"]:.1f}%')
        c3.metric("Speech-like windows", f'{ar["speech_like_percent"]:.1f}%')
        c4.metric("Mean RMS", f'{ar["mean_rms"]:.4f}')
        chart = pd.DataFrame({"RMS energy": ar["rms"], "Zero-crossing rate": ar["zcr"]})
        st.line_chart(chart)
        st.caption("Speech-like is a heuristic based on energy and zero-crossing rate; it is not speech recognition and may misclassify noise.")

with fusion_tab:
    st.subheader("Session-level multimodal insights")
    st.write("This combines session-level video and audio summaries by relative time. It does not claim to match a sound to an individual student.")
    if st.session_state.video_result or st.session_state.audio_result:
        insights = build_session_insights(st.session_state.video_result, st.session_state.audio_result)
        for item in insights["suggestions"]:
            st.markdown(f"- {item}")
        st.subheader("Summary")
        st.json(insights["summary"])
        if insights.get("timeline"):
            st.line_chart(pd.DataFrame(insights["timeline"]).set_index("time_seconds"))
        report = make_report(session_name, instructor, st.session_state.video_result, st.session_state.audio_result, insights)
        st.download_button("Download session report (JSON)", json.dumps(report, indent=2).encode(), "session_report.json", "application/json")
    else:
        st.info("Analyze a video and/or audio file first. Both are needed for the combined session timeline.")

with image_tab:
    st.subheader("Snapshot face-detection test")
    st.write("A still image is for testing only; one frame is not enough to establish attendance.")
    cam = st.camera_input("Take a snapshot")
    img = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"], key="still_upload")
    src = cam or img
    if src:
        import PIL.Image
        rgb = np.array(PIL.Image.open(src).convert("RGB"))
        result = analyze_image(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), blur=True)
        st.image(cv2.cvtColor(result["preview"], cv2.COLOR_BGR2RGB))
        st.metric("Faces detected", result["faces"])
        st.warning("Snapshot detection does not create attendance records.")

with records_tab:
    st.subheader("Local attendance records")
    df = load_attendance(ATTENDANCE_PATH)
    if df.empty:
        st.info("No records saved yet.")
    else:
        st.metric("Records", len(df))
        st.dataframe(df, use_container_width=True)
        st.download_button("Download attendance CSV", df.to_csv(index=False).encode(), "attendance.csv", "text/csv")
        if st.button("Delete all attendance records", type="secondary"):
            save_attendance(pd.DataFrame(), ATTENDANCE_PATH)
            st.rerun()

with about_tab:
    st.subheader("Architecture")
    st.code("""Video -> OpenCV frame sampling -> face detection -> anonymous IoU tracks
                               -> pose estimates + movement + presence duration
Audio -> local decoding -> RMS + zero-crossing rate -> activity windows
Video + audio summaries -> relative-time session fusion -> rule-based suggestions
Local CSV/JSON -> Streamlit dashboard and downloadable reports""", language="text")
    st.subheader("Privacy safeguards")
    st.markdown("""
- Processing is local; no cloud APIs or external model endpoints are called.
- Temporary uploaded media is removed after processing.
- Attendance is stored only when the user presses the save button.
- Preview faces are blurred by default.
- No face embeddings, face recognition, student-name matching, emotion inference, or speaker identification.
- Use informed consent, restrict access to the device, set a retention policy, and offer a manual correction process.
- Do not use engagement proxies for grades, discipline, disability inference, or other high-stakes decisions.
""")
    st.subheader("Interpret results carefully")
    st.markdown("""
The face detector and simple IoU tracker are not production-grade. Track IDs may split or merge. Head pose is an approximate geometric cue, not proof of attention. Movement and audio activity are weak proxies that can be affected by camera angle, lighting, disability, room noise, and teaching style. All records need human review.
""")
