from pathlib import Path
import tempfile
import os
import wave
import json

import cv2
import numpy as np
import pandas as pd

def safe_filename(name):
    cleaned = "".join(c for c in Path(name).name if c.isalnum() or c in "._-")
    return cleaned[:100] or "upload.bin"

def _classifier():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def _detect_faces(gray, classifier):
    return [tuple(map(int, f)) for f in classifier.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(35, 35)
    )]

def analyze_image(frame_bgr, blur=True):
    faces = _detect_faces(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), _classifier())
    out = frame_bgr.copy()
    for x, y, w, h in faces:
        if blur:
            roi = out[y:y+h, x:x+w]
            if roi.size:
                out[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (31, 31), 0)
        cv2.rectangle(out, (x, y), (x+w, y+h), (30, 170, 90), 2)
    return {"faces": len(faces), "preview": out}

def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = aw*ah + bw*bh - inter
    return inter / union if union else 0.0

def _head_pose(frame_bgr, face_box, face_mesh):
    """Approximate yaw/pitch/roll using Face Mesh landmarks and solvePnP."""
    if face_mesh is None:
        return None
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None
    # Use first mesh; acceptable only as a rough prototype cue.
    lm = result.multi_face_landmarks[0].landmark
    indices = [1, 152, 33, 263, 61, 291]
    image_points = np.array([[lm[i].x*w, lm[i].y*h] for i in indices], dtype=np.float64)
    # Generic 3-D facial reference points (not personalized to the subject).
    model_points = np.array([
        (0.0, 0.0, 0.0), (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0), (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1)
    ], dtype=np.float64)
    focal = float(w)
    camera = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)
    ok, rotation, _ = cv2.solvePnP(model_points, image_points, camera, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    matrix, _ = cv2.Rodrigues(rotation)
    sy = np.sqrt(matrix[0,0]**2 + matrix[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(matrix[2,1], matrix[2,2])
        y = np.arctan2(-matrix[2,0], sy)
        z = np.arctan2(matrix[1,0], matrix[0,0])
    else:
        x = np.arctan2(-matrix[1,2], matrix[1,1])
        y = np.arctan2(-matrix[2,0], sy)
        z = 0
    return {"pitch_deg": float(np.degrees(x)), "yaw_deg": float(np.degrees(y)), "roll_deg": float(np.degrees(z))}

def analyze_video(path, min_presence_seconds=10, sample_every=3, use_mediapipe=True, blur_preview=True):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open video. Try re-encoding as MP4 (H.264) or AVI.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / fps if fps > 0 else 0
    detector = _classifier()
    face_mesh = None
    mp_available = False
    if use_mediapipe:
        try:
            import mediapipe as mp
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1, refine_landmarks=False,
                min_detection_confidence=0.5
            )
            mp_available = True
        except Exception:
            face_mesh = None

    tracks = {}
    next_id = 1
    frame_idx = 0
    previous_gray = None
    motion_values = []
    timeline = []
    preview = None
    pose_samples = 0
    processed = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % max(1, int(sample_every)) != 0:
                frame_idx += 1
                continue
            processed += 1
            now_s = frame_idx / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (max(1, gray.shape[1]//2), max(1, gray.shape[0]//2)))
            motion = 0.0
            if previous_gray is not None and previous_gray.shape == small.shape:
                motion = float(np.mean(cv2.absdiff(small, previous_gray)) / 255.0)
                motion_values.append(motion)
            previous_gray = small
            faces = _detect_faces(gray, detector)

            # Greedy IoU association, with tracks expiring after 2 seconds of missed samples.
            active_ids = [tid for tid, t in tracks.items() if now_s - t["last_seen"] <= 2.0]
            assigned = set()
            detections_for_frame = []
            for box in faces:
                best_id, best_score = None, 0.0
                for tid in active_ids:
                    if tid in assigned:
                        continue
                    score = _iou(box, tracks[tid]["last_box"])
                    if score > best_score:
                        best_id, best_score = tid, score
                if best_id is None or best_score < 0.12:
                    best_id = next_id
                    next_id += 1
                    tracks[best_id] = {
                        "track_id": best_id, "first_seen": now_s, "last_seen": now_s,
                        "last_box": box, "detections": 0, "motion_samples": [],
                        "yaw": [], "pitch": [], "roll": []
                    }
                t = tracks[best_id]
                t["last_box"] = box
                t["last_seen"] = now_s
                t["detections"] += 1
                t["motion_samples"].append(motion)
                assigned.add(best_id)
                detections_for_frame.append((box, best_id))
                if mp_available and face_mesh is not None and len(t["yaw"]) < 500:
                    try:
                        pose = _head_pose(frame, box, face_mesh)
                        if pose:
                            for key in ("yaw_deg", "pitch_deg", "roll_deg"):
                                t[{"yaw_deg":"yaw","pitch_deg":"pitch","roll_deg":"roll"}[key]].append(pose[key])
                            pose_samples += 1
                    except Exception:
                        pass

            if preview is None:
                preview = frame.copy()
                for (x,y,w,h), tid in detections_for_frame:
                    if blur_preview:
                        roi = preview[y:y+h, x:x+w]
                        if roi.size:
                            preview[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (31,31), 0)
                    cv2.rectangle(preview, (x,y), (x+w,y+h), (40,180,90), 2)
                    cv2.putText(preview, f"Track {tid}", (x,max(20,y-8)), cv2.FONT_HERSHEY_SIMPLEX, .6, (40,180,90), 2)

            timeline.append({
                "time_seconds": round(now_s, 2),
                "motion_index": motion,
                "faces_detected": len(faces),
                "active_audio": np.nan
            })
            frame_idx += 1
    finally:
        cap.release()
        if face_mesh is not None:
            try:
                face_mesh.close()
            except Exception:
                pass

    out = []
    for tid, t in tracks.items():
        # Presence estimate is duration between first and last observed detections.
        present = max(0.0, t["last_seen"] - t["first_seen"])
        yaw = t["yaw"]
        pitch = t["pitch"]
        out.append({
            "track_id": tid,
            "first_seen_seconds": round(t["first_seen"], 2),
            "last_seen_seconds": round(t["last_seen"], 2),
            "present_seconds": round(present, 2),
            "detections": t["detections"],
            "attendance_status": "Present" if present >= min_presence_seconds else "Review",
            "mean_abs_yaw_deg": round(float(np.mean(np.abs(yaw))), 1) if yaw else None,
            "mean_abs_pitch_deg": round(float(np.mean(np.abs(pitch))), 1) if pitch else None,
            "head_pose_samples": len(yaw),
            "mean_motion_index": round(float(np.mean(t["motion_samples"])), 4) if t["motion_samples"] else 0.0
        })
    if preview is None:
        cap2 = cv2.VideoCapture(path)
        ok, frame = cap2.read()
        cap2.release()
        preview = frame if ok else None
    return {
        "duration_seconds": round(float(duration), 2),
        "unique_tracks": len(out),
        "mean_motion": float(np.mean(motion_values)) if motion_values else 0.0,
        "head_pose_coverage_percent": 100.0 * pose_samples / max(1, sum(t["detections"] for t in tracks.values())),
        "tracks": out,
        "timeline": timeline,
        "preview": preview,
        "mediapipe_used": mp_available,
        "tracking_note": "MediaPipe head pose was enabled." if mp_available else "MediaPipe unavailable or disabled; face detection still ran. Install requirements and restart to enable head-pose estimates."
    }

def _decode_audio(data, filename):
    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        import soundfile as sf
        audio, sr = sf.read(path, always_2d=True)
        audio = audio.mean(axis=1).astype(np.float32)
        return audio, int(sr)
    except Exception:
        if suffix != ".wav":
            raise ValueError("Could not decode this audio. Try PCM WAV or install a compatible audio decoder.")
        with wave.open(path, "rb") as wf:
            sr, width, channels = wf.getframerate(), wf.getsampwidth(), wf.getnchannels()
            raw = wf.readframes(wf.getnframes())
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
        if dtype is None:
            raise ValueError("Unsupported WAV sample width.")
        audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if width == 1:
            audio = (audio - 128) / 128.0
        else:
            audio /= float(2 ** (8*width-1))
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        return audio, int(sr)
    finally:
        try: os.unlink(path)
        except OSError: pass

def analyze_audio(data, filename, window_seconds=0.5):
    audio, sr = _decode_audio(data, filename)
    if len(audio) == 0 or sr <= 0:
        raise ValueError("Audio is empty or invalid.")
    window = max(1, int(sr * window_seconds))
    rms, zcr, times = [], [], []
    for start in range(0, len(audio), window):
        chunk = audio[start:start+window]
        if not len(chunk): continue
        rms.append(float(np.sqrt(np.mean(np.square(chunk)))))
        signs = np.signbit(chunk)
        zcr.append(float(np.mean(signs[1:] != signs[:-1])) if len(chunk) > 1 else 0.0)
        times.append(round(start/sr, 2))
    active_threshold = max(0.008, float(np.percentile(rms, 60)) * 0.30)
    active = [v >= active_threshold for v in rms]
    # Heuristic only: non-silent energy with a broad ZCR range may be speech-like.
    speech_like = [a and 0.015 <= z <= 0.35 for a, z in zip(active, zcr)]
    return {
        "duration_seconds": len(audio)/sr,
        "rms": rms, "zcr": zcr, "time_seconds": times,
        "mean_rms": float(np.mean(rms)) if rms else 0.0,
        "active_percent": 100*sum(active)/max(1,len(active)),
        "speech_like_percent": 100*sum(speech_like)/max(1,len(speech_like)),
        "timeline": [{"time_seconds": t, "audio_rms": r, "audio_active": int(a), "speech_like": int(s)}
                     for t,r,a,s in zip(times,rms,active,speech_like)]
    }

def build_session_insights(video_result=None, audio_result=None):
    suggestions = []
    summary = {}
    video_tl = video_result.get("timeline", []) if video_result else []
    audio_tl = audio_result.get("timeline", []) if audio_result else []
    if video_result:
        summary.update({
            "video_duration_seconds": video_result["duration_seconds"],
            "anonymous_tracks": video_result["unique_tracks"],
            "mean_motion_index": round(video_result["mean_motion"], 4),
            "mediapipe_head_pose_available": video_result["mediapipe_used"]
        })
        tracks = video_result["tracks"]
        if not tracks:
            suggestions.append("No faces were detected reliably. Check camera angle, lighting, and whether the video contains visible faces.")
        else:
            short = sum(t["attendance_status"] != "Present" for t in tracks)
            if short:
                suggestions.append(f"{short} track(s) fell below the configured sustained-presence threshold. Review the clip manually before deciding attendance.")
            if video_result["mean_motion"] < 0.008:
                suggestions.append("The visual motion index was low. This may indicate a static camera/lecture, not low attention; consider adding a short poll or think-pair-share activity if pedagogically appropriate.")
            else:
                suggestions.append("Movement was detected in the video. Check whether it corresponds to discussion, note-taking, camera movement, or unrelated motion before interpreting it.")
            poses = [t for t in tracks if t["mean_abs_yaw_deg"] is not None]
            if poses:
                avg_yaw = float(np.mean([t["mean_abs_yaw_deg"] for t in poses]))
                summary["mean_absolute_head_yaw_degrees"] = round(avg_yaw, 1)
                suggestions.append("Head-pose estimates are available for some tracks. Use them only as rough camera-relative orientation cues; do not equate gaze direction with attention.")
            else:
                suggestions.append("Head-pose estimates were unavailable. Install MediaPipe successfully and use a front-facing, well-lit test clip.")
    if audio_result:
        summary.update({
            "audio_duration_seconds": round(audio_result["duration_seconds"], 2),
            "active_audio_percent": round(audio_result["active_percent"], 1),
            "speech_like_audio_percent_heuristic": round(audio_result["speech_like_percent"], 1)
        })
        if audio_result["active_percent"] < 10:
            suggestions.append("Audio activity was low. Verify microphone placement and recording level; silence alone does not indicate disengagement.")
        elif audio_result["speech_like_percent"] > 25:
            suggestions.append("The audio contains speech-like acoustic windows. If useful, invite a brief recap or question to check understanding; the app cannot identify speakers or confirm participation.")
        else:
            suggestions.append("Audio activity was detected but speech-like evidence was limited. Background noise and quiet speech can confuse this heuristic.")
    if video_result and audio_result:
        delta = abs(video_result["duration_seconds"] - audio_result["duration_seconds"])
        summary["video_audio_duration_difference_seconds"] = round(delta, 2)
        if delta > 2:
            suggestions.append("Video and audio lengths differ by more than two seconds. Timeline fusion is approximate; trim/synchronize the files before interpreting trends.")
        suggestions.append("Combined analysis is session-level only. It does not link audio events to specific people.")
    if not suggestions:
        suggestions.append("Upload video or audio to generate evidence-based session suggestions.")
    return {"summary": summary, "suggestions": suggestions, "timeline": _fuse_timelines(video_tl, audio_tl)}

def _fuse_timelines(video_tl, audio_tl):
    if not video_tl and not audio_tl:
        return []
    # Merge by nearest 0.5-second bucket, using relative time from the start.
    buckets = {}
    for row in video_tl:
        key = round(float(row["time_seconds"])*2)/2
        b = buckets.setdefault(key, {"time_seconds": key})
        b["motion_index"] = row.get("motion_index")
        b["faces_detected"] = row.get("faces_detected")
    for row in audio_tl:
        key = round(float(row["time_seconds"])*2)/2
        b = buckets.setdefault(key, {"time_seconds": key})
        b["audio_rms"] = row.get("audio_rms")
        b["audio_active"] = row.get("audio_active")
        b["speech_like"] = row.get("speech_like")
    return [buckets[k] for k in sorted(buckets)]

def make_report(session, instructor, video, audio, insights):
    clean_video = None
    if video:
        clean_video = {k:v for k,v in video.items() if k != "preview"}
    return {
        "generated_at": pd.Timestamp.now().isoformat(),
        "session": session,
        "instructor": instructor,
        "video_analysis": clean_video,
        "audio_analysis": audio,
        "insights": insights,
        "limitations": [
            "Anonymous tracks are not real-world identities and may split/merge.",
            "Head pose, motion, RMS and speech-like estimates are noisy proxies, not attention or learning measurements.",
            "Human review is required before attendance is finalized."
        ]
    }

def load_attendance(path):
    path = Path(path)
    if path.exists() and path.stat().st_size:
        try: return pd.read_csv(path)
        except Exception: pass
    return pd.DataFrame(columns=["timestamp","session","instructor","participant_id","present_seconds","attendance_status","source","review_required"])

def save_attendance(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
