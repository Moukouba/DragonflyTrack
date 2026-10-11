"""Test script for DragonflyTrack-X video tracking using YOLOv8 and appearance embedder."""

import os
import time
import cv2
import numpy as np

from dragonflytrack import (
    DragonflyTrackX,
    VideoFrameSource,
    YOLODetectorAdapter,
    HSVColorHistEmbedder,
    draw_tracks,
    MOTWriter,
)


def run_footage_test(
    video_path="footage/Shopping People Commerce Mall Many Crowd Walking Free Stock vide.mp4",
    yolo_weights="yolov8n.pt",
    out_dir="results_x",
    conf=0.15,
    save_video=True,
    max_frames=None,
):
    os.makedirs(out_dir, exist_ok=True)
    mot_out_path = os.path.join(out_dir, "footage_tracks.txt")
    video_out_path = os.path.join(out_dir, "footage_tracked.mp4")

    print(f"[+] Initializing video source: {video_path}")
    source = VideoFrameSource(video_path)

    print(f"[+] Initializing YOLOv8 detector with model: {yolo_weights}")
    detector = YOLODetectorAdapter(yolo_weights, conf=conf, classes=(0,))

    print(f"[+] Initializing HSV Color Histogram Appearance Embedder")
    embedder = HSVColorHistEmbedder()

    print(f"[+] Initializing DragonflyTrackX tracker with long-occlusion memory (max_age=60)")
    tracker = DragonflyTrackX(
        det_thresh=0.35,
        low_thresh=0.1,
        new_thresh=0.45,
        max_age=60,
        lost_maha_gate=60.0,
        reid_dim=embedder.dim,
        gmc=True,
        use_diou=True,
        use_ocsort=True,
    )

    writer = MOTWriter(mot_out_path)
    video_writer = None

    t0 = time.time()
    frame_count = 0
    total_dets = 0
    tracks_per_frame = []

    print("[+] Processing video frames...")

    try:
        for fid, frame in source:
            dets = detector(frame, fid)
            tracks = tracker.update(dets, frame=frame, embedder=embedder)
            writer.write(fid, tracks)

            frame_count += 1
            total_dets += len(dets)
            tracks_per_frame.append(len(tracks))

            if save_video:
                annotated = draw_tracks(frame, tracks)
                bgr_frame = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

                if video_writer is None:
                    h, w = bgr_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(video_out_path, fourcc, 25.0, (w, h))
                video_writer.write(bgr_frame)

            if frame_count % 50 == 0:
                fps = frame_count / (time.time() - t0)
                print(
                    f"    Frame {fid:3d} | Live Tracks: {len(tracks):2d} | Dets: {len(dets):2d} | Speed: {fps:.1f} FPS"
                )

            if max_frames and frame_count >= max_frames:
                break
    finally:
        writer.close()
        if video_writer is not None:
            video_writer.release()
        source.close()

    elapsed = time.time() - t0
    avg_fps = frame_count / elapsed if elapsed > 0 else 0
    unique_ids = tracker._next_id - 1

    print("\n" + "=" * 55)
    print("           DRAGONFLYTRACK-X FOOTAGE TEST SUMMARY")
    print("=" * 55)
    print(f"Total Frames Processed : {frame_count}")
    print(f"Total Execution Time   : {elapsed:.2f} seconds")
    print(f"Average Speed          : {avg_fps:.2f} FPS")
    print(f"Total Detections       : {total_dets}")
    print(f"Unique Track IDs       : {unique_ids}")
    print(f"Avg Live Tracks/Frame  : {np.mean(tracks_per_frame):.2f}")
    print("-" * 55)
    print("Tracker Statistics Breakdown:")
    for k, v in tracker.stats.items():
        print(f"  - {k:15s} : {v}")
    print("-" * 55)
    print(f"MOT Format Track File  : {mot_out_path}")
    if save_video:
        print(f"Annotated Video File   : {video_out_path}")
    print("=" * 55 + "\n")

    return {
        "frames": frame_count,
        "fps": avg_fps,
        "unique_ids": unique_ids,
        "stats": tracker.stats,
        "mot_file": mot_out_path,
        "video_file": video_out_path,
    }


if __name__ == "__main__":
    import sys
    # Check if yolo_weights is passed as command line argument
    yolo_weights = "yolov8n.pt"
    for arg in sys.argv[1:]:
        if "yolo_weights" in arg or arg.endswith(".pt"):
            if "=" in arg:
                yolo_weights = arg.split("=")[1]
            else:
                yolo_weights = arg
    run_footage_test(yolo_weights=yolo_weights)
