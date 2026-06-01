import cv2
from ultralytics import YOLO
from utils import (read_video,save_video)
from trackers import PlayerTracker , BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
def main():
    # Read input video path
    input__video_path = "input_videos/Untitled video - Made with Clipchamp (1).mp4"
    video_frames,fps = read_video(input__video_path)


    # Detect players
    player_tracker = PlayerTracker(model_path="yolo11x.pt")
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub = True , stub_path ="tracker_stubs/player_detection3.pkl")

    # Detect tennis ball
    ball_tracker = BallTracker(model_path="models/tennis_ball_best.pt")
    ball_detections = ball_tracker.detect_frames(video_frames,read_from_stub = True , stub_path ="tracker_stubs/ball_detection3.pkl")
    # initialize  Mini Court
    mini_court = MiniCourt(video_frames[0])

    #Interpolate ball Postions
    ball_detections = ball_tracker.interpolate_ball_postions(ball_detections)

    # Detect tennis court keypoints
    court_model_path="models/keypoints_model_50.pth"
    court_line_detector= CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])


    # Choose the two Players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints,player_detections)


    # Draw player bounding Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames,player_detections)
    # Draw tennis ball bounding Boxes
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames,ball_detections)
    # Draw court keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames,court_keypoints)

    # Draw the Mini court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)


    
    # Convert Positions into Mini Court Positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, ball_detections, court_keypoints)

    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)

    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0, 255, 255))


    # Save output video
    save_video(output_video_frames,"output_videos/output3.avi",fps)

if __name__ == "__main__":
    main()