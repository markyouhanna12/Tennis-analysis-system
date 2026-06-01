from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
    def detect_frame(self,frame):
        results = self.model.predict(frame,conf=0.15)[0]
        #class_names = results.names
        ball_dict = {}
        for box in results.boxes:
            #track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
            #class_ids = box.cls.tolist()[0]
            #det_class_names = class_names[class_ids]
            #if det_class_names == "person":
                #ball_dict[track_id] = result
        return ball_dict

    def detect_frames(self,frames, read_from_stub = False, stub_path = None):
        ball_detections = []
        if read_from_stub and stub_path is not None:
            with open (stub_path,'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        if stub_path is not None:
            with open (stub_path,"wb") as f:
                pickle.dump(ball_detections, f)
        return ball_detections

    def draw_bboxes(self, video_frames, ball_detections):
        output_videos_frames = []
        for frame, player_dict in zip(video_frames, ball_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            output_videos_frames.append(frame)  # <-- moved outside inner loop
        return output_videos_frames


    def interpolate_ball_postions(self,ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=["x1","y1","x2","y2"])
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return  ball_positions



