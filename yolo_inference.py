from ultralytics import YOLO
model = YOLO("models/tennis_ball_best.pt")
results=model.predict("input_videos/input_video.mp4",conf=0.2,save = True ,show=True)
#results=model.track("input_videos/input_video.mp4",conf=0.2, save = True ,persist= True)
