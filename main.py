import cv2
from ultralytics import YOLO
import argparse

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Object Detection and Tracking using YOLOv8")
    parser.add_argument("--source", type=str, default="0", help="Video source: '0' for webcam or path to video file")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model to use (e.g., yolov8n.pt, yolov8m.pt)")
    args = parser.parse_args()

    # Handle webcam source (convert "0" string to int 0)
    source = args.source
    if source == "0":
        source = 0
    
    # Load the YOLOv8 model
    print(f"Loading model {args.model}...")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    print("Starting Object Detection and Tracking. Press 'q' to exit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("End of video or failed to read frame.")
            break

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        # persist=True is crucial for tracking to maintain IDs
        results = model.track(source=frame, persist=True, tracker="bytetrack.yaml")

        # Visualize the results on the frame
        # plot() draws bounding boxes, labels, and confidence scores
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Object Detection & Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
