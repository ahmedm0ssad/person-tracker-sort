import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os
import argparse
from scipy.interpolate import interp1d

# Add the correct path to find sort.py
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir + '/sort')
from sort import Sort

# Global variable for person to track - accessible in callback function
target_id = None

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Track a person in a video and visualize their path")
    parser.add_argument('--input', type=str, default='Mossad.mp4', help='Path to input video')
    parser.add_argument('--output', type=str, default='output_tracking.mp4', help='Path to output video')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (process only 10 seconds)')
    parser.add_argument('--thickness', type=int, default=5, help='Thickness of the path line')
    parser.add_argument('--color', type=str, default='red', help='Color of the path (red, green, blue, yellow, etc.)')
    parser.add_argument('--model', type=str, default='yolo12n.pt', help='YOLOv12 model to use (n, s, m, l, x)')
    parser.add_argument('--min-hits', type=int, default=3, help='Minimum hits for SORT tracker')
    parser.add_argument('--max-age', type=int, default=30, help='Maximum age for SORT tracker (frames)')
    parser.add_argument('--iou-threshold', type=float, default=0.3, help='IOU threshold for SORT tracker')
    parser.add_argument('--fade-effect', action='store_true', help='Apply fade effect to path visualization')
    
    return parser.parse_args()

def get_color(color_name):
    """
    Convert color name to BGR tuple
    """
    color_map = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),
        'purple': (255, 0, 255),
        'cyan': (255, 255, 0),
        'orange': (0, 165, 255),
        'pink': (203, 192, 255),
        'white': (255, 255, 255),
        'black': (0, 0, 0)
    }
    
    return color_map.get(color_name.lower(), (0, 0, 255)) 

def load_video(video_path):
    """
    Load video from path and return VideoCapture object
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return None
    return cap

def initialize_video_writer(cap, output_path):
    """
    Initialize a video writer with same properties as input video
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    return out, (width, height), fps

def smooth_path(path_points, smoothing_factor=10):
    """
    Smooth the path using interpolation for more natural visualization
    
    Args:
        path_points: List of (x, y) tuples representing the path
        smoothing_factor: Higher values create smoother paths
        
    Returns:
        Smoothed path as list of (x, y) tuples
    """
    if len(path_points) < 3:
        return path_points
    
    # Extract x and y coordinates
    x_coords = [p[0] for p in path_points]
    y_coords = [p[1] for p in path_points]
    
    # Create an array of indices
    indices = np.arange(len(path_points))
    
    try:
        # Create interpolation functions - use linear if cubic fails
        try:
            x_interp = interp1d(indices, x_coords, kind='cubic', fill_value='extrapolate')
            y_interp = interp1d(indices, y_coords, kind='cubic', fill_value='extrapolate')
        except Exception as e:
            print(f"Warning: Cubic interpolation failed, using linear. Error: {e}")
            x_interp = interp1d(indices, x_coords, kind='linear', fill_value='extrapolate')
            y_interp = interp1d(indices, y_coords, kind='linear', fill_value='extrapolate')
        
        # Create more points for a smoother curve
        smooth_indices = np.linspace(0, len(path_points) - 1, len(path_points) * smoothing_factor)
        
        # Interpolate to get the smooth path
        smooth_path = []
        for idx in smooth_indices:
            smooth_path.append((int(x_interp(idx)), int(y_interp(idx))))
        
        return smooth_path
    except Exception as e:
        print(f"Error in smoothing path: {e}")
        return path_points  # Return original path if smoothing fails

def draw_path_with_history(frame, path, color, thickness, fade_effect=True):
    """
    Draw the path with a fading effect for older points
    
    Args:
        frame: The current video frame
        path: List of (x, y) tuples representing the path
        color: BGR color tuple for the path
        thickness: Base thickness of the path
        fade_effect: Whether to apply fading effect to older points
    """
    if len(path) < 2:
        return
    
    # Draw the path with fading effect
    if fade_effect:
        for i in range(1, len(path)):
            # Calculate alpha based on position in path (newer points are more opaque)
            alpha = min(1.0, (i / len(path)) * 1.5)
            
            # Adjust color opacity
            curr_color = (
                int(color[0] * alpha),
                int(color[1] * alpha),
                int(color[2] * alpha)
            )
            
            # Draw line segment
            cv2.line(frame, path[i-1], path[i], curr_color, thickness)
    else:
        # Just draw the polyline without fading
        pts = np.array(path, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], False, color, thickness=thickness)

def main():
    global target_id
    print("Starting person tracking and path visualization...")
    print("------------------------------------------------------")
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Print instructions
    print(f"Input video: {args.input}")
    print(f"Output video: {args.output}")
    print("\nInstructions:")
    print("- Person with ID 1 will be automatically tracked.")
    print("- Press 'p' to pause/resume")
    print("- Press 'q' to quit")
    print("------------------------------------------------------")
    
    # Parameters
    video_path = args.input
    output_path = args.output
    line_thickness = args.thickness
    line_color = get_color(args.color)
    debugging = args.debug
    fade_effect = args.fade_effect
    max_debug_frames = int(10 * 30)  
    # Initialize the YOLO model for person detection
    model = YOLO(args.model)  # Using YOLOv12 model specified in arguments
    
    # Initialize the SORT tracker
    tracker = Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_threshold)
    
    # Load the video
    cap = load_video(video_path)
    if cap is None:
        return
    
    # Initialize the video writer
    out, (width, height), fps = initialize_video_writer(cap, output_path)
    
    # Dictionary to store paths for each tracked object
    paths = {}
    
    # Initialize the tracker with a specific person ID to track
    target_id = 1
    
    # Frame counter and timing variables
    frame_count = 0
    
    print(f"Press 'q' to quit, 'p' to pause/resume. Tracking person with ID: {target_id}")
    
    paused = False
    
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Convert frame to RGB for YOLO (it expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run YOLO detection
            results = model(rgb_frame, classes=0)  # Class 0 is person in COCO dataset
            
            # Extract detections for SORT tracker (format: [x1, y1, x2, y2, confidence])
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    detections.append([x1, y1, x2, y2, conf])
            
            # Update tracker with new detections
            if detections:
                track_bbs_ids = tracker.update(np.array(detections))
            else:
                track_bbs_ids = np.empty((0, 5))
            
            # Display and process tracking results
            for track in track_bbs_ids:
                x1, y1, x2, y2, track_id = track
                track_id = int(track_id)
                
                # Calculate centroid of the bounding box
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)
                
                # If this is a newly detected object or we're tracking it
                if track_id not in paths:
                    paths[track_id] = []
                
                # Add the current position to the path
                paths[track_id].append((centroid_x, centroid_y))
                
                # Draw bounding box for all tracked persons
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                if target_id is not None and track_id == target_id:
                    # Highlight the selected person
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), line_color, 3)
                    
                    # Smooth the path for visualization
                    if len(paths[track_id]) > 2:
                        smooth_points = smooth_path(paths[track_id])
                        draw_path_with_history(frame, smooth_points, line_color, line_thickness, fade_effect=fade_effect)
            
            
            # Add tracking status
            if target_id is not None:
                status = f"Tracking person ID: {target_id}"
            else:
                status = "No person selected"
                    
            cv2.putText(frame, status, (width - 300, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Normal processing
            cv2.imshow('Tracking', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
            
            # Write the frame to the output video
            out.write(frame)
            
            # If debugging, break after processing the debug clip
            if debugging and frame_count >= max_debug_frames:
                print(f"Finished processing {max_debug_frames} frames (debug mode). Exiting.")
                break
        else:
            # When paused, just show the current frame and check for key presses
            cv2.imshow('Tracking (Paused)', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Tracking complete! Output saved to {output_path}")

if __name__ == "__main__":
    main()
