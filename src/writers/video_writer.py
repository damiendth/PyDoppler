import cv2
import numpy as np

def write_video(
    frames: np.ndarray, 
    output_file: str, 
    fps: int = 30, 
    format: str = 'avi'
) -> None:
    """
    Write greyscale frames to video file with AVI (default) or MP4 format.
    
    Parameters:
    - frames: 3D numpy array (number of frames, height, width) - greyscale
    - output_file: Output file path
    - fps: Frame rate (default: 30)
    - format: 'avi' (default) or 'mp4'
    """

    # Validate format
    if format not in ['avi', 'mp4']:
        raise ValueError("Format must be 'avi' or 'mp4'")
    
    num_frames, height, width = frames.shape
    
    # Convert to uint8 if needed
    if frames.dtype != np.uint8:
        if np.iscomplexobj(frames):
            frames = np.abs(frames)
        frames = cv2.normalize(frames, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # type: ignore
    
    # Set codec based on format
    codec = 'XVID' if format == 'avi' else 'mp4v'
    fourcc = cv2.VideoWriter_fourcc(*codec) # type: ignore
    
    # Ensure correct file extension
    if not output_file.endswith(f'.{format}'):
        output_file = f"{output_file}.{format}"
    
    # Create video writer and write frames
    video_writer = cv2.VideoWriter("output/" + output_file, fourcc, fps, (width, height))
    
    for i in range(num_frames):
        frame_bgr = cv2.cvtColor(frames[i], cv2.COLOR_GRAY2BGR)
        video_writer.write(frame_bgr)
    
    video_writer.release()
    print(f"Video saved: output/{output_file} ({format}, {num_frames} frames, {fps} FPS)")
