from moviepy.editor import VideoFileClip
import os

def cut_video(input_path, output_path, start_minute, end_minute):
    # Convert minutes to seconds
    start_time = start_minute * 60
    end_time = end_minute * 60
    
    try:
        # Load the video
        video = VideoFileClip(input_path)
        
        # Cut the video
        cut_video = video.subclip(start_time, end_time)
        
        # Write the result to a file
        cut_video.write_videofile(output_path)
        
        # Close the video to free up resources
        video.close()
        cut_video.close()
        
        print(f"Video đã được cắt thành công và lưu tại: {output_path}")
        
    except Exception as e:
        print(f"Có lỗi xảy ra: {str(e)}")

if __name__ == "__main__":
    # Đường dẫn đến video đầu vào
    input_video = "F:\hiv00007.mp4"  # Thay đổi tên file input của bạn ở đây
    
    # Tạo tên file output
    output_video = "F:\output_cut7.mp4"
    
    # Cắt video từ phút 50 đến phút 54
    cut_video(input_video, output_video, 50, 54) 