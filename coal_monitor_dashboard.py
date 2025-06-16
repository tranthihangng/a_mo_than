import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime
from new_coal import extract_roi_features, visualize_analysis

class CoalMonitorDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Bảng Điều Khiển Giám Sát Than")
        self.root.state('zoomed')
        
        # ROI cố định theo kích thước gốc của ảnh
        self.original_roi_points = np.array([[663, 839], [1201, 661], [1717, 663], [1785, 1021]], np.int32)
        self.roi_points = self.original_roi_points.copy()
        
        # Khởi tạo biến
        self.current_image = None
        self.current_video = None
        self.is_video_playing = False
        self.monitoring_active = False
        self.original_image_size = None  # Lưu kích thước gốc của ảnh
        
        self.setup_gui()
        
    def setup_gui(self):
        # Tạo menu
        self.create_menu()
        
        # Tạo main container
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Image/Video display
        left_panel = ttk.Frame(main_container)
        main_container.add(left_panel, weight=2)
        
        # Right panel - Analysis and monitoring
        right_panel = ttk.Frame(main_container)
        main_container.add(right_panel, weight=1)
        
        self.setup_left_panel(left_panel)
        self.setup_right_panel(right_panel)
        
    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Mở ảnh", command=self.open_image)
        file_menu.add_command(label="Mở video", command=self.open_video)
        file_menu.add_separator()
        file_menu.add_command(label="Thoát", command=self.root.quit)
        
        # Monitoring menu
        monitor_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Giám sát", menu=monitor_menu)
        monitor_menu.add_command(label="Bắt đầu giám sát", command=self.start_monitoring)
        monitor_menu.add_command(label="Dừng giám sát", command=self.stop_monitoring)
        
    def setup_left_panel(self, parent):
        # Image/Video display area
        self.display_frame = ttk.LabelFrame(parent, text="Hiển thị")
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(self.display_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Video controls
        self.video_controls = ttk.Frame(parent)
        self.video_controls.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(self.video_controls, text="Play", command=self.play_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.video_controls, text="Pause", command=self.pause_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.video_controls, text="Stop", command=self.stop_video).pack(side=tk.LEFT, padx=5)
        
        # Initially hide video controls
        self.video_controls.pack_forget()
        
    def setup_right_panel(self, parent):
        # Create notebook for different views
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Real-time monitoring tab
        monitoring_frame = ttk.Frame(self.notebook)
        self.notebook.add(monitoring_frame, text="Giám sát thời gian thực")
        self.setup_monitoring(monitoring_frame)
        
        # Analysis tab
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="Phân tích chi tiết")
        self.setup_analysis(analysis_frame)
        
    def setup_monitoring(self, parent):
        # Create monitoring plots
        self.monitor_figure = plt.Figure(figsize=(8, 6), dpi=100)
        self.monitor_canvas = FigureCanvasTkAgg(self.monitor_figure, parent)
        self.monitor_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create subplots
        self.intensity_ax = self.monitor_figure.add_subplot(211)
        self.ratio_ax = self.monitor_figure.add_subplot(212)
        
        # Initialize data storage
        self.times = []
        self.intensities = []
        self.dark_ratios = []
        self.bright_ratios = []
        
        # Add monitoring controls
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Bắt đầu giám sát", 
                  command=self.start_monitoring).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Dừng giám sát", 
                  command=self.stop_monitoring).pack(side=tk.LEFT, padx=5)
        
    def setup_analysis(self, parent):
        # Create scrollable frame for analysis results
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add analysis labels
        self.analysis_labels = {}
        analysis_items = [
            "Thời gian phân tích",
            "Độ sáng trung bình",
            "Độ lệch chuẩn",
            "Hệ số biến thiên",
            "Tỷ lệ than",
            "Tỷ lệ tạp chất",
            "Độ đồng nhất",
            "Độ tương phản",
            "Độ tương quan",
            "Đánh giá chất lượng"
        ]
        
        for item in analysis_items:
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(frame, text=f"{item}:").pack(side=tk.LEFT)
            self.analysis_labels[item] = ttk.Label(frame, text="--")
            self.analysis_labels[item].pack(side=tk.RIGHT)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.current_image = cv2.imread(file_path)
            self.display_image(self.current_image)
            self.video_controls.pack_forget()
            self.analyze_current_frame()
            
    def open_video(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if file_path:
            self.current_video = cv2.VideoCapture(file_path)
            self.video_controls.pack()
            self.play_video()
            
    def display_image(self, image):
        if image is None:
            return
            
        # Lưu kích thước gốc của ảnh
        self.original_image_size = image.shape[:2]
            
        # Convert to RGB for display
        if len(image.shape) == 3:
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            display_image = image
            
        # Resize image to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # Tính toán tỷ lệ scale
            height, width = display_image.shape[:2]
            scale = min(canvas_width/width, canvas_height/height)
            
            # Scale ROI points
            self.roi_points = (self.original_roi_points * scale).astype(np.int32)
            
            # Resize image
            display_image = cv2.resize(display_image, (int(width * scale), int(height * scale)))
        
        # Draw ROI
        display_image = display_image.copy()
        cv2.polylines(display_image, [self.roi_points], True, (255, 0, 0), 2)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(display_image))
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
    def analyze_current_frame(self):
        if self.current_image is None:
            return
            
        # Tính toán tỷ lệ scale để chuyển ROI về kích thước gốc
        if self.original_image_size is not None:
            height, width = self.current_image.shape[:2]
            scale = min(self.original_image_size[0]/height, self.original_image_size[1]/width)
            original_roi = (self.roi_points / scale).astype(np.int32)
        else:
            original_roi = self.original_roi_points
            
        # Extract features using original ROI
        features, roi, gray_roi, mask = extract_roi_features(self.current_image, original_roi)
        
        # Update analysis labels
        self.analysis_labels["Thời gian phân tích"].config(
            text=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        self.analysis_labels["Độ sáng trung bình"].config(
            text=f"{features['mean_intensity']:.2f}"
        )
        self.analysis_labels["Độ lệch chuẩn"].config(
            text=f"{features['std_intensity']:.2f}"
        )
        self.analysis_labels["Hệ số biến thiên"].config(
            text=f"{features['coefficient_variation']:.3f}"
        )
        self.analysis_labels["Tỷ lệ than"].config(
            text=f"{features['dark_ratio']*100:.1f}%"
        )
        self.analysis_labels["Tỷ lệ tạp chất"].config(
            text=f"{features['bright_ratio']*100:.1f}%"
        )
        self.analysis_labels["Độ đồng nhất"].config(
            text=f"{features['avg_homogeneity']:.3f}"
        )
        self.analysis_labels["Độ tương phản"].config(
            text=f"{features['avg_contrast']:.3f}"
        )
        self.analysis_labels["Độ tương quan"].config(
            text=f"{features['avg_correlation']:.3f}"
        )
        
        # Update monitoring data
        current_time = time.time()
        self.times.append(current_time)
        self.intensities.append(features['mean_intensity'])
        self.dark_ratios.append(features['dark_ratio'])
        self.bright_ratios.append(features['bright_ratio'])
        
        # Keep only last 100 points
        if len(self.times) > 100:
            self.times = self.times[-100:]
            self.intensities = self.intensities[-100:]
            self.dark_ratios = self.dark_ratios[-100:]
            self.bright_ratios = self.bright_ratios[-100:]
            
        # Update plots
        self.update_monitoring_plots()
        
    def update_monitoring_plots(self):
        # Update intensity plot
        self.intensity_ax.clear()
        self.intensity_ax.plot(self.times, self.intensities, 'b-')
        self.intensity_ax.set_title("Độ sáng trung bình theo thời gian")
        self.intensity_ax.set_ylabel("Độ sáng")
        
        # Update ratio plot
        self.ratio_ax.clear()
        self.ratio_ax.plot(self.times, self.dark_ratios, 'k-', label='Than')
        self.ratio_ax.plot(self.times, self.bright_ratios, 'r-', label='Tạp chất')
        self.ratio_ax.set_title("Tỷ lệ than và tạp chất")
        self.ratio_ax.set_xlabel("Thời gian")
        self.ratio_ax.set_ylabel("Tỷ lệ")
        self.ratio_ax.legend()
        
        self.monitor_figure.tight_layout()
        self.monitor_canvas.draw()
        
    def play_video(self):
        if self.current_video is None:
            return
            
        self.is_video_playing = True
        threading.Thread(target=self._video_loop, daemon=True).start()
        
    def pause_video(self):
        self.is_video_playing = False
        
    def stop_video(self):
        self.is_video_playing = False
        if self.current_video is not None:
            self.current_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
    def _video_loop(self):
        while self.is_video_playing:
            ret, frame = self.current_video.read()
            if not ret:
                self.current_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            self.current_image = frame
            self.display_image(frame)
            self.analyze_current_frame()
            time.sleep(1/30)  # 30 FPS
            
    def start_monitoring(self):
        self.monitoring_active = True
        threading.Thread(target=self._monitoring_loop, daemon=True).start()
        
    def stop_monitoring(self):
        self.monitoring_active = False
        
    def _monitoring_loop(self):
        while self.monitoring_active:
            if self.current_image is not None:
                self.analyze_current_frame()
            time.sleep(1)  # Update every second

if __name__ == "__main__":
    root = tk.Tk()
    app = CoalMonitorDashboard(root)
    root.mainloop() 