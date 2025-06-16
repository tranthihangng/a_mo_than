import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageTk
import threading
import time
from coal import CoalFeatureExtractor
import joblib
from datetime import datetime

class CoalMonitorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống Giám sát và Phân tích Than")
        self.root.state('zoomed')  # Mở cửa sổ full màn hình
        
        # Khởi tạo các biến
        self.current_image = None
        self.current_video = None
        self.is_video_playing = False
        self.roi_points = None
        self.feature_extractor = CoalFeatureExtractor()
        
        # Load model và scaler
        try:
            self.model = joblib.load('coal_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.le = joblib.load('label_encoder.pkl')
        except:
            print("Không thể load model. Chạy ở chế độ demo.")
            self.model = None
        
        self.setup_gui()
        
    def setup_gui(self):
        # Tạo menu bar
        self.create_menu()
        
        # Tạo main container
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Image/Video display
        left_panel = ttk.Frame(main_container)
        main_container.add(left_panel, weight=2)
        
        # Right panel - Analysis and controls
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
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Phân tích", menu=analysis_menu)
        analysis_menu.add_command(label="Phân tích ROI", command=self.analyze_roi)
        analysis_menu.add_command(label="Vẽ biểu đồ", command=self.plot_analysis)
        
    def setup_left_panel(self, parent):
        # Image/Video display area
        self.display_frame = ttk.LabelFrame(parent, text="Hiển thị")
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(self.display_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # ROI drawing controls
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Vẽ ROI", command=self.start_roi_drawing).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Xóa ROI", command=self.clear_roi).pack(side=tk.LEFT, padx=5)
        
        # Video controls
        self.video_controls = ttk.Frame(parent)
        self.video_controls.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(self.video_controls, text="Play", command=self.play_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.video_controls, text="Pause", command=self.pause_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.video_controls, text="Stop", command=self.stop_video).pack(side=tk.LEFT, padx=5)
        
        # Initially hide video controls
        self.video_controls.pack_forget()
        
    def setup_right_panel(self, parent):
        # Analysis results
        analysis_frame = ttk.LabelFrame(parent, text="Kết quả phân tích")
        analysis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create notebook for different analysis views
        self.notebook = ttk.Notebook(analysis_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Basic info tab
        basic_info = ttk.Frame(self.notebook)
        self.notebook.add(basic_info, text="Thông tin cơ bản")
        self.setup_basic_info(basic_info)
        
        # Histogram tab
        histogram_frame = ttk.Frame(self.notebook)
        self.notebook.add(histogram_frame, text="Histogram")
        self.setup_histogram(histogram_frame)
        
        # 3D Analysis tab
        analysis_3d = ttk.Frame(self.notebook)
        self.notebook.add(analysis_3d, text="Phân tích 3D")
        self.setup_3d_analysis(analysis_3d)
        
        # Real-time monitoring tab
        monitoring = ttk.Frame(self.notebook)
        self.notebook.add(monitoring, text="Giám sát thời gian thực")
        self.setup_monitoring(monitoring)
        
    def setup_basic_info(self, parent):
        # Create scrollable frame for basic info
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add basic information labels
        self.info_labels = {}
        info_items = [
            "Thời gian phân tích",
            "Loại than",
            "Độ sáng trung bình",
            "Độ đồng nhất",
            "Tỷ lệ than",
            "Tỷ lệ tạp chất",
            "Độ phức tạp",
            "Độ tương phản"
        ]
        
        for item in info_items:
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(frame, text=f"{item}:").pack(side=tk.LEFT)
            self.info_labels[item] = ttk.Label(frame, text="--")
            self.info_labels[item].pack(side=tk.RIGHT)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def setup_histogram(self, parent):
        self.hist_figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.hist_canvas = FigureCanvasTkAgg(self.hist_figure, parent)
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def setup_3d_analysis(self, parent):
        self.analysis_3d_figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.analysis_3d_canvas = FigureCanvasTkAgg(self.analysis_3d_figure, parent)
        self.analysis_3d_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def setup_monitoring(self, parent):
        # Create real-time monitoring plot
        self.monitor_figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.monitor_canvas = FigureCanvasTkAgg(self.monitor_figure, parent)
        self.monitor_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add monitoring controls
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Bắt đầu giám sát", 
                  command=self.start_monitoring).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Dừng giám sát", 
                  command=self.stop_monitoring).pack(side=tk.LEFT, padx=5)
        
    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.current_image = cv2.imread(file_path)
            self.display_image(self.current_image)
            self.video_controls.pack_forget()
            
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
            
        # Convert to RGB for display
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:  # Ensure canvas is initialized
            image = self.resize_image(image, canvas_width, canvas_height)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(image))
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
    def resize_image(self, image, max_width, max_height):
        height, width = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(max_width/width, max_height/height)
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        return cv2.resize(image, (new_width, new_height))
        
    def start_roi_drawing(self):
        if self.current_image is None:
            return
            
        self.roi_points = []
        self.canvas.bind("<Button-1>", self.add_roi_point)
        self.canvas.bind("<Button-3>", self.finish_roi_drawing)
        
    def add_roi_point(self, event):
        if len(self.roi_points) < 4:
            self.roi_points.append((event.x, event.y))
            self.canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, 
                                  fill="red", outline="red")
            
            if len(self.roi_points) > 1:
                p1 = self.roi_points[-2]
                p2 = self.roi_points[-1]
                self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], 
                                      fill="red", width=2)
                
    def finish_roi_drawing(self, event):
        if len(self.roi_points) >= 3:
            # Close the polygon
            p1 = self.roi_points[-1]
            p2 = self.roi_points[0]
            self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], 
                                  fill="red", width=2)
            
            # Unbind events
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<Button-3>")
            
            # Analyze ROI
            self.analyze_roi()
            
    def clear_roi(self):
        self.roi_points = None
        self.canvas.delete("all")
        if self.current_image is not None:
            self.display_image(self.current_image)
            
    def analyze_roi(self):
        if self.current_image is None or self.roi_points is None:
            return
            
        # Convert ROI points to numpy array
        roi_points = np.array(self.roi_points)
        
        # Create mask
        mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [roi_points], 255)
        
        # Extract features
        features = self.feature_extractor.extract_ml_features(
            cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY),
            mask
        )
        
        # Update basic info
        self.update_basic_info(features)
        
        # Update histogram
        self.update_histogram(self.current_image, mask)
        
        # Update 3D analysis
        self.update_3d_analysis(features)
        
    def update_basic_info(self, features):
        # Update time
        self.info_labels["Thời gian phân tích"].config(
            text=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Update other features
        if self.model is not None:
            feature_vector = self.feature_extractor.get_feature_vector(features)
            feature_vector = self.scaler.transform([feature_vector])
            pred = self.model.predict(feature_vector)
            if self.le:
                pred = self.le.inverse_transform(pred)
            self.info_labels["Loại than"].config(text=str(pred[0]))
        
        # Update other metrics
        self.info_labels["Độ sáng trung bình"].config(
            text=f"{features['mean_intensity']:.2f}"
        )
        self.info_labels["Độ đồng nhất"].config(
            text=f"{features['std_intensity']:.2f}"
        )
        self.info_labels["Tỷ lệ than"].config(
            text=f"{features['dark_ratio']*100:.1f}%"
        )
        self.info_labels["Tỷ lệ tạp chất"].config(
            text=f"{features['bright_ratio']*100:.1f}%"
        )
        self.info_labels["Độ phức tạp"].config(
            text=f"{features['hist_entropy']:.2f}"
        )
        self.info_labels["Độ tương phản"].config(
            text=f"{features['glcm_contrast_mean']:.2f}"
        )
        
    def update_histogram(self, image, mask):
        self.hist_figure.clear()
        ax = self.hist_figure.add_subplot(111)
        
        # Get ROI pixels
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        roi_pixels = gray[mask == 255]
        
        # Plot histogram
        ax.hist(roi_pixels.ravel(), bins=256, range=(0, 256), 
                color='gray', alpha=0.7)
        ax.set_title("Histogram của vùng ROI")
        ax.set_xlabel("Độ sáng")
        ax.set_ylabel("Tần suất")
        
        self.hist_canvas.draw()
        
    def update_3d_analysis(self, features):
        self.analysis_3d_figure.clear()
        ax = self.analysis_3d_figure.add_subplot(111, projection='3d')
        
        # Select features for 3D plot
        x = features['mean_intensity']
        y = features['std_intensity']
        z = features['hist_entropy']
        
        # Create scatter plot
        ax.scatter(x, y, z, c='red', marker='o')
        
        # Set labels
        ax.set_xlabel('Độ sáng trung bình')
        ax.set_ylabel('Độ đồng nhất')
        ax.set_zlabel('Độ phức tạp')
        
        self.analysis_3d_canvas.draw()
        
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
                
            self.display_image(frame)
            time.sleep(1/30)  # 30 FPS
            
    def start_monitoring(self):
        # Start real-time monitoring
        self.monitoring_active = True
        threading.Thread(target=self._monitoring_loop, daemon=True).start()
        
    def stop_monitoring(self):
        self.monitoring_active = False
        
    def _monitoring_loop(self):
        self.monitor_figure.clear()
        ax = self.monitor_figure.add_subplot(111)
        
        times = []
        values = []
        
        while self.monitoring_active:
            if self.current_image is not None and self.roi_points is not None:
                # Get current time
                current_time = time.time()
                times.append(current_time)
                
                # Get current value (e.g., mean intensity)
                mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(self.roi_points)], 255)
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
                value = np.mean(gray[mask == 255])
                values.append(value)
                
                # Keep only last 100 points
                if len(times) > 100:
                    times = times[-100:]
                    values = values[-100:]
                
                # Update plot
                ax.clear()
                ax.plot(times, values)
                ax.set_title("Giám sát thời gian thực")
                ax.set_xlabel("Thời gian")
                ax.set_ylabel("Độ sáng trung bình")
                
                self.monitor_canvas.draw()
                
            time.sleep(1)  # Update every second

    def plot_analysis(self):
        """Vẽ các biểu đồ phân tích chi tiết"""
        if self.current_image is None or self.roi_points is None:
            return
            
        # Tạo cửa sổ mới cho biểu đồ
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Phân tích chi tiết")
        plot_window.geometry("1200x800")
        
        # Tạo figure với nhiều subplot
        fig = plt.Figure(figsize=(12, 8), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tạo mask cho ROI
        mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(self.roi_points)], 255)
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        roi_pixels = gray[mask == 255]
        
        # 1. Histogram với đường phân phối chuẩn
        ax1 = fig.add_subplot(221)
        ax1.hist(roi_pixels.ravel(), bins=50, density=True, alpha=0.7, color='gray')
        ax1.set_title("Histogram và phân phối chuẩn")
        ax1.set_xlabel("Độ sáng")
        ax1.set_ylabel("Tần suất")
        
        # Thêm đường phân phối chuẩn
        mean = np.mean(roi_pixels)
        std = np.std(roi_pixels)
        x = np.linspace(mean - 3*std, mean + 3*std, 100)
        p = np.exp(-(x - mean)**2 / (2*std**2)) / (std * np.sqrt(2*np.pi))
        ax1.plot(x, p, 'r-', lw=2)
        
        # 2. Box plot
        ax2 = fig.add_subplot(222)
        ax2.boxplot(roi_pixels)
        ax2.set_title("Box plot độ sáng")
        ax2.set_ylabel("Độ sáng")
        
        # 3. Scatter plot của các đặc trưng
        ax3 = fig.add_subplot(223)
        features = self.feature_extractor.extract_ml_features(gray, mask)
        x = features['mean_intensity']
        y = features['std_intensity']
        ax3.scatter(x, y, c='red', marker='o')
        ax3.set_title("Tương quan độ sáng và độ đồng nhất")
        ax3.set_xlabel("Độ sáng trung bình")
        ax3.set_ylabel("Độ đồng nhất")
        
        # 4. Bar plot của các tỷ lệ
        ax4 = fig.add_subplot(224)
        ratios = [
            features['dark_ratio'],
            features['medium_ratio'],
            features['bright_ratio']
        ]
        labels = ['Than', 'Trung bình', 'Tạp chất']
        ax4.bar(labels, ratios)
        ax4.set_title("Tỷ lệ các thành phần")
        ax4.set_ylabel("Tỷ lệ")
        
        # Điều chỉnh layout
        fig.tight_layout()
        canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = CoalMonitorGUI(root)
    root.mainloop() 