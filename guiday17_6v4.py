import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import joblib
import os
from PIL import Image, ImageTk
from feature_extractor import CoalFeatureExtractor

class CoalClassificationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Coal Classification System - Hệ thống phân loại mức than")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Biến lưu trữ
        self.current_image = None
        self.current_path = None
        self.roi_points = np.array([[663, 839], [1201, 661], [1717, 663], [1785, 1021]])
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.running_video = False
        self.level_var = tk.StringVar(value="0.0")
        self.bg_status = "close"  # close hoặc open
        
        # Load model
        self.load_models()
        
        # Tạo giao diện
        self.create_widgets()
        
    def load_models(self):
        """Load các model đã train"""
        try:
            self.model = joblib.load('coal_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            try:
                self.label_encoder = joblib.load('label_encoder.pkl')
            except:
                self.label_encoder = None
            print("Models loaded successfully!")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể load model: {str(e)}")
            
    def create_widgets(self):
        """Tạo các widget cho giao diện"""
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0')
        style.configure('Info.TLabel', font=('Arial', 10), background='#f0f0f0')
        style.configure('Result.TLabel', font=('Arial', 12, 'bold'), background='#e8f4fd')
        
        # Main title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill='x', pady=(0, 10))
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="HỆ THỐNG PHÂN LOẠI MỨC THAN", 
                              font=('Arial', 18, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(pady=15)
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left panel: Controls + Result
        left_frame = tk.Frame(main_frame, bg='#f0f0f0', width=350)
        left_frame.pack(side='left', fill='y', padx=(0, 10))
        left_frame.pack_propagate(False)
        
        # File selection
        file_frame = tk.LabelFrame(left_frame, text="Chọn tệp", font=('Arial', 12, 'bold'), 
                                  bg='#f0f0f0', fg='#2c3e50')
        file_frame.pack(fill='x', pady=(0, 10))
        
        self.browse_btn = tk.Button(file_frame, text="📁 Browse Image/Video", 
                                   command=self.browse_file, font=('Arial', 11, 'bold'),
                                   bg='#3498db', fg='white', relief='flat', pady=8)
        self.browse_btn.pack(fill='x', padx=10, pady=10)
        
        self.file_label = tk.Label(file_frame, text="Chưa chọn tệp", 
                                  font=('Arial', 9), bg='#f0f0f0', fg='#7f8c8d')
        self.file_label.pack(padx=10, pady=(0, 10))
        
        # Process button
        self.process_btn = tk.Button(file_frame, text="🔍 Phân tích và Dự đoán", 
                                    command=self.process_file, font=('Arial', 11, 'bold'),
                                    bg='#e74c3c', fg='white', relief='flat', pady=8,
                                    state='disabled')
        self.process_btn.pack(fill='x', padx=10, pady=(0, 10))
        
        # Prediction results
        result_frame = tk.LabelFrame(left_frame, text="Prediction Results", font=('Arial', 12, 'bold'),
                                    bg='#f0f0f0', fg='#2c3e50')
        result_frame.pack(fill='x', expand=False, padx=5)

        # Title with icon
        title_frame = tk.Frame(result_frame, bg='#f0f0f0')
        title_frame.pack(fill='x', padx=10, pady=(5,0))

        tk.Label(title_frame, text="🎯 Classification Confidence", 
                font=('Arial', 11, 'bold'), bg='#f0f0f0', fg='#2c3e50').pack(side='left')

        # Container for result bars
        self.result_container = tk.Frame(result_frame, bg='#f0f0f0')
        self.result_container.pack(fill='both', expand=True, padx=10, pady=5)

        # Create styled result bars for each class
        self.class_bars = {}
        self.class_labels = {}
        self.class_values = {}
        
        for class_name in ['Empty', 'Low', 'Medium', 'Full']:
            # Container for each class
            class_frame = tk.Frame(self.result_container, bg='#f0f0f0')
            class_frame.pack(fill='x', pady=2)
            
            # Label (left)
            label = tk.Label(class_frame, text=class_name, width=8, anchor='w',
                           font=('Arial', 10), bg='#f0f0f0', fg='#2c3e50')
            label.pack(side='left')
            
            # Progress bar (center)
            style_name = f'{class_name.lower()}.Horizontal.TProgressbar'
            style = ttk.Style()
            
            # Custom colors for each class
            colors = {
                'Empty': '#e74c3c',    # Red
                'Low': '#f39c12',      # Orange
                'Medium': '#3498db',    # Blue
                'Full': '#2ecc71'      # Green
            }
            
            style.configure(style_name,
                          troughcolor='#f5f5f5',
                          background=colors[class_name],
                          thickness=12,
                          borderwidth=0)
            
            bar = ttk.Progressbar(class_frame, style=style_name,
                                length=150, mode='determinate',
                                maximum=100, value=0)
            bar.pack(side='left', padx=5)
            
            # Value label (right)
            value_label = tk.Label(class_frame, text="0%", width=5,
                                 font=('Arial', 10), bg='#f0f0f0', fg=colors[class_name])
            value_label.pack(side='left')
            
            # Store references
            self.class_bars[class_name] = bar
            self.class_labels[class_name] = label
            self.class_values[class_name] = value_label

        # Add a separator
        ttk.Separator(result_frame, orient='horizontal').pack(fill='x', padx=10, pady=5)

        # Final prediction frame
        final_pred_frame = tk.Frame(result_frame, bg='#f0f0f0')
        final_pred_frame.pack(fill='x', padx=10, pady=5)

        tk.Label(final_pred_frame, text="Final Prediction:", 
                font=('Arial', 10, 'bold'), bg='#f0f0f0', fg='#2c3e50').pack(side='left')

        self.final_pred_label = tk.Label(final_pred_frame, text="--",
                                       font=('Arial', 12, 'bold'), bg='#f0f0f0', fg='#27ae60',
                                       width=10)
        self.final_pred_label.pack(side='left', padx=5)

        # Thêm frame hiển thị ảnh ngay dưới result_frame
        image_frame = tk.LabelFrame(left_frame, text="Ảnh/Frame và ROI", font=('Arial', 12, 'bold'),
                                    bg='#f0f0f0', fg='#2c3e50')
        image_frame.pack(fill='both', expand=True, padx=(0, 0), pady=(10, 0))

        # Frame chứa ảnh và progress bar dọc
        self.image_canvas_frame = tk.Frame(image_frame, bg='#f0f0f0')
        self.image_canvas_frame.pack(fill='both', expand=True)

        self.image_canvas = tk.Canvas(self.image_canvas_frame, bg='white', height=300, width=300)
        self.image_canvas.pack(side='left', fill='both', expand=True, padx=(0, 0), pady=0)

        # Style cho progress bar dọc rộng hơn
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('green.Vertical.TProgressbar', troughcolor='#e0e0e0', background='#00cc44', thickness=18)

        self.fullness_var = tk.DoubleVar(value=0)
        self.fullness_bar = ttk.Progressbar(
            self.image_canvas_frame,
            orient='vertical',
            length=300,
            mode='determinate',
            variable=self.fullness_var,
            maximum=100,
            style='green.Vertical.TProgressbar'
        )
        self.fullness_bar.pack(side='left', fill='y', padx=(5, 0), pady=0)

        # Nhãn phần trăm nằm trên thanh bar
        self.fullness_label = tk.Label(self.image_canvas_frame, text="0%", font=('Arial', 10, 'bold'), bg='#f0f0f0', fg='#00cc44')
        self.fullness_label.place(relx=0.98, rely=0.02, anchor='ne')  # Đặt sát trên bên phải
        
        # Right panel: Notebook with tabs
        right_frame = tk.Frame(main_frame, bg='#f0f0f0')
        right_frame.pack(side='right', fill='both', expand=True)
        
        # Create notebook
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # Create tabs
        self.background_tab = ttk.Frame(self.notebook)
        self.visualization_tab = ttk.Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.background_tab, text='Background')
        self.notebook.add(self.visualization_tab, text='3D Visualization')
        
        # Top: WinCC controls
        control_frame = tk.Frame(self.background_tab, bg='#f0f0f0')
        control_frame.pack(fill='x', pady=(0, 10))

        # Control Panel
        control_panel = tk.LabelFrame(control_frame, text="Control Panel", font=('Arial', 12, 'bold'),
                                    bg='#eaf6fb', fg='#2c3e50', padx=15, pady=10)
        control_panel.pack(fill='x', padx=10)

        # Left side: Motor controls
        motor_frame = tk.Frame(control_panel, bg='#eaf6fb')
        motor_frame.pack(side='left', padx=(0, 20))

        # Motor Title with icon
        motor_title = tk.Label(motor_frame, text="🔄 Motor Control", 
                             font=('Arial', 11, 'bold'), bg='#eaf6fb', fg='#2c3e50')
        motor_title.pack(pady=(0, 5))

        # Motor inputs with better layout
        motor_inputs = [
            ("Conveyor Motor (Hz)", "conveyor_speed"),
            ("Gate Motor (Hz)", "gate_speed"),
            ("Coal Level (%)", "coal_level")
        ]

        for label_text, name in motor_inputs:
            input_frame = tk.Frame(motor_frame, bg='#eaf6fb')
            input_frame.pack(fill='x', pady=2)
            
            label = tk.Label(input_frame, text=label_text, width=15, anchor='e',
                           font=('Arial', 10), bg='#eaf6fb', fg='#2c3e50')
            label.pack(side='left', padx=(0, 5))
            
            entry = tk.Entry(input_frame, width=8, font=('Arial', 10),
                           justify='center', relief='solid', bd=1)
            entry.pack(side='left')
            setattr(self, f"{name}_entry", entry)

        # Center: Mode and System controls
        mode_sys_frame = tk.Frame(control_panel, bg='#eaf6fb')
        mode_sys_frame.pack(side='left', padx=20)

        # Mode controls
        mode_title = tk.Label(mode_sys_frame, text="⚙️ Operation Mode", 
                            font=('Arial', 11, 'bold'), bg='#eaf6fb', fg='#2c3e50')
        mode_title.pack(pady=(0, 5))

        mode_buttons_frame = tk.Frame(mode_sys_frame, bg='#eaf6fb')
        mode_buttons_frame.pack()

        self.man_btn = tk.Button(mode_buttons_frame, text="MANUAL", width=8,
                                font=('Arial', 10, 'bold'), bg='#4CAF50', fg='white',
                                relief='raised', command=lambda: self.switch_mode('man'))
        self.man_btn.pack(side='left', padx=5)

        self.auto_btn = tk.Button(mode_buttons_frame, text="AUTO", width=8,
                                 font=('Arial', 10, 'bold'), bg='#9E9E9E', fg='white',
                                 relief='raised', command=lambda: self.switch_mode('auto'))
        self.auto_btn.pack(side='left', padx=5)

        # System controls
        sys_title = tk.Label(mode_sys_frame, text="🔌 System Control", 
                           font=('Arial', 11, 'bold'), bg='#eaf6fb', fg='#2c3e50')
        sys_title.pack(pady=(15, 5))

        sys_buttons_frame = tk.Frame(mode_sys_frame, bg='#eaf6fb')
        sys_buttons_frame.pack()

        self.start_btn = tk.Button(sys_buttons_frame, text="START", width=8,
                                 font=('Arial', 10, 'bold'), bg='#2196F3', fg='white',
                                 relief='raised', command=self.start_system)
        self.start_btn.pack(side='left', padx=5)

        self.stop_btn = tk.Button(sys_buttons_frame, text="STOP", width=8,
                                font=('Arial', 10, 'bold'), bg='#F44336', fg='white',
                                relief='raised', command=self.stop_system)
        self.stop_btn.pack(side='left', padx=5)

        # Right side: Coal Level Display
        level_frame = tk.Frame(control_panel, bg='#eaf6fb')
        level_frame.pack(side='right', padx=20)

        level_title = tk.Label(level_frame, text="📊 Coal Level", 
                             font=('Arial', 11, 'bold'), bg='#eaf6fb', fg='#2c3e50')
        level_title.pack(pady=(0, 5))

        self.level_display = tk.Label(level_frame, textvariable=self.level_var,
                                    font=('Arial', 24, 'bold'), bg='#eaf6fb', fg='#2c3e50',
                                    width=6, relief='solid', bd=1)
        self.level_display.pack()

        tk.Label(level_frame, text="Percent (%)", 
                font=('Arial', 10), bg='#eaf6fb', fg='#2c3e50').pack(pady=(5, 0))

        # Create main display area in background tab
        self.main_display = tk.Canvas(self.background_tab, bg='white')
        self.main_display.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create placeholder for 3D visualization tab
        self.visualization_placeholder = tk.Label(self.visualization_tab, 
                                                text="3D Visualization Coming Soon",
                                                font=('Arial', 14))
        self.visualization_placeholder.pack(expand=True)
        
        # Hiển thị ảnh nền lên canvas của tab Background
        try:
            bg_image = Image.open("close.png")
            # Đợi canvas render xong để lấy kích thước thực tế
            self.root.update_idletasks()
            canvas_width = self.main_display.winfo_width()
            canvas_height = self.main_display.winfo_height()
            if canvas_width < 10 or canvas_height < 10:
                canvas_width, canvas_height = 1000, 600  # fallback mặc định
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.ANTIALIAS
            bg_image = bg_image.resize((canvas_width, canvas_height), resample)
            self.bg_photo = ImageTk.PhotoImage(bg_image)
            self.main_display.create_image(0, 0, anchor='nw', image=self.bg_photo)
        except Exception as e:
            print(f"Lỗi khi load ảnh nền: {e}")
        
        self.main_display.bind("<Configure>", self.draw_bg_image)
        self.draw_bg_image()  # Vẽ lần đầu
        
    def draw_bg_image(self, event=None):
        try:
            bg_file = "close.png" if self.bg_status == "close" else "open.png"
            bg_image = Image.open(bg_file)
            canvas_width = self.main_display.winfo_width()
            canvas_height = self.main_display.winfo_height()
            if canvas_width < 10 or canvas_height < 10:
                return  # canvas chưa sẵn sàng
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.ANTIALIAS
            bg_image = bg_image.resize((canvas_width, canvas_height), resample)
            self.bg_photo = ImageTk.PhotoImage(bg_image)
            self.main_display.delete("all")
            self.main_display.create_image(0, 0, anchor='nw', image=self.bg_photo)
        except Exception as e:
            print(f"Lỗi khi load ảnh nền: {e}")
        
    def browse_file(self):
        """Chọn file ảnh hoặc video"""
        self.running_video = False
        file_types = [
            ('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff'),
            ('Video files', '*.mp4 *.avi *.mov *.mkv'),
            ('All files', '*.*')
        ]
        
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh hoặc video",
            filetypes=file_types
        )
        
        if file_path:
            self.current_path = file_path
            filename = os.path.basename(file_path)
            self.file_label.config(text=f"Đã chọn: {filename}")
            self.process_btn.config(state='normal')
            
            # Load và hiển thị ảnh preview
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                self.load_image_preview(file_path)
            
    def load_image_preview(self, file_path):
        """Load và hiển thị preview ảnh"""
        try:
            # Load ảnh
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Lỗi", "Không thể đọc ảnh!")
                return
                
            self.current_image = image
            
            # Resize ảnh để hiển thị
            height, width = image.shape[:2]
            max_size = 400
            
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                display_image = cv2.resize(image, (new_width, new_height))
            else:
                display_image = image.copy()
            
            # Chuyển đổi sang RGB và PIL
            display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(display_image)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Hiển thị trên canvas
            self.image_canvas.delete("all")
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                x = (canvas_width - pil_image.width) // 2
                y = (canvas_height - pil_image.height) // 2
                self.image_canvas.create_image(x, y, anchor='nw', image=photo)
                self.image_canvas.image = photo  # Keep reference
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi load ảnh: {str(e)}")
            
    def process_file(self):
        """Xử lý file và thực hiện dự đoán"""
        self.running_video = False
        if not self.current_path:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn file trước!")
            return
            
        if not self.model:
            messagebox.showerror("Lỗi", "Model chưa được load!")
            return
            
        try:
            if self.current_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                self.process_image()
            elif self.current_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                self.process_video()
            else:
                messagebox.showerror("Lỗi", "Định dạng file không được hỗ trợ!")
                
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi xử lý: {str(e)}")
            
    def process_video(self):
        """Xử lý video theo thời gian thực"""
        try:
            self.running_video = True
            cap = cv2.VideoCapture(self.current_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_idx = 0
            step = int(fps)  # Phân tích 1 frame mỗi giây
            
            extractor = CoalFeatureExtractor()
            
            while cap.isOpened() and self.running_video:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_idx % step == 0:
                    # Xử lý frame giống như xử lý ảnh
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    mask = np.zeros(gray_frame.shape, dtype=np.uint8)
                    cv2.fillPoly(mask, [self.roi_points], 255)
                    
                    features = extractor.extract_ml_features(gray_frame, mask)
                    feature_vector = extractor.get_feature_vector(features)
                    feature_vector_scaled = self.scaler.transform([feature_vector])
                    
                    prediction = self.model.predict(feature_vector_scaled)
                    probabilities = self.model.predict_proba(feature_vector_scaled)[0]
                    
                    # Hiển thị kết quả và visualization giống như xử lý ảnh
                    self.display_prediction_results(prediction[0], probabilities)
                    self.create_visualizations(frame, gray_frame, mask)
                    
                    # Hiển thị frame với ROI và scale đúng tỷ lệ
                    self.show_image_with_roi(frame, self.roi_points, self.image_canvas)
                
                # Cập nhật giao diện
                self.root.update()
                
                # Thêm delay nhỏ để video không chạy quá nhanh
                self.root.after(10)  # 10ms delay
                
                frame_idx += 1
                
            cap.release()
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi xử lý video: {str(e)}")
            
    def process_image(self):
        """Xử lý ảnh và dự đoán"""
        try:
            # Load ảnh
            image = cv2.imread(self.current_path)
            if image is None:
                messagebox.showerror("Lỗi", "Không thể đọc ảnh!")
                return
                
            # Chuyển sang grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Trích xuất features
            extractor = CoalFeatureExtractor()
            mask = np.zeros(gray_image.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [self.roi_points], 255)
            
            features = extractor.extract_ml_features(gray_image, mask)
            feature_vector = extractor.get_feature_vector(features)
            
            # Dự đoán
            feature_vector_scaled = self.scaler.transform([feature_vector])
            prediction = self.model.predict(feature_vector_scaled)
            probabilities = self.model.predict_proba(feature_vector_scaled)[0]
            
            # Hiển thị kết quả và visualization
            self.display_prediction_results(prediction[0], probabilities)
            self.create_visualizations(image, gray_image, mask)
            
            # Hiển thị ảnh với ROI
            self.show_image_with_roi(image, self.roi_points, self.image_canvas)
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi xử lý ảnh: {str(e)}")
            
    def display_prediction_results(self, prediction, probabilities):
        """Hiển thị kết quả dự đoán"""
        # Đảm bảo đúng thứ tự: Empty, Low, Medium, Full (Empty là 'rong')
        class_mapping = {
            'rong': 'Empty',
            'low': 'Low',
            'medium': 'Medium',
            'full': 'Full'
        }
        
        if self.label_encoder:
            class_names = list(self.label_encoder.classes_)
        else:
            class_names = ['rong', 'low', 'medium', 'full']

        # Tạo dict: tên lớp -> xác suất
        prob_dict = dict(zip(class_names, probabilities))

        # Tính toán fullness_percent dựa trên xác suất của từng mức
        level_map = {'rong': 0, 'low': 33, 'medium': 66, 'full': 100}
        fullness_percent = 0
        max_prob = 0
        final_prediction = "Unknown"

        for orig_name, display_name in class_mapping.items():
            prob = prob_dict.get(orig_name, 0.0)
            percentage = prob * 100
            
            # Cập nhật progress bar và label
            self.class_bars[display_name]['value'] = percentage
            self.class_values[display_name].config(text=f"{percentage:.1f}%")
            
            # Tính toán fullness
            fullness_percent += prob * level_map[orig_name]
            
            # Tìm prediction với probability cao nhất
            if prob > max_prob:
                max_prob = prob
                final_prediction = display_name

        # Cập nhật final prediction
        self.final_pred_label.config(text=final_prediction)

        # Cập nhật thanh bar và label tổng thể
        self.fullness_var.set(fullness_percent)
        self.fullness_label.config(text=f"{fullness_percent:.1f}%")
        self.level_var.set(f"{fullness_percent:.1f}")

        # Đổi background nếu cần
        prev_status = self.bg_status
        if fullness_percent < 5:
            self.bg_status = "close"
        else:
            self.bg_status = "open"
        if self.bg_status != prev_status:
            self.draw_bg_image()

    def create_visualizations(self, image, gray_image, mask):
        """Tạo các visualization"""
        try:
            # Clear previous plots
            self.fig.clear()
            
    #         # ROI processing
    #         roi = cv2.bitwise_and(image, image, mask=mask)
    #         gray_roi = cv2.bitwise_and(gray_image, gray_image, mask=mask)
            
    #         # Get coordinates and values
    #         y, x = np.where(mask == 255)
    #         z = gray_roi[y, x]
            
    #         # Create grid for interpolation
    #         if len(x) > 0 and len(y) > 0:
    #             grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    #             try:
    #                 grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
    #             except:
    #                 grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
            
    #         # Create subplots
    #         ax1 = self.fig.add_subplot(221)
    #         ax2 = self.fig.add_subplot(222)
    #         ax3 = self.fig.add_subplot(223, projection='3d')
    #         ax4 = self.fig.add_subplot(224, projection='3d')
            
            # # Original image
            # ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # ax1.set_title('Ảnh gốc', fontsize=12, fontweight='bold')
            # ax1.axis('off')
            
            # # ROI
            # ax2.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            # ax2.set_title('Vùng quan tâm (ROI)', fontsize=12, fontweight='bold')
            # ax2.axis('off')
            
            # # 3D Scatter
            # if len(x) > 0:
            #     scatter = ax3.scatter(x[::10], y[::10], z[::10], c=z[::10], cmap='viridis', alpha=0.6)
            #     ax3.set_title('Phân tích 3D - Scatter', fontsize=12, fontweight='bold')
            #     ax3.set_xlabel('X')
            #     ax3.set_ylabel('Y')
            #     ax3.set_zlabel('Độ xám')
                
            #     # 3D Surface
            #     if 'grid_z' in locals() and grid_z is not None:
            #         ax4.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', alpha=0.8)
            #         ax4.set_title('Bề mặt nội suy 3D', fontsize=12, fontweight='bold')
            #         ax4.set_xlabel('X')
            #         ax4.set_ylabel('Y')
            #         ax4.set_zlabel('Giá trị nội suy')
                    
        #     plt.tight_layout()
        #     self.canvas.draw()
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            
    def show_image_with_roi(self, image, roi_points, canvas, max_size=300):
        """
        Hiển thị ảnh với ROI lên canvas, scale đúng tỷ lệ, không méo ảnh.
        image: ảnh gốc (numpy array)
        roi_points: np.array([[x1, y1], ...]) gốc
        canvas: canvas Tkinter để hiển thị
        max_size: kích thước tối đa (chiều rộng hoặc cao)
        """
        # Tính tỉ lệ scale
        h, w = image.shape[:2]
        scale = min(max_size / w, max_size / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Scale ROI
        roi_scaled = (roi_points * scale).astype(np.int32)

        # Vẽ ROI lên ảnh
        cv2.polylines(image_resized, [roi_scaled], True, (0, 255, 0), 2)

        # Chuyển sang RGB và PIL
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        photo = ImageTk.PhotoImage(pil_image)

        # Hiển thị lên canvas (giữa canvas)
        canvas.config(width=new_w, height=new_h)
        self.fullness_bar.config(length=new_h)
        canvas.delete("all")
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()
        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2
        canvas.create_image(x, y, anchor='nw', image=photo)
        canvas.image = photo  # giữ tham chiếu

    def switch_mode(self, mode):
        """Chuyển đổi giữa chế độ MAN và AUTO"""
        if mode == 'man':
            self.man_btn.config(bg='#4CAF50')  # Active color
            self.auto_btn.config(bg='#9E9E9E')  # Inactive color
            # Enable manual controls
            self.conveyor_speed_entry.config(state='normal')
            self.gate_speed_entry.config(state='normal')
            self.coal_level_entry.config(state='normal')
        else:
            self.auto_btn.config(bg='#4CAF50')  # Active color
            self.man_btn.config(bg='#9E9E9E')  # Inactive color
            # Disable manual controls in auto mode
            self.conveyor_speed_entry.config(state='disabled')
            self.gate_speed_entry.config(state='disabled')
            self.coal_level_entry.config(state='disabled')

    def start_system(self):
        """Bắt đầu hệ thống"""
        self.start_btn.config(state='disabled', bg='#9E9E9E')
        self.stop_btn.config(state='normal', bg='#F44336')
        # Thêm code xử lý start system ở đây
        messagebox.showinfo("System Status", "System started successfully!")

    def stop_system(self):
        """Dừng hệ thống"""
        self.stop_btn.config(state='disabled', bg='#9E9E9E')
        self.start_btn.config(state='normal', bg='#2196F3')
        # Thêm code xử lý stop system ở đây
        messagebox.showinfo("System Status", "System stopped successfully!")

if __name__ == "__main__":
    root = tk.Tk()
    app = CoalClassificationGUI(root)
    root.mainloop()
