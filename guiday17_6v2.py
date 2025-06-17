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
        result_frame = tk.LabelFrame(left_frame, text="Kết quả dự đoán", font=('Arial', 12, 'bold'),
                                    bg='#f0f0f0', fg='#2c3e50')
        result_frame.pack(fill='x', expand=False)
        
        self.result_text = tk.Text(result_frame, height=10, width=35, font=('Arial', 10),
                                  bg='#e8f4fd', fg='#2c3e50', relief='flat', wrap='word')
        self.result_text.pack(padx=10, pady=10, fill='both', expand=True)
        
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
        
        # Motor controls
        motor_frame = tk.LabelFrame(control_frame, text="Motor", bg='#eaf6fb', fg='#2c3e50')
        motor_frame.pack(side='left', padx=10)
        tk.Label(motor_frame, text="Motor kéo tàu:").pack()
        tk.Entry(motor_frame, width=6).pack()
        tk.Label(motor_frame, text="Motor đóng mở:").pack()
        tk.Entry(motor_frame, width=6).pack()
        tk.Label(motor_frame, text="Chọn mức than:").pack()
        tk.Entry(motor_frame, width=6).pack()
        
        # Mode controls
        mode_frame = tk.LabelFrame(control_frame, text="Chế độ", bg='#eaf6fb', fg='#2c3e50')
        mode_frame.pack(side='left', padx=10)
        tk.Button(mode_frame, text="MAN", bg='lime').pack(side='left', padx=5)
        tk.Button(mode_frame, text="AUTO", bg='violet').pack(side='left', padx=5)
        
        # System controls
        sys_frame = tk.LabelFrame(control_frame, text="Hệ thống", bg='#eaf6fb', fg='#2c3e50')
        sys_frame.pack(side='left', padx=10)
        tk.Button(sys_frame, text="START", bg='lime').pack(side='left', padx=5)
        tk.Button(sys_frame, text="STOP", bg='red').pack(side='left', padx=5)
        
        # Mức than
        level_frame = tk.LabelFrame(control_frame, text="Mức than", bg='#eaf6fb', fg='#2c3e50')
        level_frame.pack(side='left', padx=10)
        self.level_var = tk.IntVar(value=0)
        tk.Label(level_frame, textvariable=self.level_var, font=('Arial', 18, 'bold')).pack()
        
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
            bg_image = Image.open("bg.png")
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
            bg_image = Image.open("bg.png")
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
        result_text = f"""🎯 KẾT QUẢ DỰ ĐOÁN
{'='*30}
📊 Mức than dự đoán: {prediction}

📈 XÁC SUẤT TỪNG LOẠI:
{'='*30}
"""
        
        # Đảm bảo đúng thứ tự label: rong, low, medium, full
        level_map = {'rong': 0, 'low': 33, 'medium': 66, 'full': 100}
        if self.label_encoder:
            level_values = np.array([level_map.get(lbl, 0) for lbl in self.label_encoder.classes_])
        else:
            level_values = np.array([0, 33, 66, 100])
        fullness_percent = float(np.dot(probabilities, level_values))
        
        self.fullness_var.set(fullness_percent)
        self.fullness_label.config(text=f"{fullness_percent:.1f}%")
        
        # Đổi màu progress bar sang xanh lá cây
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('green.Vertical.TProgressbar', troughcolor='#e0e0e0', background='#00cc44')
        self.fullness_bar.config(style='green.Vertical.TProgressbar')
        
        if self.label_encoder:
            class_names = self.label_encoder.classes_
        else:
            class_names = [f"Class_{i}" for i in range(len(probabilities))]
            
        for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
            bar_length = int(prob * 20)  # Thanh progress 20 ký tự
            bar = '█' * bar_length + '░' * (20 - bar_length)
            result_text += f"{class_name}: {prob:.4f} |{bar}| {prob*100:.1f}%\n"
            
        # Thêm thông tin chi tiết
        max_prob_idx = np.argmax(probabilities)
        confidence = probabilities[max_prob_idx]
        
        result_text += f"""
📊 CHI TIẾT:
{'='*30}
🔍 Độ tin cậy: {confidence:.4f} ({confidence*100:.1f}%)
🎯 Lớp có xác suất cao nhất: {class_names[max_prob_idx]}
📈 Entropy: {-np.sum(probabilities * np.log2(probabilities + 1e-10)):.4f}
"""
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, result_text)
        
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

if __name__ == "__main__":
    root = tk.Tk()
    app = CoalClassificationGUI(root)
    root.mainloop()
