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
        
        # Left panel - Controls
        left_frame = tk.Frame(main_frame, bg='#f0f0f0', width=300)
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
        result_frame.pack(fill='both', expand=True)
        
        self.result_text = tk.Text(result_frame, height=10, width=35, font=('Arial', 10),
                                  bg='#e8f4fd', fg='#2c3e50', relief='flat', wrap='word')
        self.result_text.pack(padx=10, pady=10, fill='both', expand=True)
        
        # Right panel - Visualizations
        right_frame = tk.Frame(main_frame, bg='#f0f0f0')
        right_frame.pack(side='right', fill='both', expand=True)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # Create tabs
        self.create_image_tab()
        self.create_3d_tab()
        
    def create_image_tab(self):
        """Tạo tab hiển thị ảnh"""
        self.image_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.image_frame, text='📷 Ảnh gốc và ROI')
        
        # Canvas for images
        self.image_canvas = tk.Canvas(self.image_frame, bg='white')
        self.image_canvas.pack(fill='both', expand=True, padx=10, pady=10)
        
    def create_3d_tab(self):
        """Tạo tab hiển thị 3D"""
        self.viz_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.viz_frame, text='📊 Phân tích 3D')
        
        # Matplotlib figure
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.patch.set_facecolor('white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.viz_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
    def browse_file(self):
        """Chọn file ảnh hoặc video"""
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
            cap = cv2.VideoCapture(self.current_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_idx = 0
            step = int(fps)  # Phân tích 1 frame mỗi giây
            
            extractor = CoalFeatureExtractor()
            
            while cap.isOpened():
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
                    
                    # Hiển thị frame với ROI
                    display_frame = frame.copy()
                    cv2.polylines(display_frame, [self.roi_points], True, (0, 255, 0), 2)
                    
                    # Resize frame để hiển thị
                    height, width = display_frame.shape[:2]
                    max_size = 800
                    if max(height, width) > max_size:
                        scale = max_size / max(height, width)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        display_frame = cv2.resize(display_frame, (new_width, new_height))
                    
                    # Chuyển đổi sang RGB và PIL
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(display_frame)
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
            display_image = image.copy()
            cv2.polylines(display_image, [self.roi_points], True, (0, 255, 0), 2)
            
            # Resize ảnh để hiển thị
            height, width = display_image.shape[:2]
            max_size = 800
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                display_image = cv2.resize(display_image, (new_width, new_height))
            
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
            messagebox.showerror("Lỗi", f"Lỗi khi xử lý ảnh: {str(e)}")
            
    def display_prediction_results(self, prediction, probabilities):
        """Hiển thị kết quả dự đoán"""
        result_text = f"""🎯 KẾT QUẢ DỰ ĐOÁN
{'='*30}
📊 Mức than dự đoán: {prediction}

📈 XÁC SUẤT TỪNG LOẠI:
{'='*30}
"""
        
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
            
if __name__ == "__main__":
    root = tk.Tk()
    app = CoalClassificationGUI(root)
    root.mainloop()