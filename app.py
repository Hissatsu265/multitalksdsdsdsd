import gradio as gr
import json
import os
import subprocess
import shutil
from datetime import datetime
import uuid

def create_multitalk_demo():
    # Tạo thư mục lưu trữ nếu chưa tồn tại
    os.makedirs("multitalk_data", exist_ok=True)
    
    def process_multitalk(prompt, image_path, audio1_path, bbox1_coords, 
                         audio2_path, bbox2_coords, use_two_audio):
        try:
            # Tạo ID duy nhất cho session
            session_id = str(uuid.uuid4())[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Tạo thư mục cho session này
            session_folder = f"multitalk_data/{timestamp}_{session_id}"
            os.makedirs(session_folder, exist_ok=True)
            
            # Copy các file input vào session folder
            if image_path and os.path.exists(image_path):
                image_filename = f"image_{session_id}.{image_path.split('.')[-1]}"
                image_dest = os.path.join(session_folder, image_filename)
                shutil.copy(image_path, image_dest)
            else:
                return "Lỗi: Không tìm thấy file ảnh", None
            
            if audio1_path and os.path.exists(audio1_path):
                audio1_filename = f"audio1_{session_id}.{audio1_path.split('.')[-1]}"
                audio1_dest = os.path.join(session_folder, audio1_filename)
                shutil.copy(audio1_path, audio1_dest)
            else:
                return "Lỗi: Không tìm thấy file audio 1", None
            
            # Tạo JSON config
            config = {
                "prompt": prompt if prompt else "",
                "cond_image": image_dest,
                "cond_audio": {
                    "person1": audio1_dest
                }
            }
            
            # Nếu sử dụng 2 audio
            if use_two_audio and audio2_path and os.path.exists(audio2_path):
                audio2_filename = f"audio2_{session_id}.{audio2_path.split('.')[-1]}"
                audio2_dest = os.path.join(session_folder, audio2_filename)
                shutil.copy(audio2_path, audio2_dest)
                
                config["audio_type"] = "add"
                config["cond_audio"]["person2"] = audio2_dest
                
                # Parse bbox coordinates
                try:
                    if bbox1_coords:
                        bbox1 = [int(x.strip()) for x in bbox1_coords.split(',')]
                        if len(bbox1) != 4:
                            return "Lỗi: Bbox person1 phải có 4 số (x,y,w,h)", None
                    else:
                        bbox1 = [350, 600, 220, 400]  # default
                    
                    if bbox2_coords:
                        bbox2 = [int(x.strip()) for x in bbox2_coords.split(',')]
                        if len(bbox2) != 4:
                            return "Lỗi: Bbox person2 phải có 4 số (x,y,w,h)", None
                    else:
                        bbox2 = [300, 300, 280, 300]  # default
                    
                    config["bbox"] = {
                        "person1": bbox1,
                        "person2": bbox2
                    }
                except ValueError:
                    return "Lỗi: Tọa độ bbox phải là số nguyên", None
            
            # Lưu JSON config
            json_path = os.path.join(session_folder, f"config_{session_id}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            # Tạo tên file output
            output_name = f"multiperson_{session_id}"
            
            # Tạo lệnh chạy
            cmd = [
                "python3", "generate_multitalk.py",
                "--ckpt_dir", "weights/Wan2.1-I2V-14B-480P",
                "--wav2vec_dir", "weights/chinese-wav2vec2-base",
                "--input_json", json_path,
                "--sample_steps", "15",
                "--num_persistent_param_in_dit", "11000000000",
                "--mode", "streaming",
                "--use_teacache",
                "--sample_shift", "9",
                "--use_apg", "--apg_momentum", "-0.74", "--apg_norm_threshold", "52",
                "--teacache_thresh", "0.2",
                "--size", "multitalk-480",
                "--motion_frame", "25",
                "--sample_text_guide_scale", "7",
                "--sample_audio_guide_scale", "4",
                "--save_file", output_name
            ]
            
            # Chạy lệnh
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            if result.returncode != 0:
                error_msg = f"Lỗi khi chạy MultiTalk:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
                return error_msg, None
            
            # Tìm file video kết quả
            possible_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            video_path = None
            
            for ext in possible_extensions:
                potential_path = f"{output_name}{ext}"
                if os.path.exists(potential_path):
                    video_path = potential_path
                    break
            
            if not video_path:
                return f"Không tìm thấy video kết quả với tên {output_name}", None
            
            # Copy video vào session folder
            video_dest = os.path.join(session_folder, os.path.basename(video_path))
            shutil.copy(video_path, video_dest)
            
            success_msg = f"Đã tạo video thành công!\nSession ID: {session_id}\nThư mục: {session_folder}\nConfig JSON: {json_path}"
            
            return success_msg, video_dest
            
        except Exception as e:
            return f"Lỗi: {str(e)}", None
    
    def toggle_audio2_inputs(use_two_audio):
        return [
            gr.update(visible=use_two_audio),  # audio2
            gr.update(visible=use_two_audio),  # bbox1
            gr.update(visible=use_two_audio)   # bbox2
        ]
    
    # Tạo giao diện Gradio
    with gr.Blocks(title="MultiTalk Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎭 MultiTalk Demo")
        gr.Markdown("Tạo video talking head từ hình ảnh và audio")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📝 Inputs")
                
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Nhập prompt mô tả (có thể để trống)",
                    lines=2
                )
                
                image_path = gr.Textbox(
                    label="Đường dẫn ảnh",
                    placeholder="/path/to/image.jpg",
                    info="Đường dẫn tới file ảnh"
                )
                
                audio1_path = gr.Textbox(
                    label="Đường dẫn audio người 1",
                    placeholder="/path/to/audio1.wav",
                    info="Đường dẫn tới file audio cho người 1"
                )
                
                use_two_audio = gr.Checkbox(
                    label="Sử dụng 2 người (2 audio)",
                    value=False,
                    info="Tick để bật chế độ 2 người"
                )
                
                audio2_path = gr.Textbox(
                    label="Đường dẫn audio người 2",
                    placeholder="/path/to/audio2.wav",
                    info="Đường dẫn tới file audio cho người 2",
                    visible=False
                )
                
                bbox1_coords = gr.Textbox(
                    label="Tọa độ bbox người 1 (x,y,w,h)",
                    placeholder="350,600,220,400",
                    info="4 số cách nhau bởi dấu phẩy",
                    visible=False
                )
                
                bbox2_coords = gr.Textbox(
                    label="Tọa độ bbox người 2 (x,y,w,h)",
                    placeholder="300,300,280,300",
                    info="4 số cách nhau bởi dấu phẩy",
                    visible=False
                )
                
                generate_btn = gr.Button("🚀 Tạo Video", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                gr.Markdown("## 📤 Outputs")
                
                status_output = gr.Textbox(
                    label="Trạng thái",
                    lines=10,
                    max_lines=20,
                    interactive=False
                )
                
                video_output = gr.Video(
                    label="Video kết quả",
                    height=400
                )
        
        # Event handlers
        use_two_audio.change(
            fn=toggle_audio2_inputs,
            inputs=[use_two_audio],
            outputs=[audio2_path, bbox1_coords, bbox2_coords]
        )
        
        generate_btn.click(
            fn=process_multitalk,
            inputs=[
                prompt, image_path, audio1_path, bbox1_coords,
                audio2_path, bbox2_coords, use_two_audio
            ],
            outputs=[status_output, video_output]
        )
        
        # Thêm thông tin hướng dẫn
        with gr.Accordion("📋 Hướng dẫn sử dụng", open=False):
            gr.Markdown("""
            ### Cách sử dụng:
            1. **Prompt**: Nhập mô tả văn bản (có thể để trống)
            2. **Đường dẫn ảnh**: Nhập đường dẫn tới file ảnh đầu vào
            3. **Audio người 1**: Nhập đường dẫn tới file audio cho người thứ nhất
            4. **Sử dụng 2 người**: Tick vào nếu muốn sử dụng 2 người trong video
            5. **Audio người 2**: (Nếu dùng 2 người) Đường dẫn audio cho người thứ hai
            6. **Tọa độ bbox**: (Nếu dùng 2 người) Tọa độ vùng mặt, format: x,y,width,height
            
            ### Lưu ý:
            - Tất cả files sẽ được lưu trong thư mục `multitalk_data/`
            - File JSON config sẽ được tự động tạo
            - Video kết quả sẽ có tên `multiperson_[session_id]`
            - Đảm bảo các đường dẫn file input đã tồn tại
            """)
    
    return demo

# Chạy demo
if __name__ == "__main__":
    demo = create_multitalk_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )