from PIL import Image, ImageDraw
import gradio as gr

def draw_bounding_box(image: Image.Image, bbox_str: str):
    try:
        x, y, w, h = map(int, bbox_str.strip().split(","))
    except:
        return None, "❌ Sai định dạng! Nhập dạng: x, y, width, height"

    image_with_box = image.copy()
    draw = ImageDraw.Draw(image_with_box)
    draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=3)

    return image_with_box, f"✅ Box vẽ tại (x={x}, y={y}, w={w}, h={h})"

with gr.Blocks() as demo:
    gr.Markdown("## 🖼️ Tải ảnh + Nhập tọa độ Bounding Box")

    with gr.Row():
        img_input = gr.Image(type="pil", label="Ảnh đầu vào")
        bbox_input = gr.Textbox(label="Tọa độ Bounding Box (x, y, w, h)", placeholder="VD: 50, 30, 120, 160")

    with gr.Row():
        img_output = gr.Image(label="Ảnh sau khi vẽ bounding box")
        msg_output = gr.Textbox(label="Thông báo")

    btn = gr.Button("Vẽ Bounding Box")

    btn.click(fn=draw_bounding_box, inputs=[img_input, bbox_input], outputs=[img_output, msg_output])

demo.launch(share=True)