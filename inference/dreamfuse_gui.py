import gradio as gr
import spaces
from PIL import Image, ImageDraw, ImageOps
import base64, json
from io import BytesIO
import torch.nn.functional as F
import json
from typing import List
from dataclasses import dataclass, field
from .dreamfuse_inference import DreamFuseInference, InferenceConfig
import numpy as np
import os
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
import torch
import subprocess
import base64

subprocess.run("rm -rf /data-nvme/zerogpu-offload/*", env={}, shell=True)
generated_images = []


RMBG_model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
RMBG_model = RMBG_model.to("cuda")
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@spaces.GPU
def remove_bg(image):
    im = image.convert("RGB")
    input_tensor = transform(im).unsqueeze(0).to("cuda")
    with torch.no_grad():
        preds = RMBG_model(input_tensor)[-1].sigmoid().cpu()[0].squeeze()
    mask = transforms.ToPILImage()(preds).resize(im.size)
    return mask




def get_base64_logo(path="examples/logo.png"):
    image = Image.open(path).convert("RGBA")
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_img = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{base64_img}"


class DreamFuseGUI:
    def __init__(self):
        self.examples = [
            ["./examples/valid/002_1.png", "./examples/valid/002_0.png"],
            ["./examples/valid/010_1.png", "./examples/valid/010_0.png"],
        ]
        self.examples = [[Image.open(x) for x in example] for example in self.examples]
        self.css_style = self._get_css_style()
        self.js_script = self._get_js_script()

    def _get_css_style(self):
        return """
        input[type="file"] {
        border: 1px solid #ccc !important;
        background-color: #f9f9f9 !important;
        padding: 8px !important;
        border-radius: 6px !important;
        }
        body {
        background-color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #212121;
        }
        .gradio-container {
        max-width: 1200px;
        margin: auto;
        background: #ffffff;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0px 4px 16px rgba(0, 0, 0, 0.05);
        }
        h1, h2 {
        text-align: center;
        color: #222;
        }
        #canvas_preview {
        min-height: 420px;
        border: 2px dashed #ccc;
        background-color: #fafafa;
        border-radius: 8px;
        padding: 10px;
        display: flex;
        justify-content: center;
        align-items: center;
        color: #999;
        font-size: 16px;
        }
        .gr-button {
        background-color: #2196f3;
        border: 1px solid #1976d2;
        color: #ffffff;
        padding: 10px 20px;
        border-radius: 6px;
        font-size: 15px;
        cursor: pointer;
        transition: background-color 0.2s ease;
        }
        .gr-button:hover {
        background-color: #1976d2;
        }
        #small-examples {
        width: 200px;
        margin: 10px 0;
        border: 1px solid #ddd;
        border-radius: 8px;
        overflow: hidden;
        background: #fff;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        }
        .markdown-text {
        color: #333;
        font-size: 15px;
        line-height: 1.6;
        }
        #canvas-preview-container {
        background: #fafafa;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        margin-top: 10px;
        }
        [id^="section-"] {
        background-color: #ffffff;
        border: 1px solid #dddddd;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        margin-bottom: 16px;
        }
        .svelte-1ipelgc {
        flex-wrap: nowrap !important;
        gap: 24px !important;
        }
        """

    def _get_js_script(self):
        return r"""
        async () => {
            window.updateTransformation = function() {
                const img = document.getElementById('draggable-img');
                const container = document.getElementById('canvas-container');
                if (!img || !container) return;
                const left = parseFloat(img.style.left) || 0;
                const top = parseFloat(img.style.top) || 0;
                const canvasSize = 400;
                const data_original_width = parseFloat(img.getAttribute('data-original-width'));
                const data_original_height = parseFloat(img.getAttribute('data-original-height'));
                const bgWidth = parseFloat(container.dataset.bgWidth);
                const bgHeight = parseFloat(container.dataset.bgHeight);
                const scale_ratio = img.clientWidth / data_original_width;
                const transformation = {
                    drag_left: left,
                    drag_top: top,
                    drag_width: img.clientWidth,
                    drag_height: img.clientHeight,
                    data_original_width: data_original_width,
                    data_original_height: data_original_height,
                    scale_ratio: scale_ratio
                };
                const transInput = document.querySelector("#transformation_info textarea");
                if(transInput){
                   const newValue = JSON.stringify(transformation);
                   const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
                   nativeSetter.call(transInput, newValue);
                   transInput.dispatchEvent(new Event('input', { bubbles: true }));
                   console.log("Transformation info updated: ", newValue);
                } else {
                   console.log("找不到 transformation_info 的 textarea 元素");
                }
            };
            globalThis.initializeDrag = () => {
                const oldImg = document.getElementById('draggable-img');
                const container = document.getElementById('canvas-container');
                const slider = document.getElementById('scale-slider');
                if (!oldImg || !container || !slider) {
                    return;
                }
                const img = oldImg.cloneNode(true);
                oldImg.replaceWith(img);
                img.ondragstart = (e) => { e.preventDefault(); return false; };
                let offsetX = 0, offsetY = 0;
                let isDragging = false;
                let scaleAnchor = null;
                img.addEventListener('mousedown', (e) => {
                    isDragging = true;
                    img.style.cursor = 'grabbing';
                    const imgRect = img.getBoundingClientRect();
                    offsetX = e.clientX - imgRect.left;
                    offsetY = e.clientY - imgRect.top;
                    img.style.transform = "none";
                    img.style.left = img.offsetLeft + "px";
                    img.style.top = img.offsetTop + "px";
                    console.log("mousedown: left=", img.style.left, "top=", img.style.top);
                });
                document.addEventListener('mousemove', (e) => {
                    if (!isDragging) return;
                    e.preventDefault();
                    const containerRect = container.getBoundingClientRect();
                    let left = e.clientX - containerRect.left - offsetX;
                    let top  = e.clientY - containerRect.top  - offsetY;
                    const minLeft = -img.clientWidth * (7/8);
                    const maxLeft = containerRect.width - img.clientWidth * (1/8);
                    const minTop = -img.clientHeight * (7/8);
                    const maxTop = containerRect.height - img.clientHeight * (1/8);
                    if (left < minLeft) left = minLeft;
                    if (left > maxLeft) left = maxLeft;
                    if (top < minTop) top = minTop;
                    if (top > maxTop) top = maxTop;
                    img.style.left = left + "px";
                    img.style.top = top + "px";
                });
                window.addEventListener('mouseup', (e) => {
                    if (isDragging) {
                        isDragging = false;
                        img.style.cursor = 'grab';
                        const containerRect = container.getBoundingClientRect();
                        const bgWidth = parseFloat(container.dataset.bgWidth);
                        const bgHeight = parseFloat(container.dataset.bgHeight);
                        const offsetLeft = (containerRect.width - bgWidth) / 2;
                        const offsetTop = (containerRect.height - bgHeight) / 2;
                        const absoluteLeft = parseFloat(img.style.left);
                        const absoluteTop = parseFloat(img.style.top);
                        const relativeX = absoluteLeft - offsetLeft;
                        const relativeY = absoluteTop - offsetTop;
                        document.getElementById("coordinate").textContent =
                            `Location: (x=${relativeX.toFixed(2)}, y=${relativeY.toFixed(2)})`;
                        updateTransformation();
                    }
                    scaleAnchor = null;
                });
                slider.addEventListener('mousedown', (e) => {
                    const containerRect = container.getBoundingClientRect();
                    const imgRect = img.getBoundingClientRect();
                    scaleAnchor = {
                        x: imgRect.left + imgRect.width/2 - containerRect.left,
                        y: imgRect.top + imgRect.height/2 - containerRect.top
                    };
                    console.log("Slider mousedown, captured scaleAnchor: ", scaleAnchor);
                });
                slider.addEventListener('input', (e) => {
                    const scale = parseFloat(e.target.value);
                    const originalWidth = parseFloat(img.getAttribute('data-original-width'));
                    const originalHeight = parseFloat(img.getAttribute('data-original-height'));
                    const newWidth = originalWidth * scale;
                    const newHeight = originalHeight * scale;
                    const containerRect = container.getBoundingClientRect();
                    let centerX, centerY;
                    if (scaleAnchor) {
                        centerX = scaleAnchor.x;
                        centerY = scaleAnchor.y;
                    } else {
                        const imgRect = img.getBoundingClientRect();
                        centerX = imgRect.left + imgRect.width/2 - containerRect.left;
                        centerY = imgRect.top + imgRect.height/2 - containerRect.top;
                    }
                    const newLeft = centerX - newWidth/2;
                    const newTop = centerY - newHeight/2;
                    img.style.width = newWidth + "px";
                    img.style.height = newHeight + "px";
                    img.style.left = newLeft + "px";
                    img.style.top = newTop + "px";
                    console.log("slider: scale=", scale, "newWidth=", newWidth, "newHeight=", newHeight);
                    updateTransformation();
                });
                slider.addEventListener('mouseup', (e) => {
                    scaleAnchor = null;
                });
                console.log("✅ 拖拽和缩放事件已绑定");
            };
        }
        """


    def get_next_sequence(self, folder_path):
        # 列出文件夹中的所有文件名
        filenames = os.listdir(folder_path)
        # 提取文件名中的序列号部分（假设是前三位数字）
        sequences = [int(name.split('_')[0]) for name in filenames if name.split('_')[0].isdigit()]
        # 找到最大序列号
        max_sequence = max(sequences, default=-1)
        # 返回下一位序列号，格式为三位数字（如002）
        return f"{max_sequence + 1:03d}"


    def pil_to_base64(self, img):
        if img is None:
            return ""
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        buffered = BytesIO()
        img.save(buffered, format="PNG", optimize=True)
        img_bytes = buffered.getvalue()
        base64_str = base64.b64encode(img_bytes).decode()
        return f"data:image/png;base64,{base64_str}"

    def resize_background_image(self, img, max_size=400):
        if img is None:
            return None
        w, h = img.size
        if w > max_size or h > max_size:
            ratio = min(max_size / w, max_size / h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        return img

    def resize_draggable_image(self, img, max_size=400):
        if img is None:
            return None
        w, h = img.size
        if w > max_size or h > max_size:
            ratio = min(max_size / w, max_size / h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        return img

    def generate_html(self, background_img_b64, bg_width, bg_height, draggable_img_b64, draggable_width, draggable_height, canvas_size=400):
        html_code = f"""
        <html>
        <head>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    text-align: center;
                    font-family: sans-serif;
                    background: transparent;
                    color: #fff;
                }}
                h2 {{
                    margin-top: 1rem;
                }}
                #scale-control {{
                    margin: 1rem auto;
                    width: 400px;
                    text-align: left;
                }}
                #scale-control label {{
                    font-size: 1rem;
                    margin-right: 0.5rem;
                }}
                #canvas-container {{
                    position: relative;
                    width: {canvas_size}px;
                    height: {canvas_size}px;
                    margin: 0 auto;
                    border: 1px dashed rgba(255,255,255,0.5);
                    overflow: hidden;
                    background-image: url('{background_img_b64}');
                    background-repeat: no-repeat;
                    background-position: center;
                    background-size: contain;
                    border-radius: 8px;
                }}
                #draggable-img {{
                    position: absolute;
                    cursor: grab;
                    left: 50%;
                    top: 50%;
                    transform: translate(-50%, -50%);
                    background-color: transparent;
                }}
                #coordinate {{
                    color: #fff;
                    margin-top: 1rem;
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <h2> 3️⃣ Drag and Resize</h2>
            <div id="scale-control">
                <label for="scale-slider">Resize FG:</label>
                <input type="range" id="scale-slider" min="0.1" max="2" step="0.01" value="1">
            </div>
            <div id="canvas-container" data-bg-width="{bg_width}" data-bg-height="{bg_height}">
                <img id="draggable-img" 
                     src="{draggable_img_b64}" 
                     alt="Draggable Image"
                     draggable="false"
                     data-original-width="{draggable_width}"
                     data-original-height="{draggable_height}"
                />
            </div>
            <p id="coordinate">location: (x=?, y=?)</p>
        </body>
        </html>
        """
        return html_code

    def on_upload(self, background_img, draggable_img):
        if background_img is None or draggable_img is None:
            return "<p style='color:red;'>Please upload the background and foreground images。</p>"
        
        if draggable_img.mode != "RGB":
            draggable_img = draggable_img.convert("RGB")
        draggable_img_mask = remove_bg(draggable_img)
        alpha_channel = draggable_img_mask.convert("L")
        draggable_img = draggable_img.convert("RGBA")
        draggable_img.putalpha(alpha_channel)

        resized_bg = self.resize_background_image(background_img, max_size=400)
        bg_w, bg_h = resized_bg.size

        resized_fg = self.resize_draggable_image(draggable_img, max_size=400)
        draggable_width, draggable_height = resized_fg.size

        background_img_b64 = self.pil_to_base64(resized_bg)
        draggable_img_b64 = self.pil_to_base64(resized_fg)

        return self.generate_html(
            background_img_b64, bg_w, bg_h, 
            draggable_img_b64, draggable_width, draggable_height,
            canvas_size=400
        ), draggable_img



    def create_gui(self):
        config = InferenceConfig()
        config.lora_id = 'LL3RD/DreamFuse'
        
        # pipeline = None
        pipeline = DreamFuseInference(config)
        pipeline.gradio_generate = spaces.GPU(duratioin=120)(pipeline.gradio_generate)

        with gr.Blocks(css=self.css_style) as demo:
            modified_fg_state = gr.State() 
            logo_data_url = get_base64_logo()
            gr.HTML(
                f"""
                <div style="display: flex; align-items: center; justify-content: center; gap: 12px; margin-bottom: 20px;">
                    <img src="{logo_data_url}" style="height: 80px;" />
                    <h1 style="margin: 0; font-size: 32px;">DreamFuse</h1>
                </div>
                """
            )
            gr.Markdown('## 📌 4 Easy Steps to Create Your Fusion Image:')
            gr.Markdown(
                """
                1. Upload the foreground and background images you want to fuse.  
                2. Click 'Generate Canvas' to preview the result.  
                3. Drag and resize the foreground object to position it as you like.  
                4. Click 'Run Model' to create the final fused image.
                """,
                elem_classes=["markdown-text"]
            )
            with gr.Row():
                with gr.Column(scale=1, elem_id="section-upload"):
                    gr.Markdown("### 1️⃣ FG&BG Image Upload")
                    with gr.Row():
                        with gr.Column(scale=1):
                            background_img_in = gr.Image(label="Background Image", type="pil", height=240, width=200)
                        with gr.Column(scale=1):
                            draggable_img_in = gr.Image(label="Foreground Image", type="pil", image_mode="RGBA", height=240, width=200)
                    generate_btn = gr.Button("2️⃣ Generate Canvas")

                with gr.Column(scale=1, elem_id="section-preview"):
                    gr.Markdown("### Preview Region")
                    html_out = gr.HTML(
                        value="<p style='text-align:center; color:#aaa;'>Waiting for generating canvas...</p>", 
                        label="drag and resize", 
                        elem_id="canvas_preview"
                    )

            with gr.Row():
                with gr.Column(scale=1, elem_id="section-parameters"):
                    gr.Markdown("### Parameters")
                    seed_slider = gr.Slider(minimum=-1, maximum=100000, step=1, label="Seed", value=12345)
                    cfg_slider = gr.Slider(minimum=1, maximum=10, step=0.1, label="CFG", value=3.5)
                    size_select = gr.Radio(
                        choices=["512", "768", "1024"],
                        value="512",
                        label="Resolution (Higher resolution improves quality, but slows down generation.)",
                    )
                    prompt_text = gr.Textbox(label="Prompt", placeholder="text prompt", value="")
                    text_strength = gr.Slider(minimum=1, maximum=10, step=1, label="Text Strength (Improve text strength to increase responsiveness)", value=1, visible=True)
                    enable_gui = gr.Checkbox(label="GUI", value=True, visible=False)
                    enable_truecfg = gr.Checkbox(label="TrueCFG", value=False, visible=False)
                with gr.Column(scale=1, elem_id="section-results"):
                    gr.Markdown("### Model Result")
                    model_generate_btn = gr.Button("4️⃣ Run Model")
                    transformation_text = gr.Textbox(label="Transformation Info", elem_id="transformation_info", visible=False)
                    model_output = gr.Image(label="Model Output", type="pil", height=512, width=512)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Examples(
                        examples=[self.examples[0]],
                        inputs=[background_img_in, draggable_img_in],
                        # elem_id="small-examples"
                    )
                with gr.Column(scale=1):
                    gr.Examples(
                        examples=[self.examples[2]],
                        inputs=[background_img_in, draggable_img_in],
                        # elem_id="small-examples"
                    )
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Examples(
                        examples=[self.examples[1]],
                        inputs=[background_img_in, draggable_img_in],
                        # elem_id="small-examples"
                    )
                with gr.Column(scale=1):
                    gr.Examples(
                        examples=[self.examples[3]],
                        inputs=[background_img_in, draggable_img_in],
                        # elem_id="small-examples"
                    )
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Examples(
                        examples=[self.examples[4]],
                        inputs=[background_img_in, draggable_img_in],
                        # elem_id="small-examples"
                    )
                with gr.Column(scale=1):
                    gr.Examples(
                        examples=[self.examples[5]],
                        inputs=[background_img_in, draggable_img_in],
                        # elem_id="small-examples"
                    )

            generate_btn.click(
                fn=self.on_upload,
                inputs=[background_img_in, draggable_img_in],
                outputs=[html_out, modified_fg_state],
            ).then(
                fn=None,
                inputs=None,
                outputs=None,
                js="initializeDrag"
            )

            model_generate_btn.click(
                fn=pipeline.gradio_generate,
                # fn=self.pil_to_base64,
                inputs=[background_img_in, modified_fg_state, transformation_text, seed_slider, \
                    prompt_text, enable_gui, cfg_slider, size_select, text_strength, enable_truecfg],
                outputs=model_output
            )
            demo.load(None, None, None, js=self.js_script)
            generate_btn.click(fn=None, inputs=None, outputs=None, js="initializeDrag")

        return demo

if __name__ == "__main__":

    gui = DreamFuseGUI()
    demo = gui.create_gui()
    demo.queue()
    demo.launch()