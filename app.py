import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import supervision as sv
from rfdetr import RFDETRSmall
import os
import io
import base64

# ─── Model Loading ────────────────────────────────────────────────────────────
MODEL_PATH = "opM3weights.pt"
CLASS_NAMES = ["oil_palm"]   # adjust if your model has more classes

print("⏳ Loading RF-DETR Small model...")
model = RFDETRSmall(pretrain_weights=None)

checkpoint = torch.load(MODEL_PATH, map_location="cpu")

# Handle various checkpoint formats
if isinstance(checkpoint, dict):
    state_dict = (
        checkpoint.get("model")
        or checkpoint.get("state_dict")
        or checkpoint.get("model_state_dict")
        or checkpoint
    )
else:
    state_dict = checkpoint

try:
    model.model.load_state_dict(state_dict, strict=True)
except Exception:
    # Try loading from nested key if needed
    try:
        model.model.load_state_dict(state_dict, strict=False)
        print("⚠️  Loaded with strict=False — some weights may differ.")
    except Exception as e:
        print(f"❌ Weight loading error: {e}")

model.model.eval()
print("✅ Model loaded successfully!")

# ─── Annotators ───────────────────────────────────────────────────────────────
BOX_COLOR    = sv.Color(r=34, g=197, b=94)   # emerald green
TEXT_COLOR   = sv.Color(r=255, g=255, b=255)
BOX_THICKNESS = 2

box_annotator   = sv.BoxAnnotator(color=BOX_COLOR, thickness=BOX_THICKNESS)
label_annotator = sv.LabelAnnotator(
    color=BOX_COLOR,
    text_color=TEXT_COLOR,
    text_scale=0.5,
    text_thickness=1,
    text_padding=4,
)

# ─── Inference ────────────────────────────────────────────────────────────────
def predict(image: Image.Image, confidence: float):
    if image is None:
        return None, "⚠️ No image provided."

    image_rgb = image.convert("RGB")

    # RF-DETR predict returns supervision Detections directly
    detections = model.predict(image_rgb, threshold=confidence)

    if len(detections) == 0:
        annotated = image_rgb
        summary = "🌿 No oil palm trees detected above the confidence threshold."
    else:
        labels = [
            f"{CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f'class_{cid}'} {conf:.0%}"
            for cid, conf in zip(detections.class_id, detections.confidence)
        ]
        annotated = np.array(image_rgb)
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
        annotated = Image.fromarray(annotated)

        count     = len(detections)
        avg_conf  = float(np.mean(detections.confidence))
        summary   = (
            f"🌴 **{count} oil palm tree{'s' if count != 1 else ''} detected**  \n"
            f"📊 Average confidence: **{avg_conf:.1%}**  \n"
            f"🎯 Threshold used: **{confidence:.0%}**"
        )

    return annotated, summary


# ─── Custom CSS ───────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: #0a0f0a !important;
}

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 24px !important;
}

/* ── Header ── */
#app-header {
    text-align: center;
    padding: 40px 20px 28px;
    background: linear-gradient(135deg, #0d1f0d 0%, #112211 50%, #0d1f0d 100%);
    border-radius: 20px;
    border: 1px solid #1e3a1e;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
#app-header::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(34,197,94,0.12) 0%, transparent 70%);
    pointer-events: none;
}
#app-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 8px;
    letter-spacing: -0.5px;
}
#app-header h1 span { color: #22c55e; }
#app-header p {
    font-size: 0.95rem;
    color: #6b9e6b;
    margin: 0;
    font-weight: 400;
}

/* ── Badges ── */
.badge-row {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 16px;
}
.badge {
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.25);
    color: #4ade80;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.3px;
}

/* ── Panels ── */
.panel {
    background: #0f1f0f !important;
    border: 1px solid #1a2e1a !important;
    border-radius: 16px !important;
}

/* ── Labels ── */
label span, .label-wrap span {
    color: #4ade80 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
}

/* ── Image widgets ── */
.svelte-1ipelgc, [data-testid="image"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid #1e3a1e !important;
    background: #080d08 !important;
}

/* ── Slider ── */
input[type=range] {
    accent-color: #22c55e !important;
}
.gr-slider .output-text {
    color: #4ade80 !important;
}

/* ── Predict button ── */
#predict-btn {
    background: linear-gradient(135deg, #16a34a, #22c55e) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 14px 0 !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: opacity 0.2s ease, transform 0.15s ease !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 20px rgba(34,197,94,0.25) !important;
}
#predict-btn:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}
#predict-btn:active { transform: translateY(0) !important; }

/* ── Clear button ── */
#clear-btn {
    background: transparent !important;
    border: 1px solid #1e3a1e !important;
    border-radius: 12px !important;
    color: #6b9e6b !important;
    font-weight: 500 !important;
    padding: 10px 0 !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
#clear-btn:hover {
    border-color: #22c55e !important;
    color: #22c55e !important;
}

/* ── Result summary box ── */
#result-summary {
    background: #0b1a0b !important;
    border: 1px solid #1a2e1a !important;
    border-radius: 12px !important;
    padding: 16px !important;
    color: #a3d9a3 !important;
    font-size: 0.92rem !important;
    line-height: 1.7 !important;
    min-height: 80px !important;
}
#result-summary strong, #result-summary b { color: #4ade80 !important; }

/* ── Footer ── */
#footer {
    text-align: center;
    padding: 20px;
    color: #3d5c3d;
    font-size: 0.78rem;
    margin-top: 16px;
}

/* ── Hide Gradio branding ── */
footer { display: none !important; }
"""

# ─── Layout ───────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="Oil Palm Detector") as demo:

    gr.HTML("""
    <div id="app-header">
      <h1>🌴 Oil Palm <span>Detector</span></h1>
      <p>Aerial-view detection powered by RF-DETR · Fine-tuned on oil palm datasets</p>
      <div class="badge-row">
        <span class="badge">RF-DETR Small</span>
        <span class="badge">Aerial Imagery</span>
        <span class="badge">Real-time Inference</span>
      </div>
    </div>
    """)

    with gr.Row(equal_height=True):
        # ── Left column: inputs ──
        with gr.Column(scale=1, elem_classes="panel"):
            input_image = gr.Image(
                type="pil",
                label="Upload or Paste Image",
                elem_id="input-image",
                height=380,
                sources=["upload", "clipboard"],
            )
            confidence_slider = gr.Slider(
                minimum=0.05,
                maximum=0.95,
                value=0.3,
                step=0.05,
                label="Confidence Threshold",
                elem_id="conf-slider",
            )
            with gr.Row():
                predict_btn = gr.Button("🔍 Detect Oil Palms", elem_id="predict-btn", variant="primary")
            with gr.Row():
                clear_btn = gr.ClearButton(
                    components=[input_image],
                    value="✕  Clear",
                    elem_id="clear-btn",
                )

        # ── Right column: outputs ──
        with gr.Column(scale=1, elem_classes="panel"):
            output_image = gr.Image(
                type="pil",
                label="Detection Result",
                elem_id="output-image",
                height=380,
                interactive=False,
            )
            result_text = gr.Markdown(
                value="Upload an image and press **Detect** to begin.",
                elem_id="result-summary",
            )

    # ── Examples ──
    gr.Examples(
        examples=[],           # add example image paths here if you have them
        inputs=input_image,
        label="",
    )

    gr.HTML('<div id="footer">Oil Palm Detection · RF-DETR Small · Built with Gradio</div>')

    # ── Events ──
    predict_btn.click(
        fn=predict,
        inputs=[input_image, confidence_slider],
        outputs=[output_image, result_text],
    )

    # Also trigger on image change for quick feedback
    input_image.change(
        fn=lambda img, conf: predict(img, conf) if img is not None else (None, "Upload an image and press **Detect** to begin."),
        inputs=[input_image, confidence_slider],
        outputs=[output_image, result_text],
    )

if __name__ == "__main__":
    demo.launch()
