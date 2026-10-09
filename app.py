"""Streamlit Web Application for Text-to-Image Generation and Model Analytics."""
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional
from PIL import Image
import torch
import streamlit as st

# Setup path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.loader import load_config
from src.inference.generator import ImageGenerator
from src.monitoring.memory_monitor import MemoryMonitor
from src.utils.image_utils import save_image_safely
from src.utils.paths import get_timestamp_str

# Configure page settings
st.set_page_config(
    page_title="AI Diffusion Studio | Text-to-Image",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling (Dark Mode, Glassmorphism, Modern Gradients)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
    }
    
    /* Header Gradient & Badge */
    .main-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(236, 72, 153, 0.15));
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .header-title {
        font-size: 2.2rem;
        background: linear-gradient(90deg, #6366F1, #EC4899, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .badge-gpu {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    /* Glassmorphic Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 16px;
        backdrop-filter: blur(8px);
    }
    
    /* Primary Action Button */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #6366F1 0%, #A855F7 50%, #EC4899 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 12px 28px;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(236, 72, 153, 0.6);
    }
    
    /* Prompt Suggestion Chips */
    .chip-btn {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.82rem;
        cursor: pointer;
        display: inline-block;
        margin: 4px;
        transition: all 0.2s ease;
    }
    
    /* Metadata Badge Grid */
    .meta-box {
        background: rgba(15, 23, 42, 0.7);
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .meta-label {
        font-size: 0.75rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .meta-val {
        font-size: 1.05rem;
        font-weight: 600;
        color: #F8FAFC;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_generator(checkpoint_path: Optional[str]) -> ImageGenerator:
    """Load and cache the fine-tuned Stable Diffusion 1.5 pipeline."""
    config = load_config("configs/config.yaml")
    ckpt = checkpoint_path if checkpoint_path != "Base SD 1.5 (Pretrained)" else None
    generator = ImageGenerator(config=config, checkpoint_path=ckpt)
    return generator


def get_available_checkpoints() -> List[str]:
    """Scan checkpoints directory for available model weights."""
    ckpt_dir = Path("checkpoints")
    options = []
    if (ckpt_dir / "best_checkpoint").exists():
        options.append(str(ckpt_dir / "best_checkpoint"))
    if (ckpt_dir / "final_checkpoint").exists():
        options.append(str(ckpt_dir / "final_checkpoint"))

    if ckpt_dir.exists():
        for p in sorted(ckpt_dir.glob("checkpoint_step_*")):
            if p.is_dir() and str(p) not in options:
                options.append(str(p))

    options.append("Base SD 1.5 (Pretrained)")
    return options


def get_gallery_images() -> List[Path]:
    """Collect generated images for gallery view."""
    img_paths = []
    for search_dir in ["outputs/inference", "outputs/generated_images", "outputs/test_results"]:
        p = Path(search_dir)
        if p.exists():
            img_paths.extend(sorted(p.glob("*.png"), key=os.path.getmtime, reverse=True))
    return img_paths


def main():
    # Header Banner
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Mode"
    vram_str = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB" if cuda_avail else "N/A"

    st.markdown(
        f"""
        <div class="main-header">
            <div>
                <h1 class="header-title">🎨 Text-to-Image Diffusion Studio</h1>
                <p style="color: #94A3B8; margin: 4px 0 0 0; font-size: 0.95rem;">
                    Config-Driven Stable Diffusion 1.5 Fine-Tuning & High-Fidelity Generation Engine
                </p>
            </div>
            <div class="badge-gpu">
                ⚡ {gpu_name} ({vram_str} VRAM)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Controls
    with st.sidebar:
        st.markdown("### ⚙️ Model & Generation Controls")
        
        # Checkpoint Selector
        checkpoints = get_available_checkpoints()
        selected_ckpt = st.selectbox(
            "Select Checkpoint",
            options=checkpoints,
            index=0 if checkpoints else 0,
            help="Choose between your best fine-tuned Pokemon model or the base SD 1.5 checkpoint.",
        )

        st.divider()
        st.markdown("#### 📐 Dimensions & Quality")

        res_choice = st.selectbox(
            "Image Aspect Ratio",
            options=["512 x 512 (Square 1:1)", "512 x 768 (Portrait 2:3)", "768 x 512 (Landscape 3:2)"],
            index=0,
        )
        if "Square" in res_choice:
            img_h, img_w = 512, 512
        elif "Portrait" in res_choice:
            img_h, img_w = 768, 512
        else:
            img_h, img_w = 512, 768

        steps = st.slider("Denoising Steps", min_value=15, max_value=50, value=30, step=5,
                          help="Higher steps improve details (25-35 recommended for DPMSolver).")
        
        guidance = st.slider("Guidance Scale (CFG)", min_value=1.0, max_value=15.0, value=7.5, step=0.5,
                             help="How strictly the image adheres to the text prompt.")

        num_images = st.selectbox("Number of Images", options=[1, 2, 4], index=0)

        st.divider()
        st.markdown("#### 🎛️ Sampler & Seed")
        scheduler_type = st.selectbox(
            "Noise Scheduler",
            options=["DPMSolverMultistepScheduler", "EulerDiscreteScheduler", "PNDMScheduler", "DDPMScheduler"],
            index=0,
        )

        random_seed = st.checkbox("🎲 Randomize Seed", value=True)
        if random_seed:
            seed_val = int(torch.randint(0, 1000000, (1,)).item())
        else:
            seed_val = st.number_input("Fixed Seed", min_value=0, max_value=9999999, value=42)

    # Main Navigation Tabs
    tab_studio, tab_gallery, tab_analytics = st.tabs([
        "✨ Generation Studio",
        "🖼️ Gallery & History",
        "📊 Training Analytics & Report",
    ])

    # ---------------- TAB 1: STUDIO ----------------
    with tab_studio:
        st.markdown("### 💬 Enter Text Prompt")
        
        # Quick Inspiration Prompt Chips
        st.markdown("<div style='font-size: 0.85rem; color: #94A3B8; margin-bottom: 6px;'>💡 Quick Inspiration Presets (Click to use):</div>", unsafe_allow_html=True)
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        
        preset_prompt = None
        with col_c1:
            if st.button("⚡ Neon Electric Pikachu"):
                preset_prompt = "A cute yellow electric mouse pokemon surrounded by blue lightning sparks, neon cyberpunk style, highly detailed"
        with col_c2:
            if st.button("🐉 Fiery Winged Dragon"):
                preset_prompt = "A majestic fiery red dragon pokemon with ruby scales flying over a volcanic mountain, high resolution"
        with col_c3:
            if st.button("🌊 Mystic Water Turtle"):
                preset_prompt = "A cute aquatic turtle pokemon with a glowing turquoise shell resting near a waterfall in a vibrant jungle"
        with col_c4:
            if st.button("🌸 Blossom Fairy Fox"):
                preset_prompt = "A cute magical nine-tailed fairy fox pokemon sitting under cherry blossom trees in soft sunlight, 8k"

        # Text input area
        default_prompt = preset_prompt if preset_prompt else "A cute pokemon with blue fur and golden eyes sitting peacefully on soft green grass"
        user_prompt = st.text_area(
            "Prompt",
            value=default_prompt,
            height=90,
            placeholder="Describe what you want to generate...",
            label_visibility="collapsed",
        )

        with st.expander("🛡️ Negative Prompt (Quality Filters & Artifact Protection)"):
            neg_prompt = st.text_area(
                "Negative Prompt",
                value="blurry, low quality, distorted, deformed, ugly, artifacts, bad anatomy, extra limbs, watermark",
                height=60,
                label_visibility="collapsed",
            )

        # Generate Action Button
        generate_clicked = st.button("🚀 Generate Image(s)")

        if generate_clicked:
            if not user_prompt.strip():
                st.error("Please enter a valid text prompt.")
            else:
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                with status_placeholder.container():
                    st.info(f"⏳ Loading pipeline & synthesizing image for: *\"{user_prompt}\"*...")

                try:
                    start_t = time.perf_counter()
                    generator = get_generator(selected_ckpt)

                    with st.spinner("Generating high-resolution diffusion output..."):
                        results = generator.generate_images(
                            prompt=user_prompt,
                            negative_prompt=neg_prompt,
                            height=img_h,
                            width=img_w,
                            num_inference_steps=steps,
                            guidance_scale=guidance,
                            num_images_per_prompt=num_images,
                            seed=seed_val,
                            scheduler_type=scheduler_type,
                            output_dir="outputs/inference",
                        )
                    elapsed = time.perf_counter() - start_t
                    status_placeholder.empty()

                    st.success(f"🎉 Generated {len(results)} image(s) in {elapsed:.2f} seconds!")

                    # Render Results
                    cols = st.columns(len(results))
                    for i, res in enumerate(results):
                        with cols[i]:
                            img = res["image"]
                            meta = res["metadata"]
                            st.image(img, use_container_width=True, caption=f"Sample #{i+1} (Seed: {seed_val})")
                            
                            # Download Button
                            img_path = Path(meta["file_path"])
                            with open(img_path, "rb") as f:
                                st.download_button(
                                    label=f"💾 Download Sample #{i+1}",
                                    data=f.read(),
                                    file_name=img_path.name,
                                    mime="image/png",
                                    key=f"dl_{img_path.name}_{i}",
                                )

                    # Metadata Summary Card
                    st.markdown("---")
                    st.markdown("#### 📋 Generation Metadata")
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.markdown(f"<div class='meta-box'><div class='meta-label'>Model</div><div class='meta-val'>{Path(selected_ckpt).name if selected_ckpt != 'Base SD 1.5 (Pretrained)' else 'Base SD1.5'}</div></div>", unsafe_allow_html=True)
                    m2.markdown(f"<div class='meta-box'><div class='meta-label'>Resolution</div><div class='meta-val'>{img_w}x{img_h}</div></div>", unsafe_allow_html=True)
                    m3.markdown(f"<div class='meta-box'><div class='meta-label'>Steps / CFG</div><div class='meta-val'>{steps} / {guidance}</div></div>", unsafe_allow_html=True)
                    m4.markdown(f"<div class='meta-box'><div class='meta-label'>Seed</div><div class='meta-val'>{seed_val}</div></div>", unsafe_allow_html=True)
                    m5.markdown(f"<div class='meta-box'><div class='meta-label'>Latency</div><div class='meta-val'>{elapsed:.2f}s</div></div>", unsafe_allow_html=True)

                except Exception as e:
                    status_placeholder.empty()
                    st.error(f"Error generating image: {e}")

    # ---------------- TAB 2: GALLERY ----------------
    with tab_gallery:
        st.markdown("### 🖼️ Generated Images Gallery")
        gallery_images = get_gallery_images()

        if not gallery_images:
            st.info("No images generated yet. Go to the 'Generation Studio' tab to create your first image!")
        else:
            st.write(f"Showing **{len(gallery_images)}** generated images:")
            g_cols = st.columns(3)
            for idx, img_p in enumerate(gallery_images):
                with g_cols[idx % 3]:
                    try:
                        im = Image.open(img_p)
                        st.image(im, use_container_width=True)
                        st.caption(f"📁 `{img_p.name}`")
                        
                        # Check for metadata json
                        meta_p = img_p.with_name(img_p.stem + "_metadata.json")
                        if meta_p.exists():
                            with open(meta_p, "r", encoding="utf-8") as mf:
                                m_data = json.load(mf)
                                with st.expander("ℹ️ Prompt & Parameters"):
                                    st.write(f"**Prompt**: {m_data.get('prompt')}")
                                    st.write(f"**Seed**: `{m_data.get('seed')}` | **Steps**: `{m_data.get('num_inference_steps')}` | **CFG**: `{m_data.get('guidance_scale')}`")

                        with open(img_p, "rb") as bf:
                            st.download_button(
                                "📥 Download",
                                data=bf.read(),
                                file_name=img_p.name,
                                mime="image/png",
                                key=f"gal_dl_{idx}_{img_p.name}",
                            )
                    except Exception as err:
                        st.warning(f"Could not load {img_p.name}: {err}")

    # ---------------- TAB 3: TRAINING ANALYTICS ----------------
    with tab_analytics:
        st.markdown("### 📊 Experiment Diagnostics & Training Summary")
        
        report_files = sorted(Path("logs").glob("**/diagnostic_report.md"), key=os.path.getmtime, reverse=True)
        if report_files:
            latest_report = report_files[0]
            with open(latest_report, "r", encoding="utf-8") as rf:
                report_content = rf.read()
            st.markdown(report_content)
        else:
            st.info("No diagnostic report found in `logs/` yet.")

        st.divider()
        st.markdown("#### 📸 Training Preview Snapshots")
        preview_imgs = sorted(Path("logs").glob("**/generated_previews/*.png"), key=os.path.getmtime, reverse=True)
        if preview_imgs:
            prev_cols = st.columns(4)
            for p_idx, p_path in enumerate(preview_imgs[:12]):
                with prev_cols[p_idx % 4]:
                    st.image(Image.open(p_path), use_container_width=True, caption=p_path.name)
        else:
            st.info("No training preview images found.")


if __name__ == "__main__":
    main()
