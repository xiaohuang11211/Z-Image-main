"""Z-Image Web UI — 干净布局版"""
import json, os, time, warnings, threading
from pathlib import Path

import torch
import gradio as gr
import psutil

warnings.filterwarnings("ignore")
from utils import ensure_model_weights, load_from_local_dir, set_attention_backend
from zimage import generate

os.environ["ZIMAGE_ATTENTION"] = os.environ.get("ZIMAGE_ATTENTION", "native")
DTYPE = torch.bfloat16
VAE_SCALE = 16

DEFAULT_MODELS = {
    "Z-Image-Turbo": {"repo": "Tongyi-MAI/Z-Image-Turbo", "path": "ckpts/Z-Image-Turbo"},
    "Z-Image":       {"repo": "Tongyi-MAI/Z-Image",       "path": "ckpts/Z-Image"},
}

ATTENTION_OPTIONS = [
    ("native（默认）", "native"),
    ("native_flash", "_native_flash"),
    ("Flash Attn 2", "flash"),
    ("Flash Attn 3", "_flash_3"),
]
HISTORY_FILE = Path("outputs/history.json")
OUTPUT_DIR   = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
CKPT_DIR = Path("ckpts")
CKPT_DIR.mkdir(exist_ok=True)

_cache = {}
_device = None
_cancel_event = threading.Event()


def get_device():
    global _device
    if _device is not None:
        return _device
    if torch.cuda.is_available():
        _device = "cuda"
    elif torch.backends.mps.is_available():
        _device = "mps"
    else:
        _device = "cpu"
    return _device


def scan_local_models():
    models = dict(DEFAULT_MODELS)
    if CKPT_DIR.exists():
        for subdir in CKPT_DIR.iterdir():
            if subdir.is_dir() and (subdir / "transformer").is_dir():
                name = subdir.name
                if name not in models:
                    models[name] = {"repo": None, "path": str(subdir)}
    return models


MODEL_CONFIGS = scan_local_models()


def refresh_model_list():
    global MODEL_CONFIGS
    MODEL_CONFIGS = scan_local_models()
    choices = list(MODEL_CONFIGS.keys())
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def load_model(model_key: str, use_compile: bool = False, attn_backend: str = "native"):
    cache_key = f"{model_key}_c{use_compile}_a{attn_backend}"
    if cache_key in _cache:
        return _cache[cache_key]

    if model_key not in MODEL_CONFIGS:
        repo_id = model_key
        local_path = str(CKPT_DIR / model_key.replace("/", "_"))
        cfg = {"repo": repo_id, "path": local_path}
        MODEL_CONFIGS[model_key] = cfg
    else:
        cfg = MODEL_CONFIGS[model_key]

    device = get_device()
    if cfg["repo"]:
        mp = ensure_model_weights(cfg["path"], repo_id=cfg["repo"], verify=False)
    else:
        mp = Path(cfg["path"])
        if not mp.exists():
            raise gr.Error(f"模型目录不存在: {mp}，请先下载或指定 HuggingFace repo ID")

    os.environ["ZIMAGE_ATTENTION"] = attn_backend
    comp = load_from_local_dir(mp, device=device, dtype=DTYPE, compile=use_compile)
    comp["text_encoder"] = comp["text_encoder"].to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    set_attention_backend(attn_backend)
    _cache[cache_key] = comp
    return comp


def get_system_stats():
    gpu_txt = "N/A"
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used = total - free
        gpu_txt = f"{used/1024**3:.1f}/{total/1024**3:.1f} GB ({used/total*100:.0f}%)"
    cpu_pct = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    return f"""<span>🖥 GPU: <b>{gpu_txt}</b></span>
<span>🧠 CPU: <b>{cpu_pct}%</b></span>
<span>💾 内存: <b>{mem.used/1024**3:.1f}/{mem.total/1024**3:.1f} GB ({mem.percent}%)</b></span>"""


def load_history():
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    return []


def save_history(records):
    HISTORY_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def build_gallery(records):
    items = []
    for r in records:
        p = r["image_path"]
        if Path(p).exists():
            label = f"#{r['id']} | {r['params']['steps']}步 | CFG:{r['params']['guidance_scale']} | {r['elapsed']}s"
            items.append((p, label))
    return items


def format_stage_log(pct, desc, elapsed_list):
    stages = [
        ("📝 编码文本", 0.00, 0.04),
        ("🔄 去噪",     0.05, 0.84),
        ("🎨 解码 VAE", 0.85, 0.99),
        ("✅ 完成",     1.00, 1.00),
    ]
    lines = []
    for label, start, end in stages:
        if desc.startswith("📝"):
            active = True
        elif desc.startswith("🔄"):
            active = start <= pct
        elif desc.startswith("🎨"):
            active = pct >= 0.85
        elif desc.startswith("✅"):
            active = pct >= 1.0
        else:
            active = False

        if desc.startswith("🔄") and label == "🔄 去噪":
            step_info = desc.replace("🔄 ", "")
            lines.append(
                f'<span style="color:{"#4caf50" if pct>=end else "#555"};font-size:14px">{"✓" if pct>=end else "○"}</span>'
                f'<span style="color:{"#fff" if active else "#888"}">{label}</span>'
                f'<span style="color:#aaa;font-size:0.85rem;margin:0 4px">{step_info}</span>'
            )
        else:
            done = (label == "📝 编码文本" and pct >= 0.05) or \
                   (label == "🎨 解码 VAE" and pct >= 0.85) or \
                   (label == "✅ 完成" and pct >= 1.0)
            lines.append(
                f'<span style="color:{"#4caf50" if done else "#555"};font-size:14px">{"✓" if done else "○"}</span>'
                f'<span style="color:{"#fff" if active else "#888"}">{label}</span>'
            )
        lines.append('<span style="color:#444;margin:0 4px">|</span>')
    bar_pct = max(2, int(pct * 100))
    bar = (f'<div style="margin-top:6px;height:4px;background:#333;border-radius:2px;overflow:hidden">'
           f'<div style="width:{bar_pct}%;height:100%;background:linear-gradient(90deg,#4caf50,#8bc34a);border-radius:2px;transition:width 0.3s"></div></div>')
    elapsed = f'<span style="margin-left:auto;color:#aaa;font-size:0.85rem">{elapsed_list[-1] if elapsed_list else ""}s</span>'
    inner = "".join(lines) + elapsed
    return f'<div style="display:flex;align-items:center;gap:2px;flex-wrap:wrap;font-family:monospace;font-size:0.9rem">{inner}</div>{bar}'


CSS = """
.gradio-container{max-width:1500px!important;margin:auto!important}
.output-img img{border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,0.3)}
.stat-card{background:var(--background-fill-secondary);border-radius:8px;padding:0.5rem 1rem;display:flex;gap:1.5rem;flex-wrap:wrap}
.section-divider{color:#555;font-size:0.8rem;letter-spacing:1px;margin:4px 0 2px;border-bottom:1px solid var(--border-color-primary)}
"""

with gr.Blocks(title="Z-Image 文生图", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("""# ⚡ Z-Image 文生图  ·  阿里巴巴通义实验室
<small style="color:#888">输入提示词 → 选择模型 → 生成图片 · 支持取消/自定义模型/自动下载</small>""")

    history_state = gr.State([])
    stage_log_state = gr.State([])

    with gr.Row(equal_height=False):
        # ======== 左栏 ========
        with gr.Column(scale=2, min_width=400):
            prompt = gr.Textbox(label="📝 提示词", placeholder="描述你想要的图片…", lines=4)
            neg_prompt = gr.Textbox(label="➖ 负向提示词（可选）", lines=2)

            gr.Markdown("### 🤖 模型", elem_classes="section-divider")
            model_choice = gr.Dropdown(
                choices=list(MODEL_CONFIGS.keys()),
                value="Z-Image-Turbo" if "Z-Image-Turbo" in MODEL_CONFIGS else (list(MODEL_CONFIGS.keys())[0] if MODEL_CONFIGS else None),
                allow_custom_value=True,
                label="选择或输入 HuggingFace repo ID",
            )
            with gr.Row():
                refresh_models_btn = gr.Button("🔄 刷新本地", size="sm", scale=1)
                download_btn = gr.Button("📥 自动下载", size="sm", scale=1, variant="primary")
            model_status = gr.Markdown("")

            gr.Markdown("### ⚙️ 参数", elem_classes="section-divider")
            with gr.Row():
                width  = gr.Slider(512, 2048, 1024, step=64, label="宽度")
                height = gr.Slider(512, 2048, 1024, step=64, label="高度")
            with gr.Row():
                steps = gr.Slider(1, 100, 8, step=1, label="步数")
                guidance_scale = gr.Slider(0.0, 10.0, 0.0, step=0.5, label="CFG")
                seed = gr.Number(-1, label="种子", minimum=-1, precision=0)
            with gr.Row():
                use_compile = gr.Checkbox(value=False, label="torch.compile", info="首次慢，后续快")
                attn_backend = gr.Dropdown(
                    choices=ATTENTION_OPTIONS, value="native",
                    label="Attention", scale=2,
                )

            with gr.Accordion("⚙️ 高级参数", open=False):
                cfg_norm = gr.Checkbox(value=False, label="CFG 归一化（写实风格用 True）")
                with gr.Row():
                    cfg_trunc = gr.Slider(0.0, 1.0, 1.0, step=0.05, label="CFG 截断")
                    max_seq_len = gr.Slider(128, 1024, 512, step=64, label="最大序列长度")

            with gr.Row():
                gen_btn = gr.Button("✨ 生成", variant="primary", size="lg", scale=3)
                cancel_btn = gr.Button("⏹ 取消", variant="stop", size="lg", scale=1)
                clear_btn = gr.Button("🗑 清空", size="lg", scale=1)

            system_monitor = gr.HTML(value=get_system_stats(), elem_classes="stat-card")

        # ======== 右栏 ========
        with gr.Column(scale=3, min_width=500):
            output_image = gr.Image(label="生成结果", type="pil", height=520, elem_classes="output-img")
            stage_display = gr.HTML(
                value='<div style="color:#666;font-family:monospace;font-size:0.9rem;padding:4px 0">就绪</div>',
            )
            elapsed_display = gr.Markdown("")

            gr.Markdown("### 🖼 历史记录", elem_classes="section-divider")
            history_gallery = gr.Gallery(
                label=None, columns=4, height=240,
                object_fit="contain", allow_preview=False,
            )
            history_detail = gr.Markdown("点击上方缩略图查看参数详情")

    # ======== 底部日志 ========
    log_output = gr.Textbox(
        label="📋 生成日志", lines=5, max_lines=10,
        value="等待生成...", interactive=False,
    )
    save_path = gr.Textbox(label="保存路径", visible=False)

    # ── 事件 ──────────────────────────────────
    refresh_models_btn.click(fn=refresh_model_list, outputs=model_choice).then(
        fn=lambda: "✅ 本地模型已刷新", outputs=model_status,
    )

    def download_model(repo_id):
        if not repo_id or not repo_id.strip():
            return "❌ 请输入有效 repo ID"
        repo_id = repo_id.strip()
        local_path = str(CKPT_DIR / repo_id.replace("/", "_"))
        try:
            ensure_model_weights(local_path, repo_id=repo_id, verify=False)
            refresh_model_list()
            return f"✅ 下载完成: {repo_id}"
        except Exception as e:
            return f"❌ 下载失败: {e}"

    download_btn.click(fn=download_model, inputs=model_choice, outputs=model_status)

    def generate_image(
        prompt, neg_prompt, model_choice,
        width, height, steps, guidance_scale,
        cfg_norm, cfg_trunc, max_seq_len, seed,
        use_compile, attn_backend,
        history_state, stage_log_state,
        progress=gr.Progress(track_tqdm=True),
    ):
        if not prompt or not prompt.strip():
            raise gr.Error("请输入提示词")
        width = (width // VAE_SCALE) * VAE_SCALE
        height = (height // VAE_SCALE) * VAE_SCALE
        _cancel_event.clear()
        device = get_device()

        progress(0, desc="加载模型中...")
        log_lines = [f"[{time.strftime('%H:%M:%S')}] 加载模型: {model_choice}"]
        try:
            comp = load_model(model_choice, use_compile, attn_backend)
            log_lines.append(f"[{time.strftime('%H:%M:%S')}] 模型就绪 ✅")
        except Exception as e:
            raise gr.Error(f"模型加载失败: {e}")

        if _cancel_event.is_set():
            raise gr.CancelledError()

        use_cfg = guidance_scale > 1.0
        gen = torch.Generator(device).manual_seed(seed) if seed >= 0 else None

        def on_progress(pct, desc):
            if _cancel_event.is_set():
                raise gr.CancelledError()
            elapsed = time.time() - t0
            log_lines.append(f"[{time.strftime('%H:%M:%S')}] {desc} ({elapsed:.1f}s)")
            progress(pct, desc=desc)

        t0 = time.time()
        log_lines.append(f"[{time.strftime('%H:%M:%S')}] 开始生成 ({width}x{height}, {steps}步, CFG={guidance_scale})")
        progress(0.01, desc="📝 编码文本...")
        try:
            images = generate(
                prompt=prompt,
                negative_prompt=neg_prompt if use_cfg else None,
                **comp,
                height=height, width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=gen,
                cfg_normalization=cfg_norm,
                cfg_truncation=cfg_trunc if cfg_trunc > 0 else None,
                max_sequence_length=max_seq_len,
                _progress_callback=on_progress,
            )
        except gr.CancelledError:
            raise
        except Exception as e:
            raise gr.Error(f"生成失败: {e}")

        if _cancel_event.is_set():
            raise gr.CancelledError()

        elapsed = time.time() - t0
        ts = time.strftime("%Y%m%d_%H%M%S")
        img = images[0]
        save_path = str(OUTPUT_DIR / f"zimage_{ts}.png")
        img.save(save_path)
        log_lines.append(f"[{time.strftime('%H:%M:%S')}] 已保存: {save_path}")

        record = {
            "id": len(history_state),
            "timestamp": ts,
            "image_path": save_path,
            "prompt": prompt,
            "negative_prompt": neg_prompt or "",
            "params": {
                "model": model_choice,
                "width": width, "height": height,
                "steps": steps, "guidance_scale": guidance_scale,
                "seed": seed, "cfg_normalization": cfg_norm,
                "cfg_truncation": cfg_trunc, "max_seq_len": max_seq_len,
                "compile": use_compile, "attn_backend": attn_backend,
            },
            "elapsed": round(elapsed, 1),
        }
        history_state = [record] + history_state
        save_history(history_state)
        gallery = build_gallery(history_state)
        stats = get_system_stats()

        stage_html = format_stage_log(1.0, "✅ 完成", [round(elapsed, 1)])
        detail_md = (
            f"**提示词:** {prompt[:150]}{'…' if len(prompt)>150 else ''}\n\n"
            f"**负向提示:** {neg_prompt[:100] or '(无)'}\n\n"
            f"**模型:** {model_choice} | **尺寸:** {width}×{height} | "
            f"**步数:** {steps} | **CFG:** {guidance_scale} | **种子:** {seed}\n\n"
            f"⏱ **耗时:** {elapsed:.1f}s | "
            f"**编译:** {'✅' if use_compile else '❌'} | **Attention:** {attn_backend}"
        )

        return (
            img, stage_html,
            f"⏱ 总耗时: **{elapsed:.1f}秒** | Steps: {steps} | CFG: {guidance_scale} | Seed: {seed}",
            history_state, gallery, detail_md,
            stats, save_path,
            "\n".join(log_lines[-20:]),
        )

    def select_history(evt: gr.SelectData, history_state):
        if not history_state:
            return None, "", ""
        idx = evt.index
        r = history_state[idx]
        p = r["params"]
        detail = (
            f"**提示词:** {r['prompt']}\n\n"
            f"**负向提示:** {r['negative_prompt'] or '(无)'}\n\n"
            f"**模型:** {p['model']} | **尺寸:** {p['width']}×{p['height']}\n"
            f"**步数:** {p['steps']} | **CFG:** {p['guidance_scale']}\n"
            f"**种子:** {p['seed']} | **CFG归一:** {p['cfg_normalization']}\n"
            f"**截断:** {p['cfg_truncation']} | **序列长:** {p['max_seq_len']}\n"
            f"**编译:** {p.get('compile', 'N/A')} | **Attention:** {p.get('attn_backend', 'N/A')}\n\n"
            f"⏱ **耗时:** {r.get('elapsed','?')}秒 | 🕐 **时间:** {r['timestamp']}"
        )
        image_path = r["image_path"]
        return (
            image_path if Path(image_path).exists() else None,
            r["prompt"],
            detail,
        )

    # ── 事件绑定 ──────────────────────────────
    gen_event = gen_btn.click(
        fn=generate_image,
        inputs=[
            prompt, neg_prompt, model_choice,
            width, height, steps, guidance_scale,
            cfg_norm, cfg_trunc, max_seq_len, seed,
            use_compile, attn_backend,
            history_state, stage_log_state,
        ],
        outputs=[
            output_image, stage_display, elapsed_display,
            history_state, history_gallery, history_detail,
            system_monitor, save_path, log_output,
        ],
    )

    cancel_btn.click(fn=lambda: _cancel_event.set(), cancels=[gen_event])

    history_gallery.select(
        fn=select_history,
        inputs=history_state,
        outputs=[output_image, prompt, history_detail],
    )

    clear_btn.click(
        fn=lambda: (
            "", "", "Z-Image-Turbo" if "Z-Image-Turbo" in MODEL_CONFIGS else (list(MODEL_CONFIGS.keys())[0] if MODEL_CONFIGS else ""),
            1024, 1024, 8, 0.0, -1,
            False, 1.0, 512,
            False, "native",
            None,
            '<div style="color:#666;font-family:monospace;font-size:0.9rem;padding:4px 0">就绪</div>',
            "",
            [], [],
            "点击上方缩略图查看参数详情",
            get_system_stats(), "", "等待生成...",
        ),
        outputs=[
            prompt, neg_prompt, model_choice,
            width, height, steps, guidance_scale, seed,
            cfg_norm, cfg_trunc, max_seq_len,
            use_compile, attn_backend,
            output_image, stage_display, elapsed_display,
            history_state, history_gallery, history_detail,
            system_monitor, save_path, log_output,
        ],
    )

    demo.load(
        fn=lambda: (load_history(), build_gallery(load_history()), get_system_stats()),
        outputs=[history_state, history_gallery, system_monitor],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
