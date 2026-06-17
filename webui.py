"""Z-Image Web UI — 完整版（模型库·介绍·自动下载·回档）"""
import json, os, time, warnings, threading
from pathlib import Path
from collections import OrderedDict

import torch
import gradio as gr
import psutil

warnings.filterwarnings("ignore")
from utils import ensure_model_weights, load_from_local_dir, set_attention_backend, load_sharded_safetensors
from zimage import generate
from zimage.transformer import ZImageTransformer2DModel

os.environ["ZIMAGE_ATTENTION"] = os.environ.get("ZIMAGE_ATTENTION", "native")
DTYPE = torch.bfloat16
VAE_SCALE = 16

HISTORY_FILE = Path("outputs/history.json")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
CKPT_DIR = Path("ckpts")
CKPT_DIR.mkdir(exist_ok=True)
COMMUNITY_DIR = CKPT_DIR / "community"
COMMUNITY_DIR.mkdir(exist_ok=True)

ATTENTION_OPTIONS = [
    ("native（默认）", "native"),
    ("native_flash", "_native_flash"),
    ("Flash Attn 2", "flash"),
    ("Flash Attn 3", "_flash_3"),
]

_cache = {}
_device = None
_cancel_event = threading.Event()

# ── 模型库 ──────────────────────────────────────
MODEL_REGISTRY = OrderedDict()

def _reg(key, display, group, repo, path, desc, steps, cfg, dtype, source_url=None, file_name=None):
    MODEL_REGISTRY[key] = {
        "display": display, "group": group,
        "repo": repo, "path": path,
        "desc": desc, "steps": steps, "cfg": cfg,
        "dtype": dtype,
        "source_url": source_url, "file_name": file_name,
    }

# ── 官方模型（HuggingFace 自动下载）─────────────
_reg("Z-Image-Turbo",
     "🏛 Z-Image-Turbo", "官方",
     "Tongyi-MAI/Z-Image-Turbo", "ckpts/Z-Image-Turbo",
     "阿里通义实验室官方发布的**蒸馏快速版本**，仅需 **8 步**推理即可生成高质量图片。"
     "擅长写实照片风格、精准的中英文文字渲染。在 Artificial Analysis 排行榜上位列开源模型 **第 1 名**。\n\n"
     "⚡ 步数: 8  |  CFG: 0  |  显存: ≥16GB  |  ✓ 已下载",
     8, 0.0, "auto")

_reg("Z-Image",
     "🏛 Z-Image", "官方",
     "Tongyi-MAI/Z-Image", "ckpts/Z-Image",
     "阿里通义实验室官方发布的**基础模型**，需 **50 步**推理。支持丰富的艺术风格、"
     "多样的构图和精确的负向提示。适合创意生成和 LoRA 微调。\n\n"
     "⚡ 步数: 28-50  |  CFG: 3-5  |  显存: ≥16GB",
     50, 4.0, "auto")

_reg("Comfy-Org Z-Image-Turbo",
     "🏛 Z-Image-Turbo (社区镜像)", "官方",
     "Comfy-Org/z_image_turbo", "ckpts/Comfy-Org_z_image_turbo",
     "ComfyUI 团队维护的官方模型镜像仓库，内容与官方一致，适用于 ComfyUI 工作流。\n\n"
     "⚡ 步数: 8  |  CFG: 0  |  显存: ≥16GB",
     8, 0.0, "auto")

# ── 社区模型（HuggingFace 自动下载）─────────────
_reg("Z-Image-De-Turbo",
     "🔧 Z-Image De-Turbo（可训练）", "社区",
     "ostris/Z-Image-De-Turbo", "ckpts/Z-Image-De-Turbo",
     "社区制作的**去蒸馏版本**，移除了 Turbo 的蒸馏加速，恢复为完整的 50 步扩散过程。"
     "适合用于 **LoRA 训练** 和自定义微调，训练后无需特殊适配器即可推理。\n\n"
     "⚡ 步数: 20-30  |  CFG: 2-3  |  显存: ≥16GB",
     25, 2.5, "auto")

_reg("Z-Image-Turbo-FP8",
     "🔧 Z-Image-Turbo FP8（低显存）", "社区",
     "Kijai/Z-Image-Turbo-fp8", "ckpts/Z-Image-Turbo-FP8",
     "使用 FP8 量化压缩的 Turbo 版本，模型体积从 12GB 降至约 **6GB**，显存需求大幅降低。"
     "适合 **6-8GB 显存**的显卡（如 RTX 3060/4060）。画质损失极小。\n\n"
     "⚡ 步数: 8  |  CFG: 0  |  显存: ≥8GB",
     8, 0.0, "auto")

# ── 社区模型（CivitAI 单文件，需手动下载）───────
_reg("Juggernaut-Z",
     "⭐ Juggernaut Z（写实微调）", "社区热门",
     None, "ckpts/community/juggernaut_z",
     "由 RunDiffusion 团队出品的**写实风格微调**模型。Juggernaut 系列是 Stable Diffusion 时代最知名的写实模型品牌，"
     "本次基于 Z-Image Base 微调，继承了优秀的写实照片质感、自然皮肤纹理和光影效果。\n\n"
     "📥 从 CivitAI 下载: civitai.com/models/2600510\n"
     "⚡ 步数: 8-15  |  CFG: 1-2  |  显存: ≥12GB",
     10, 1.5, "manual",
     "https://civitai.com/models/2600510/juggernaut-z",
     "juggernaut_z.safetensors")

_reg("unStable-Revolution-ZIT",
     "⭐ unStable Revolution ZIT（写实+NSFW）", "社区热门",
     None, "ckpts/community/unstable_revolution_zit",
     "社区热门微调模型，基于 Z-Image Turbo 进行写实增强，显著改善了人物皮肤质感和细节，"
     "支持 NSFW 内容生成。在 CivitAI 上有 **3.8 万**下载量。\n\n"
     "📥 从 CivitAI 下载: civitai.com/models/2193942\n"
     "⚡ 步数: 8-10  |  CFG: 1-2  |  显存: ≥12GB",
     8, 1.0, "manual",
     "https://civitai.com/models/2193942/unstable-revolution-zit",
     "unstable_revolution_zit.safetensors")

_reg("ZOMG",
     "⭐ ZOMG（风格融合）", "社区热门",
     None, "ckpts/community/zomg",
     "融合多种风格 LoRA 的精调模型，解决了原始模型皮肤斑驳的问题，生成更美观、更稳定的图片。"
     "同时支持 SFW 和 NSFW 内容。\n\n"
     "📥 从 CivitAI 下载: civitai.com/models/2314752\n"
     "⚡ 步数: 8-12  |  CFG: 1-2  |  显存: ≥12GB",
     10, 1.5, "manual",
     "https://civitai.com/models/2314752/zomg-z-image-turbo-sfw-nsfw",
     "zomg.safetensors")

_reg("Z-Image-Turbo-Clear",
     "⭐ Z-Image-Turbo Clear（细节增强）", "社区热门",
     None, "ckpts/community/z_image_clear",
     "针对 Z-Image-Turbo 的**细节增强**微调版，提升了原始模型的细节表现力，"
     "适合对画质有更高要求的场景。配有专用 VAE。\n\n"
     "📥 从 CivitAI 下载: civitai.com/models/2197598\n"
     "⚡ 步数: 9  |  CFG: 1  |  显存: ≥12GB",
     9, 1.0, "manual",
     "https://civitai.com/models/2197598/z-image-turboclear",
     "z_image_clear.safetensors")

_reg("Swift-ZIT",
     "⭐ SWIFT（快速+细节）", "社区热门",
     None, "ckpts/community/swift_zit",
     "注重生成速度同时提升细节表现的模型。改善了 Z-Image Turbo 的女性面部多样性。"
     "GGUF 格式，加载更快。\n\n"
     "📥 从 CivitAI 下载: civitai.com/models/2534952\n"
     "⚡ 步数: 8-10  |  CFG: 1  |  显存: ≥8GB",
     8, 1.0, "manual",
     "https://civitai.com/models/2534952/swift-fast-and-detailed-zit-model",
     "swift_zit.safetensors")

# 下拉选项
MODEL_CHOICES = [v["display"] for v in MODEL_REGISTRY.values()]
MODEL_CHOICES_DICT = {v["display"]: k for k, v in MODEL_REGISTRY.items()}


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
    """检测本地已下载的模型，返回已下载的 key 集合"""
    downloaded = set()
    for key, cfg in MODEL_REGISTRY.items():
        p = Path(cfg["path"])
        if cfg["dtype"] == "auto":
            # 检查完整目录结构
            if (p / "transformer").is_dir():
                downloaded.add(key)
        else:
            # 检查单文件
            if p.is_dir():
                for f in p.iterdir():
                    if f.suffix in (".safetensors", ".bin", ".pt"):
                        downloaded.add(key)
                        break
    return downloaded


def get_model_info(model_display_name):
    """根据下拉显示名获取模型详情"""
    if model_display_name not in MODEL_CHOICES_DICT:
        return None
    key = MODEL_CHOICES_DICT[model_display_name]
    return MODEL_REGISTRY.get(key)


def build_model_info_html(model_display_name):
    """构建模型介绍 HTML 卡片"""
    info = get_model_info(model_display_name)
    if info is None:
        return "请选择一个模型"
    downloaded = scan_local_models()
    key = MODEL_CHOICES_DICT[model_display_name]
    is_downloaded = key in downloaded
    status_icon = "✅ 已下载" if is_downloaded else "⬇️ 未下载"
    if info["dtype"] == "auto":
        download_hint = "选择后点击「自动下载」即可从 HuggingFace 拉取"
    else:
        download_hint = (f'需从 CivitAI 手动下载 → <a href="{info["source_url"]}" target="_blank">点击前往</a><br>'
                         f'下载后将文件放入: <code>{info["path"]}</code>')

    cfg_note = f'CFG 建议设为 <b>{info["cfg"]}</b>' if info["cfg"] > 0 else '<b>CFG=0</b>（Turbo 模型不支持 CFG）'
    return f"""
    <div style="background:var(--background-fill-secondary);border-radius:10px;padding:12px 16px;font-size:0.9rem;line-height:1.6">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <span style="font-weight:bold;font-size:1rem">{info["display"]}</span>
        <span style="font-size:0.85rem;padding:2px 10px;border-radius:12px;background:{"#4caf50" if is_downloaded else "#666"};color:#fff">{status_icon}</span>
      </div>
      <div style="color:#ccc">{info["desc"]}</div>
      <div style="margin-top:8px;padding-top:8px;border-top:1px solid #444;color:#aaa;font-size:0.85rem">
        {cfg_note}  |  步数: <b>{info["steps"]}</b>  |  {download_hint}
      </div>
    </div>"""


def load_model(model_key: str, use_compile: bool = False, attn_backend: str = "native"):
    """加载模型（支持 auto 和 manual 两种类型）"""
    cache_key = f"{model_key}_c{use_compile}_a{attn_backend}"
    if cache_key in _cache:
        return _cache[cache_key]

    if model_key not in MODEL_REGISTRY:
        raise gr.Error(f"未知模型: {model_key}")

    cfg = MODEL_REGISTRY[model_key]
    device = get_device()
    os.environ["ZIMAGE_ATTENTION"] = attn_backend

    if cfg["dtype"] == "auto":
        # HuggingFace 完整目录结构
        if cfg["repo"]:
            mp = ensure_model_weights(cfg["path"], repo_id=cfg["repo"], verify=False)
        else:
            mp = Path(cfg["path"])
            if not mp.exists():
                raise gr.Error(f"模型目录不存在: {mp}")
        comp = load_from_local_dir(mp, device=device, dtype=DTYPE, compile=use_compile)
    else:
        # 社区单文件模型
        comp = _load_single_file_model(model_key, device, use_compile)

    comp["text_encoder"] = comp["text_encoder"].to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    set_attention_backend(attn_backend)
    _cache[cache_key] = comp
    return comp


def _load_single_file_model(model_key: str, device: str, use_compile: bool):
    """加载社区单文件 safetensors 模型"""
    cfg = MODEL_REGISTRY[model_key]
    model_dir = Path(cfg["path"])

    # 找 safetensors 文件
    safetensors_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin")) + list(model_dir.glob("*.pt"))
    if not safetensors_files:
        raise gr.Error(f"在 {model_dir} 中未找到模型文件。请先从 CivitAI 下载并放入该目录。\n"
                       f"下载地址: {cfg['source_url']}")

    # 使用 Z-Image-Turbo 作为基础框架（复用 VAE、text_encoder、scheduler）
    base_path = CKPT_DIR / "Z-Image-Turbo"
    if not base_path.exists():
        # 尝试自动下载
        ensure_model_weights(str(base_path), repo_id="Tongyi-MAI/Z-Image-Turbo", verify=False)

    # 先加载基础组件（不含 transformer）
    base_comp = load_from_local_dir(base_path, device=device, dtype=DTYPE, compile=False)

    # 替换 transformer 权重
    from config import (
        DEFAULT_TRANSFORMER_PATCH_SIZE, DEFAULT_TRANSFORMER_F_PATCH_SIZE,
        DEFAULT_TRANSFORMER_IN_CHANNELS, DEFAULT_TRANSFORMER_DIM,
        DEFAULT_TRANSFORMER_N_LAYERS, DEFAULT_TRANSFORMER_N_REFINER_LAYERS,
        DEFAULT_TRANSFORMER_N_HEADS, DEFAULT_TRANSFORMER_N_KV_HEADS,
        DEFAULT_TRANSFORMER_NORM_EPS, DEFAULT_TRANSFORMER_QK_NORM,
        DEFAULT_TRANSFORMER_CAP_FEAT_DIM, ROPE_THETA,
        DEFAULT_TRANSFORMER_T_SCALE, ROPE_AXES_DIMS, ROPE_AXES_LENS,
    )

    # 从 base 模型的 transformer config 创建新 transformer
    transformer_dir = base_path / "transformer"
    import json
    config = json.loads((transformer_dir / "config.json").read_text())

    with torch.device("meta"):
        transformer = ZImageTransformer2DModel(
            all_patch_size=tuple(config.get("all_patch_size", DEFAULT_TRANSFORMER_PATCH_SIZE)),
            all_f_patch_size=tuple(config.get("all_f_patch_size", DEFAULT_TRANSFORMER_F_PATCH_SIZE)),
            in_channels=config.get("in_channels", DEFAULT_TRANSFORMER_IN_CHANNELS),
            dim=config.get("dim", DEFAULT_TRANSFORMER_DIM),
            n_layers=config.get("n_layers", DEFAULT_TRANSFORMER_N_LAYERS),
            n_refiner_layers=config.get("n_refiner_layers", DEFAULT_TRANSFORMER_N_REFINER_LAYERS),
            n_heads=config.get("n_heads", DEFAULT_TRANSFORMER_N_HEADS),
            n_kv_heads=config.get("n_kv_heads", DEFAULT_TRANSFORMER_N_KV_HEADS),
            norm_eps=config.get("norm_eps", DEFAULT_TRANSFORMER_NORM_EPS),
            qk_norm=config.get("qk_norm", DEFAULT_TRANSFORMER_QK_NORM),
            cap_feat_dim=config.get("cap_feat_dim", DEFAULT_TRANSFORMER_CAP_FEAT_DIM),
            rope_theta=config.get("rope_theta", ROPE_THETA),
            t_scale=config.get("t_scale", DEFAULT_TRANSFORMER_T_SCALE),
            axes_dims=config.get("axes_dims", ROPE_AXES_DIMS),
            axes_lens=config.get("axes_lens", ROPE_AXES_LENS),
        ).to(DTYPE)

    # 加载 safetensors 权重
    state_dict = load_sharded_safetensors(model_dir, device="cpu", dtype=DTYPE)
    transformer.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict
    transformer = transformer.to(device)
    transformer.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if use_compile:
        transformer = torch.compile(transformer)

    base_comp["transformer"] = transformer
    return base_comp


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


CSS = """
.gradio-container{max-width:1500px!important;margin:auto!important}
.output-img img{border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,0.3)}
.stat-card{background:var(--background-fill-secondary);border-radius:8px;padding:0.5rem 1rem;display:flex;gap:1.5rem;flex-wrap:wrap}
.section-divider{color:#555;font-size:0.8rem;letter-spacing:1px;margin:4px 0 2px;border-bottom:1px solid var(--border-color-primary)}
"""

with gr.Blocks(title="Z-Image 文生图", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("""# ⚡ Z-Image 文生图  ·  阿里巴巴通义实验室
<small style="color:#888">输入提示词 → 选择模型 → 生成图片 · 支持取消/自定义模型/自动下载/社区模型</small>""")

    history_state = gr.State([])
    stage_log_state = gr.State("")
    model_info_state = gr.State("")

    with gr.Row(equal_height=False):
        with gr.Column(scale=2, min_width=400):
            prompt = gr.Textbox(label="📝 提示词", placeholder="描述你想要的图片…", lines=4)
            neg_prompt = gr.Textbox(label="➖ 负向提示词（可选）", lines=2)

            gr.Markdown("### 🤖 模型", elem_classes="section-divider")
            model_choice = gr.Dropdown(
                choices=MODEL_CHOICES,
                value=MODEL_CHOICES[0],
                label="选择模型",
            )
            model_info = gr.HTML(value=build_model_info_html(MODEL_CHOICES[0]))
            with gr.Row():
                refresh_models_btn = gr.Button("🔄 刷新状态", size="sm", scale=1)
                download_btn = gr.Button("📥 自动下载", size="sm", scale=1, variant="primary")

            gr.Markdown("### ⚙️ 参数", elem_classes="section-divider")
            with gr.Row():
                width  = gr.Slider(512, 2048, 1920, step=64, label="宽度")
                height = gr.Slider(512, 2048, 1080, step=64, label="高度")
            with gr.Row():
                steps = gr.Slider(1, 100, 8, step=1, label="步数", info="去噪步数，越高细节越丰富，但耗时越长")
                guidance_scale = gr.Slider(0.0, 10.0, 0.0, step=0.5, label="CFG", info="提示词引导强度，越大越贴合提示词")
                seed = gr.Number(-1, label="种子", minimum=-1, precision=0, info="-1=随机，固定值可复现相同结果")
            with gr.Row():
                use_compile = gr.Checkbox(value=False, label="torch.compile", info="编译模型加速推理，首次较慢后续快")
                attn_backend = gr.Dropdown(
                    choices=ATTENTION_OPTIONS, value="native",
                    label="Attention", scale=2, info="注意力机制后端，Flash Attn 2 可加速大图生成",
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

        with gr.Column(scale=3, min_width=500):
            output_image = gr.Image(label="生成结果", type="pil", height=520, elem_classes="output-img")
            with gr.Row():
                stage_display = gr.HTML(
                    value='<div style="color:#888;font-family:monospace;font-size:0.85rem;padding:6px 12px;background:var(--background-fill-secondary);border-radius:6px">就绪</div>',
                    scale=2,
                )
                system_monitor = gr.HTML(value=get_system_stats(), elem_classes="stat-card", scale=1)
            elapsed_display = gr.Markdown("")

            gr.Markdown("### 🖼 历史记录", elem_classes="section-divider")
            history_gallery = gr.Gallery(
                label=None, columns=4, height=240,
                object_fit="contain", allow_preview=False,
            )
            history_detail = gr.Markdown("点击上方缩略图查看参数详情")

    log_output = gr.Textbox(
        label="📋 生成日志", lines=5, max_lines=10,
        value="等待生成...", interactive=False,
    )
    save_path = gr.Textbox(label="保存路径", visible=False)

    # ── 事件 ──────────────────────────────────

    # 模型选中 → 更新介绍卡片
    def on_model_select(display_name):
        if display_name is None:
            return build_model_info_html(MODEL_CHOICES[0])
        return build_model_info_html(display_name)

    model_choice.change(fn=on_model_select, inputs=model_choice, outputs=model_info)

    # 刷新状态
    def refresh_status():
        downloaded = scan_local_models()
        return f"已扫描本地模型，共 {len(downloaded)} 个已下载"

    refresh_models_btn.click(fn=refresh_status, outputs=model_info)

    # 自动下载
    def download_selected(display_name):
        if display_name not in MODEL_CHOICES_DICT:
            return "❌ 无效的模型选择"
        key = MODEL_CHOICES_DICT[display_name]
        cfg = MODEL_REGISTRY.get(key)
        if cfg is None:
            return "❌ 模型未找到"
        if cfg["dtype"] != "auto":
            return (f'⚠️ 该模型为社区单文件版本，请手动从 CivitAI 下载:\n'
                    f'{cfg["source_url"]}\n\n放入目录: {cfg["path"]}')

        try:
            ensure_model_weights(cfg["path"], repo_id=cfg["repo"], verify=False)
            return f"✅ 下载完成: {cfg['display']}"
        except Exception as e:
            return f"❌ 下载失败: {e}"

    download_btn.click(fn=download_selected, inputs=model_choice, outputs=model_info)

    # 生成（generator 实时更新 UI）
    def generate_image(
        prompt, neg_prompt, model_display_name,
        width, height, steps, guidance_scale,
        cfg_norm, cfg_trunc, max_seq_len, seed,
        use_compile, attn_backend,
        history_state, stage_log_state,
        progress=gr.Progress(track_tqdm=True),
    ):
        if not prompt or not prompt.strip():
            raise gr.Error("请输入提示词")

        if model_display_name not in MODEL_CHOICES_DICT:
            raise gr.Error(f"未知模型: {model_display_name}")
        model_key = MODEL_CHOICES_DICT[model_display_name]

        width = (width // VAE_SCALE) * VAE_SCALE
        height = (height // VAE_SCALE) * VAE_SCALE
        _cancel_event.clear()
        device = get_device()
        log_lines = [f"[{time.strftime('%H:%M:%S')}] 加载模型: {model_display_name}"]

        def _yield(stage_text, pct=None):
            stats = get_system_stats()
            return (
                None,
                f'<div style="display:flex;align-items:center;gap:12px;padding:6px 12px;background:var(--background-fill-secondary);border-radius:6px;color:#888;font-family:monospace;font-size:0.85rem">'
                f'<span>⏳ {stage_text}</span><span style="margin-left:auto;color:#aaa">{getattr(_yield, "elapsed", "0.0")}s</span></div>',
                f"⏱ 进行中: {stage_text}",
                history_state, gr.skip(), gr.skip(),
                stats, "", "\n".join(log_lines[-20:]),
            )

        try:
            comp = load_model(model_key, use_compile, attn_backend)
            log_lines.append(f"[{time.strftime('%H:%M:%S')}] 模型就绪 ✅")
            yield _yield("模型就绪 ✅")
        except Exception as e:
            raise gr.Error(f"模型加载失败: {e}")

        if _cancel_event.is_set():
            raise gr.CancelledError()

        model_cfg = MODEL_REGISTRY[model_key]
        if steps == 8 and model_cfg["steps"] != 8 and steps == model_cfg["steps"]:
            pass
        if guidance_scale == 0.0 and model_cfg["cfg"] > 0:
            guidance_scale = model_cfg["cfg"]

        use_cfg = guidance_scale > 1.0
        gen = torch.Generator(device).manual_seed(seed) if seed >= 0 else None

        def on_progress(pct, desc):
            if _cancel_event.is_set():
                raise gr.CancelledError()
            _yield.elapsed = f"{time.time() - t0:.1f}"
            log_lines.append(f"[{time.strftime('%H:%M:%S')}] {desc} ({_yield.elapsed}s)")
            progress(pct, desc=desc)

        t0 = time.time()
        _yield.elapsed = "0.0"
        log_lines.append(f"[{time.strftime('%H:%M:%S')}] 开始 ({width}x{height}, {steps}步, CFG={guidance_scale})")
        yield _yield("编码文本...")
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
                "model": model_display_name,
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

        detail_md = (
            f"**提示词:** {prompt[:150]}{'…' if len(prompt)>150 else ''}\n\n"
            f"**负向提示:** {neg_prompt[:100] or '(无)'}\n\n"
            f"**模型:** {model_display_name} | **尺寸:** {width}×{height} | "
            f"**步数:** {steps} | **CFG:** {guidance_scale} | **种子:** {seed}\n\n"
            f"⏱ **耗时:** {elapsed:.1f}s | "
            f"**编译:** {'✅' if use_compile else '❌'} | **Attention:** {attn_backend}"
        )

        yield (
            img,
            f'<div style="display:flex;align-items:center;gap:12px;padding:6px 12px;background:var(--background-fill-secondary);border-radius:6px;color:#4caf50;font-family:monospace;font-size:0.85rem">'
            f'<span>✅ 完成</span><span style="margin-left:auto;color:#aaa">{elapsed:.1f}s</span></div>',
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
        concurrency_limit=1,
    )

    cancel_btn.click(fn=lambda: _cancel_event.set(), cancels=[gen_event])

    history_gallery.select(
        fn=select_history,
        inputs=history_state,
        outputs=[output_image, prompt, history_detail],
    )

    clear_btn.click(
        fn=lambda: (
            "", "", MODEL_CHOICES[0],
            1920, 1080, 8, 0.0, -1,
            False, 1.0, 512,
            False, "native",
            None,
            '<div style="color:#888;font-family:monospace;font-size:0.85rem;padding:6px 12px;background:var(--background-fill-secondary);border-radius:6px">就绪</div>',
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
    demo.queue(default_concurrency_limit=3)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
