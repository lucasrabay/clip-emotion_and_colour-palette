import io, os, json, sys
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans

# ========= CONFIG PADRÃO =========
CKPT_PATH = "models/clip_mlp_emotion.pt" 
PREFER_BACKBONE = None                    
DEFAULT_TTA = True
DEFAULT_K_COLORS = 6

# tenta importar open_clip; se faltar, avisa na UI
try:
    import open_clip
except Exception:
    st.error("Faltou `open_clip_torch`. Instale com: `pip install open_clip_torch`")
    raise

# ===================== Utils =====================
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(int(x) for x in rgb)

def _open_image_safely(raw_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(raw_bytes))
    img = ImageOps.exif_transpose(img)  # corrige orientação EXIF
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")
    return img

def extract_palette_kmeans(
    pil_image: Image.Image,
    k: int = 6,
    use_global: bool = False,
    kmeans_global=None
) -> Tuple[np.ndarray, np.ndarray]:
    """Retorna (palette[K,3] uint8, weights[K])."""
    arr = np.array(pil_image.convert("RGB")).reshape(-1, 3).astype(np.float32)
    if use_global and (kmeans_global is not None):
        labels = kmeans_global.predict(arr)
        centers = kmeans_global.cluster_centers_
        counts = np.bincount(labels, minlength=centers.shape[0])
        order = np.argsort(-counts)
        palette = centers[order][:k]
        weights = (counts[order] / max(1, counts.sum()))[:k]
    else:
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = km.fit_predict(arr)
        centers = km.cluster_centers_
        counts = np.bincount(labels)
        order = np.argsort(-counts)
        palette = centers[order]
        weights = counts[order] / max(1, counts.sum())
    palette = np.clip(palette, 0, 255).astype(np.uint8)
    return palette, weights

# ===================== Modelo (CLIP + Head) =====================
DIM_TO_MODELS = {
    512: [('ViT-B-32', 'openai'), ('ViT-B-16', 'openai')],
    768: [('ViT-L-14', 'openai'), ('ViT-L-14-336', 'openai')],
}

@st.cache_resource(show_spinner=False)
def load_clip_by_dim(target_dim: int, prefer_name: Optional[str] = None):
    """Carrega o CLIP com o output_dim desejado.
       Se prefer_name vier (ex. 'ViT-L-14'), tenta primeiro ele."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tried = []
    if prefer_name:
        model, _, preprocess = open_clip.create_model_and_transforms(prefer_name, pretrained='openai')
        if getattr(model.visual, 'output_dim', None) == target_dim:
            return model.to(device).eval(), preprocess, device, prefer_name
        tried.append(prefer_name)
    for name, pretrained in DIM_TO_MODELS.get(target_dim, []):
        if name in tried:
            continue
        model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
        if getattr(model.visual, 'output_dim', None) == target_dim:
            return model.to(device).eval(), preprocess, device, name
    raise ValueError(
        f"Não encontrei um backbone CLIP com output_dim={target_dim}. "
        f"Reextraia/treine ou ajuste manualmente o backbone."
    )

def build_mlp_head(feat_dim: int, n_classes: int) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(feat_dim, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(256, n_classes)
    )

@st.cache_resource(show_spinner=False)
def load_head_from_path(ckpt_path: str, device: str = 'cpu'):
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt['classes']
    feat_dim = ckpt['dim']
    mlp = build_mlp_head(feat_dim, len(classes)).to(device).eval()
    mlp.load_state_dict(ckpt['state_dict'])
    return mlp, classes, feat_dim

def clip_feature(pil_img: Image.Image, model, preprocess, device: str):
    x = preprocess(pil_img.convert("RGB")).unsqueeze(0).to(device)
    f = model.encode_image(x)
    return f / f.norm(dim=-1, keepdim=True)

@torch.inference_mode()
def predict_head(
    pil_img: Image.Image,
    mlp: torch.nn.Module,
    model,
    preprocess,
    classes: List[str],
    device: str,
    tta: bool = True,
    topk: int = 3
):
    feats = [clip_feature(pil_img, model, preprocess, device)]
    if tta:
        feats.append(clip_feature(pil_img.transpose(Image.FLIP_LEFT_RIGHT), model, preprocess, device))
    f = torch.stack(feats).mean(0)  # [1, D]
    logits = mlp(f)
    probs = F.softmax(logits, dim=-1).squeeze(0)  # [C]
    k = min(topk, len(classes))
    topv, topi = torch.topk(probs, k)
    top = [(classes[int(i)], float(v)) for v, i in zip(topv.tolist(), topi.tolist())]
    return top

# ===================== UI =====================
st.set_page_config(page_title="Paleta + Emoção (Head MLP)", page_icon="🎨", layout="wide")
st.title("🎨 Paleta de Cores + Emoção (Head MLP sobre CLIP)")

# Carrega head fixo + CLIP compatível
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if not os.path.exists(CKPT_PATH):
    st.error(f"Checkpoint não encontrado em '{CKPT_PATH}'. Coloque seu .pt nesse caminho e recarregue.")
    st.stop()

try:
    mlp, classes, feat_dim = load_head_from_path(CKPT_PATH, device=device)
    clip_model, preprocess, device, used_backbone = load_clip_by_dim(feat_dim, prefer_name=PREFER_BACKBONE)
except Exception as e:
    st.error(f"Falha ao carregar modelo: {e}")
    st.stop()

# Header de informações
st.markdown(
    f"**Checkpoint:** `{CKPT_PATH}`  •  **Backbone CLIP:** `{used_backbone}`  •  "
    f"**Device:** `{'GPU' if device=='cuda' else 'CPU'}`  •  **TTA:** `{DEFAULT_TTA}`  •  **k:** `{DEFAULT_K_COLORS}`"
)
st.caption("Classes: " + ", ".join(classes))
st.divider()

# Upload de imagens (principal)
imgs_up = st.file_uploader("Arraste suas imagens aqui", type=["png","jpg","jpeg","webp"], accept_multiple_files=True)
if not imgs_up:
    st.info("➕ Envie imagens para processar.")
    st.stop()

# Processamento
for up in imgs_up:
    try:
        img = _open_image_safely(up.read())

        # Paleta (k fixo; sem KMeans global)
        palette, weights = extract_palette_kmeans(
            img, k=DEFAULT_K_COLORS, use_global=False, kmeans_global=None
        )

        # Emoção (head)
        top = predict_head(img, mlp, clip_model, preprocess, classes, device, tta=DEFAULT_TTA, topk=3)
        lbl, conf = top[0]

        # ---- Render ----
        st.markdown(f"### {up.name}")
        st.image(img, use_column_width=True)

        # Paleta visual
        cols = st.columns(len(palette))
        for i, c in enumerate(palette):
            hexv = rgb_to_hex(c)
            with cols[i]:
                st.markdown(
                    f'<div style="width:100%;height:84px;border-radius:12px;border:1px solid #ddd;background:{hexv};"></div>',
                    unsafe_allow_html=True
                )
            st.write(f"**{hexv}**")
            st.caption(f"{weights[i]*100:.1f}%")    

        # Emoção
        st.markdown(f"**Emoção (Head):** {lbl}  •  **conf:** {conf:.2f}  •  **CLIP:** {used_backbone}")
        st.caption("Top-3: " + "  |  ".join(f"{a} ({p:.2f})" for a,p in top))

        # Downloads (paleta)
        palette_payload = {
            "image": up.name,
            "palette_hex": [rgb_to_hex(c) for c in palette],
            "weights": [float(w) for w in weights],
            "top_emotions": [{"label": a, "prob": float(p)} for a,p in top],
        }
        st.download_button(
            label="⬇️ Baixar paleta (JSON)",
            file_name=f"{os.path.splitext(up.name)[0]}_palette.json",
            mime="application/json",
            data=json.dumps(palette_payload, ensure_ascii=False, indent=2)
        )

        st.divider()

    except Exception as e:
        st.error(f"[ERRO] {up.name}: {e}")