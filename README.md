# 🎨 Paleta + Emoção (Head MLP sobre CLIP) — Streamlit

Gera **paletas de cores dominantes** (via KMeans) e estima a **emoção** da imagem usando **embeddings CLIP** + **head MLP treinado**. Interface simples em **Streamlit**, rodando localmente.

## 🎥 Demo (vídeo – até 10 min)
Assista à apresentação do projeto no Loom:  
https://www.loom.com/share/92287579c9a74d87a66eda07f6e6b1bd?sid=fd9007fd-eed1-41d3-8927-2aaa0f7edf1f

---

## ✨ Features
- **Paleta dominante** com **k=6** cores (ajustável no código), extraída com **KMeans** por imagem.
- **Estimativa de emoção** com **head MLP** treinado sobre embeddings do **CLIP** (sem zero-shot).
- **Top-3 emoções** com probabilidades.
- **Exporta JSON** com `palette_hex`, `weights` e `top_emotions`.
