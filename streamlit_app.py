import io
import base64
import streamlit as st
import numpy as np
from PIL import Image as PILImage

# ---- 定数 ----
ROW, COL     = 10, 10
ALPHA        = 0.05
NEIGHBOR     = 2
DISPLAY_SIZE = 400

# ---- セッション状態の初期化 ----
if "weight" not in st.session_state:
    st.session_state.weight       = np.random.random([ROW, COL, 3])
    st.session_state.current_step = 0
    st.session_state.last_color   = np.zeros(3)

# ---- SOM ----
def som_step(weight, colorvec):
    min_index = np.argmin(((weight - colorvec) ** 2).sum(axis=2))
    mini, minj = divmod(int(min_index), COL)
    for i in range(-NEIGHBOR, NEIGHBOR + 1):
        for j in range(-NEIGHBOR, NEIGHBOR + 1):
            ni, nj = mini + i, minj + j
            if 0 <= ni < ROW and 0 <= nj < COL:
                weight[ni, nj] += ALPHA * (colorvec - weight[ni, nj]) / (abs(i) + abs(j) + 1)

def to_image_html(w, step):
    """PIL画像をbase64でHTMLに直接埋め込む（サーバー再取得なし→白フラッシュなし）"""
    arr = (np.clip(w, 0, 1) * 255).astype(np.uint8)
    pil = PILImage.fromarray(arr, "RGB").resize((DISPLAY_SIZE, DISPLAY_SIZE), PILImage.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return (
        f'<img src="data:image/png;base64,{b64}" '
        f'style="width:100%;display:block;image-rendering:pixelated">'
        f'<p style="text-align:center;color:#888;font-size:0.85em;margin-top:4px">'
        f'Step: {step:,}</p>'
    )

# ---- UI ----
st.title("Self-Organizing Map")

# @st.fragment: フラグメント内だけ再描画されるのでページ全体が白くならない
@st.fragment
def som_ui():
    steps = st.select_slider(
        "ステップ数",
        options=[1, 10, 100, 1000, 10000, 100000],
        value=1000,
    )

    col_run, col_reset = st.columns([1, 1])
    run   = col_run.button("▶ 学習実行",  type="primary",   use_container_width=True)
    reset = col_reset.button("↺ リセット", use_container_width=True)

    # ---- ボタン処理（画像表示より前）----
    # caption と progress を同じスロットで兼用することで余分な隙間を作らない
    status_slot = st.empty()

    if reset:
        st.session_state.weight       = np.random.random([ROW, COL, 3])
        st.session_state.current_step = 0
        st.session_state.last_color   = np.zeros(3)

    if run:
        w          = st.session_state.weight
        last_color = st.session_state.last_color
        chunk      = max(1, steps // 100)
        done       = 0
        while done < steps:
            batch = min(chunk, steps - done)
            for _ in range(batch):
                last_color = np.random.rand(3)
                som_step(w, last_color)
                st.session_state.current_step += 1
            done += batch
            status_slot.caption(f"学習中... {done:,} / {steps:,} ステップ")
        st.session_state.last_color = last_color

    status_slot.caption(f"累計ステップ: {st.session_state.current_step:,}")

    # ---- 色ラベル＋画像を1つの markdown にまとめて隙間をなくす ----
    c = st.session_state.last_color
    r, g, b  = (int(v * 255) for v in c)
    hex_code = f"#{r:02x}{g:02x}{b:02x}"
    color_html = (
        f'最後の入力色: '
        f'<span style="font-family:monospace">{hex_code} &nbsp; '
        f'RGB({r:3d},{g:3d},{b:3d})</span> &nbsp;'
        f'<span style="display:inline-block;width:20px;height:20px;'
        f'background:{hex_code};border:1px solid #888;border-radius:3px;'
        f'vertical-align:middle"></span>'
    )
    st.markdown(
        color_html + to_image_html(st.session_state.weight, st.session_state.current_step),
        unsafe_allow_html=True,
    )

som_ui()
