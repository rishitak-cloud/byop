"""
FrameAgent - workshop edition.

Same models, same rules as web.py. Rewritten UI:
  - no hardcoded paths (tempfile)
  - one analyse() instance instead of five, cached on image bytes
  - results shown before the images, numbers rounded
  - tabs instead of a 2x2 grid, long explanations moved into expanders
  - sample images so nobody has to hunt for a photo
  - sidebar with the pipeline + a raw U^2-Net mask toggle

Run with:  streamlit run web_v2.py
"""

import os
import re
import tempfile

import numpy as np
import streamlit as st
from PIL import Image

from composition_rules import analyse
from nima_import import score as nima_score

# ----------------------------------------------------------------------------
# page config
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="FrameAgent",
    page_icon="[]",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")

FAIL_STRINGS = {
    "thirds": "Rule of Thirds failed.",
    "golden": "Golden Ratio failed.",
    "centre": "Off-centre.",
    "symmetry": "Not symmetric.",
    "lines": "Lines not found.",
}

EXPLAIN = {
    "thirds": (
        "The frame is split by two vertical and two horizontal lines at 1/3 and 2/3. "
        "The green circle is the centroid of the subject mask predicted by U^2-Net. "
        "If that centroid lands within 5% of the frame size of one of the four "
        "intersections, the rule passes."
    ),
    "golden": (
        "Same idea as the rule of thirds, but the grid sits at 0.382 and 0.618 - the "
        "golden ratio - instead of 0.333 and 0.666. The two grids are close, which is "
        "why photos often satisfy one and narrowly miss the other."
    ),
    "symmetry": (
        "The image is split down the middle, the right half is mirrored, and the two "
        "halves are compared with SSIM (structural similarity). The map you see is the "
        "per-pixel SSIM difference: white means the two halves agree, dark means they "
        "disagree. A global score above 0.4 counts as symmetric."
    ),
    "lines": (
        "OpenCV's Line Segment Detector finds straight edges. Only segments longer than "
        "20% of the frame width and angled between 15 and 75 degrees are kept - those "
        "are the diagonals that pull the eye through a photo. Verticals and horizontals "
        "are ignored on purpose."
    ),
}


# ----------------------------------------------------------------------------
# analysis (cached on the raw bytes, so re-uploading the same photo is instant)
# ----------------------------------------------------------------------------

def _first_float(text, default=None):
    m = re.search(r"[-+]?\d*\.\d+|\d+", str(text))
    return float(m.group()) if m else default


@st.cache_data(show_spinner=False, max_entries=8)
def run_analysis(image_bytes: bytes) -> dict:
    """Run the whole pipeline once and return everything the UI needs."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    try:
        tmp.write(image_bytes)
        tmp.close()
        path = tmp.name

        a = analyse(path)  # U^2-Net runs once here, not five times

        thirds_img, v_thirds = a.thirds()
        golden_img, v_golden = a.golden()
        centre_img, v_centre = a.centre()
        symm_img, v_symm = a.symmetry()
        line_img, v_lines = a.lines()

        verdicts = {
            "thirds": v_thirds,
            "golden": v_golden,
            "centre": v_centre,
            "symmetry": v_symm,
            "lines": v_lines,
        }
        passed = {k: verdicts[k] != FAIL_STRINGS[k] for k in FAIL_STRINGS}

        crop_img, crop_target = (None, None)
        if v_lines != "Lines found." and v_centre != "Object is in the centre.":
            crop_img, crop_target = a.auto_fix_image()

        raw_nima = nima_score(path)

        return {
            "images": {
                "thirds": thirds_img,
                "golden": golden_img,
                "centre": centre_img,
                "symmetry": symm_img,
                "lines": line_img,
            },
            "verdicts": verdicts,
            "passed": passed,
            "mask": a.m,
            "centroid": (a.cX, a.cY),
            "size": (a.w, a.h),
            "symmetry_score": _first_float(v_symm),
            "nima": _first_float(raw_nima),
            "crop": crop_img,
            "crop_target": crop_target,
        }
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ----------------------------------------------------------------------------
# sidebar
# ----------------------------------------------------------------------------

with st.sidebar:
    st.subheader("What's running")
    st.markdown(
        """
**1. Segmentation** — U^2-Net predicts a saliency mask for the subject.

**2. Centroid** — image moments on the thresholded mask give (cX, cY).

**3. Rule checks** — the centroid is tested against the thirds grid, the phi
grid, and the centre, with a 5% tolerance. Symmetry uses SSIM on the mirrored
halves; leading lines use OpenCV's LSD.

**4. Aesthetics** — NIMA (pyiqa) predicts a 1–10 mean-opinion score.

**5. Auto-crop** — pick the nearest compositional target, then solve for the
largest crop that puts the subject on it.
        """
    )
    st.divider()
    show_mask = st.toggle("Show raw U^2-Net mask", value=False)
    show_debug = st.toggle("Show centroid values", value=False)


# ----------------------------------------------------------------------------
# header + input
# ----------------------------------------------------------------------------

st.title("FrameAgent")
st.caption(
    "Upload a photo. A segmentation model finds the subject, then classical "
    "composition rules grade the framing and a neural aesthetic model scores it."
)

img_bytes = None
img_label = None

upload = st.file_uploader(
    "Choose a file", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed"
)

if upload is not None:
    img_bytes = upload.getvalue()
    img_label = upload.name

# sample images - drop a few jpgs into ./samples/ and they show up as buttons
if img_bytes is None and os.path.isdir(SAMPLES_DIR):
    samples = sorted(
        f
        for f in os.listdir(SAMPLES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    )[:4]
    if samples:
        st.write("No photo handy? Try one of these:")
        cols = st.columns(len(samples))
        for col, name in zip(cols, samples):
            with col:
                st.image(os.path.join(SAMPLES_DIR, name), use_container_width=True)
                if st.button(os.path.splitext(name)[0], use_container_width=True):
                    st.session_state["sample"] = name
                    st.rerun()

if img_bytes is None and st.session_state.get("sample"):
    name = st.session_state["sample"]
    p = os.path.join(SAMPLES_DIR, name)
    if os.path.exists(p):
        with open(p, "rb") as fh:
            img_bytes = fh.read()
        img_label = name

if img_bytes is None:
    st.info("Waiting for an image.")
    st.stop()


# ----------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------

with st.status("Analysing...", expanded=False) as status:
    st.write("Running U^2-Net segmentation")
    st.write("Checking composition rules")
    st.write("Scoring with NIMA")
    result = run_analysis(img_bytes)
    status.update(label="Done", state="complete")

passed = result["passed"]
verdicts = result["verdicts"]
n_passed = sum(passed.values())
nima = result["nima"]


# ----------------------------------------------------------------------------
# scorecard - the first thing on screen
# ----------------------------------------------------------------------------

m1, m2, m3, m4 = st.columns(4)
m1.metric("Rules passed", f"{n_passed}/5")
m2.metric(
    "NIMA aesthetic score",
    f"{nima:.2f}" if nima is not None else "n/a",
    delta=f"{nima - 5.0:+.2f} vs average" if nima is not None else None,
)
sym = result["symmetry_score"]
m3.metric("Symmetry (SSIM)", f"{sym:.2f}" if sym is not None else "< 0.40")
m4.metric("Subject position", "Centred" if passed["centre"] else "Off-centre")

chips = st.columns(5)
labels = {
    "thirds": "Rule of thirds",
    "golden": "Golden ratio",
    "centre": "Centred",
    "symmetry": "Symmetry",
    "lines": "Leading lines",
}
for col, key in zip(chips, labels):
    col.markdown(
        f"{'✅' if passed[key] else '❌'} **{labels[key]}**"
        if passed[key]
        else f"❌ {labels[key]}"
    )

st.divider()


# ----------------------------------------------------------------------------
# original + per-rule tabs
# ----------------------------------------------------------------------------

left, right = st.columns([1, 1.4], gap="large")

with left:
    st.subheader("Original")
    st.image(img_bytes, use_container_width=True, caption=img_label)
    if show_mask:
        st.image(
            result["mask"],
            use_container_width=True,
            caption="U^2-Net saliency mask, thresholded at 0.5",
            clamp=True,
        )
    if show_debug:
        w, h = result["size"]
        cx, cy = result["centroid"]
        st.code(
            f"size      = {w} x {h}\n"
            f"centroid  = ({cx}, {cy})\n"
            f"normalised= ({cx / w:.3f}, {cy / h:.3f})\n"
            f"tolerance = 5% of each dimension",
            language="text",
        )

with right:
    st.subheader("Rule by rule")
    t1, t2, t3, t4 = st.tabs(["Thirds", "Golden ratio", "Symmetry", "Leading lines"])

    for tab, key in zip((t1, t2, t3, t4), ("thirds", "golden", "symmetry", "lines")):
        with tab:
            st.image(result["images"][key], use_container_width=True)
            if passed[key]:
                st.success(verdicts[key])
            else:
                st.error(verdicts[key])
            with st.expander("How this is computed"):
                st.write(EXPLAIN[key])


# ----------------------------------------------------------------------------
# suggested crop
# ----------------------------------------------------------------------------

st.divider()
st.subheader("Suggested crop")

if result["crop"] is not None:
    c1, c2 = st.columns(2)
    with c1:
        st.image(img_bytes, use_container_width=True, caption="Before")
    with c2:
        st.image(
            result["crop"],
            use_container_width=True,
            caption=f"After — subject moved to {result['crop_target']}",
        )
else:
    st.info(
        "No crop suggested. The photo either already has leading lines or a centred "
        "subject, and cropping would work against both."
    )


# ----------------------------------------------------------------------------
# NIMA caveat
# ----------------------------------------------------------------------------

with st.expander("Why is the NIMA score what it is?"):
    st.write(
        "NIMA predicts aesthetic quality, which does not always match human perception. "
        "It was trained heavily on photos with a clear focal point — a person, a flower, "
        "a bird. When it scans an image and cannot find a single sharp object to lock "
        "onto, or sees a texture gradient and is unsure what it is meant to be judging, "
        "it falls back to a safe, average score. It also tends to reward high contrast."
    )
