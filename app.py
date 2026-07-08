"""
Product-asset worker — clean transparent product cutouts and synthesized motion.

Endpoints:

1) GET  /health                                    -> 200 {ok}

2) POST /cutout  { image_url }                      -> 200 image/png (alpha)
   Real product photo -> clean transparent PNG (rembg u2net) + Lanczos upscale. This is the
   canonical product image (OBRAZ A): used directly for isolated-product image slots and as the
   source for /animate. No generative model -> no hallucinated water, real product, real detail.

3) POST /animate { image_url, motion?, duration?, out_w?, out_h? }  -> 200 video/webm (alpha)
   Synthesizes a premium seamless motion loop (gentle float + sway, or fake-3D spin) from the
   transparent product PNG and encodes it to WebM-VP9-alpha. Motion is computed, not generated,
   so the product stays EXACTLY the real product — no water, no artefacts, clean transparency.

4) POST /render  { video_url }                      -> 200 video/webm (alpha)
   Opaque mp4 -> transparent WebM (per-frame rembg). Kept for ad-hoc use.

5) POST /produce { hf_auth, job_set_id, upload, callback }  -> 202
   Higgsfield platform-REST poll -> transparency -> upload -> callback. Kept; unused by the
   current store pipeline (generative product video hallucinated water on spray products).

Real browser transparency needs WebM VP9 alpha (yuva420p); rembg yields the true alpha.
"""
import math, os, glob, json, shutil, subprocess, tempfile, threading, time, urllib.request, urllib.error
import numpy as np
from scipy import ndimage
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import Response
from pydantic import BaseModel
from rembg import remove, new_session
from PIL import Image

CUTOUT_MODEL = os.environ.get("CUTOUT_MODEL", "u2net")   # full model — cleaner than u2netp
VIDEO_MODEL = os.environ.get("REMBG_MODEL", "u2netp")     # per-frame /render stays fast
CRF = os.environ.get("VP9_CRF", "30")
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "600"))
UPSCALE_LONG = int(os.environ.get("CUTOUT_UPSCALE_LONG", "1600"))
ANIM_LONG = int(os.environ.get("ANIM_LONG", "1000"))     # cap animation long side (speed)
POLL_TIMEOUT = int(os.environ.get("PRODUCE_POLL_TIMEOUT", "1200"))
HF_BASE_DEFAULT = os.environ.get("HF_BASE", "https://platform.higgsfield.ai")
UA = "curl/8.4.0"

app = FastAPI(title="product-asset-worker")
_cut_session = new_session(CUTOUT_MODEL)
_vid_session = new_session(VIDEO_MODEL)
WORKER_TOKEN = os.environ.get("WORKER_TOKEN")


class RenderIn(BaseModel):
    video_url: str


class CutoutIn(BaseModel):
    image_url: str


class AnimateIn(BaseModel):
    image_url: str
    motion: str | None = None       # "float" (default) | "spin"
    duration: float | None = None   # seconds (default 4)
    out_w: int | None = None
    out_h: int | None = None


class UploadTarget(BaseModel):
    url: str
    storage_path: str


class Callback(BaseModel):
    url: str
    token: str
    asset_id: str


class ProduceIn(BaseModel):
    hf_auth: str
    job_set_id: str
    upload: UploadTarget
    callback: Callback
    hf_base: str | None = None


def _run(cmd: list[str]):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd[:3])}... : {r.stderr[-500:]}")


def _download(url: str, dest: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def _encode_webm_alpha(cdir: str, fps: str, out: str):
    _run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", fps,
          "-i", os.path.join(cdir, "%04d.png"),
          "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", CRF, "-an", out])


@app.get("/health")
def health():
    return {"ok": True, "cutout_model": CUTOUT_MODEL, "video_model": VIDEO_MODEL}


# ---------- cutout (real product -> clean transparent PNG, upscaled) ----------

def _cutout_rgba(src_path: str) -> Image.Image:
    im = Image.open(src_path).convert("RGBA")
    out = remove(im, session=_cut_session, post_process_mask=True)
    # Crop to the product's alpha bounding box (drop transparent padding) so the product is tight
    # and centers/sizes correctly wherever it's placed (hero slot object-fit, /animate framing).
    bbox = out.getbbox()
    if bbox:
        out = out.crop(bbox)
    # Upscale so the isolated product is crisp at display size (source photos are often small).
    w, h = out.size
    lo = max(w, h)
    if lo < UPSCALE_LONG:
        scale = UPSCALE_LONG / lo
        out = out.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    return out


@app.post("/cutout")
def cutout(body: CutoutIn, authorization: str | None = Header(default=None)):
    if WORKER_TOKEN and authorization != f"Bearer {WORKER_TOKEN}":
        raise HTTPException(401, "unauthorized")
    if not body.image_url.startswith("https://"):
        raise HTTPException(400, "image_url must be https")
    work = tempfile.mkdtemp(prefix="cut_")
    try:
        src = os.path.join(work, "in")
        _download(body.image_url, src)
        out = _cutout_rgba(src)
        buf = os.path.join(work, "out.png")
        out.save(buf)
        return Response(content=open(buf, "rb").read(), media_type="image/png")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"cutout failed: {e}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------- animate (synthesize a clean motion loop from the cutout) ----------

def _fit_contain(prod: Image.Image, cw: int, ch: int, frac: float) -> Image.Image:
    w, h = prod.size
    scale = min(cw * frac / w, ch * frac / h)
    return prod.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


def _animate_bytes(src_path: str, motion: str, dur: float, out_w: int, out_h: int) -> tuple[bytes, int]:
    prod0 = Image.open(src_path).convert("RGBA")
    # cap output long side for encode speed
    lo = max(out_w, out_h)
    if lo > ANIM_LONG:
        s = ANIM_LONG / lo
        out_w, out_h = max(2, round(out_w * s)), max(2, round(out_h * s))
    out_w -= out_w % 2; out_h -= out_h % 2
    fps = 24
    n = max(24, int(dur * fps))
    base = _fit_contain(prod0, out_w, out_h, 0.80)
    work = tempfile.mkdtemp(prefix="anim_")
    try:
        cdir = os.path.join(work, "c"); os.makedirs(cdir)
        for i in range(n):
            p = 2 * math.pi * i / n  # 0..2π => seamless loop
            canvas = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
            if motion == "spin":
                # gentle "turn" illusion: subtle horizontal scale + small tilt + bob. No hard edge-on.
                sx = 0.96 + 0.04 * math.cos(p)
                fr = base.resize((max(2, round(base.width * sx)), base.height), Image.LANCZOS)
                fr = fr.rotate(2.0 * math.sin(p), resample=Image.BICUBIC, expand=True)
                dy = 0.02 * out_h * math.sin(p)
            else:  # "float": gentle levitation + sway
                fr = base.rotate(3.0 * math.sin(p), resample=Image.BICUBIC, expand=True)
                dy = 0.035 * out_h * math.sin(p)
            x = (out_w - fr.width) // 2
            y = int((out_h - fr.height) // 2 + dy)
            canvas.alpha_composite(fr, (x, y))
            canvas.save(os.path.join(cdir, f"{i + 1:04d}.png"))
        out = os.path.join(work, "out.webm")
        _encode_webm_alpha(cdir, str(fps), out)
        return open(out, "rb").read(), n
    finally:
        shutil.rmtree(work, ignore_errors=True)


@app.post("/animate")
def animate(body: AnimateIn, authorization: str | None = Header(default=None)):
    if WORKER_TOKEN and authorization != f"Bearer {WORKER_TOKEN}":
        raise HTTPException(401, "unauthorized")
    if not body.image_url.startswith("https://"):
        raise HTTPException(400, "image_url must be https")
    work = tempfile.mkdtemp(prefix="animin_")
    try:
        src = os.path.join(work, "in.png")
        _download(body.image_url, src)
        data, nframes = _animate_bytes(
            src, (body.motion or "float"), float(body.duration or 4.0),
            int(body.out_w or 1080), int(body.out_h or 1080))
        return Response(content=data, media_type="video/webm", headers={"X-Frames": str(nframes)})
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"animate failed: {e}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------- per-frame transparency (mp4 -> WebM alpha), for /render and /produce ----------

def _mp4_to_webm_alpha(src_mp4_path: str, work: str) -> tuple[bytes, int, str]:
    fdir, cdir = os.path.join(work, "f"), os.path.join(work, "c")
    os.makedirs(fdir, exist_ok=True); os.makedirs(cdir, exist_ok=True)
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", src_mp4_path, os.path.join(fdir, "%04d.png")])
    fps = subprocess.check_output(
        ["ffprobe", "-v", "0", "-of", "csv=p=0", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", src_mp4_path]).decode().strip() or "24/1"
    frames = sorted(glob.glob(os.path.join(fdir, "*.png")))
    if not frames:
        raise RuntimeError("no frames extracted")
    if len(frames) > MAX_FRAMES:
        raise RuntimeError(f"too many frames ({len(frames)} > {MAX_FRAMES})")
    for i, f in enumerate(frames):
        rgba = remove(Image.open(f).convert("RGBA"), session=_vid_session, post_process_mask=True)
        _solidify_alpha(rgba).save(os.path.join(cdir, f"{i + 1:04d}.png"))
    out = os.path.join(work, "out.webm")
    _encode_webm_alpha(cdir, fps, out)
    return open(out, "rb").read(), len(frames), fps


def _solidify_alpha(rgba: Image.Image) -> Image.Image:
    """Fill enclosed holes in the alpha (transparent product parts like a clear bottle seen over the
    generated dark bg get keyed out by rembg -> a see-through hole). Flood-fill from the frame border
    marks the true background; any transparent region NOT reachable from the border is interior product
    -> force it opaque. Also drop stray specks by keeping the largest component. Opaque products have no
    enclosed holes, so this is a no-op for them."""
    arr = np.array(rgba)  # H x W x 4
    a = arr[:, :, 3]
    fg = a > 30
    filled = ndimage.binary_fill_holes(fg)
    lbl, n = ndimage.label(filled)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lbl), lbl, index=range(1, n + 1))
        filled = lbl == (int(np.argmax(sizes)) + 1)
    holes = filled & (a <= 30)          # interior pixels that were transparent
    if holes.any():
        arr[:, :, 3][holes] = 255       # make the product solid (no see-through hole)
    arr[:, :, 3][~filled] = 0           # background stays fully transparent
    return Image.fromarray(arr, "RGBA")


@app.post("/render")
def render(body: RenderIn, authorization: str | None = Header(default=None)):
    if WORKER_TOKEN and authorization != f"Bearer {WORKER_TOKEN}":
        raise HTTPException(401, "unauthorized")
    if not body.video_url.startswith("https://"):
        raise HTTPException(400, "video_url must be https")
    work = tempfile.mkdtemp(prefix="tv_")
    try:
        src = os.path.join(work, "in.mp4")
        _download(body.video_url, src)
        data, nframes, fps = _mp4_to_webm_alpha(src, work)
        return Response(content=data, media_type="video/webm",
                        headers={"X-Frames": str(nframes), "X-Fps": fps})
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------- Higgsfield platform REST /produce (kept, unused by store pipeline) ----------

def _hf_get(base: str, auth: str, path: str) -> dict:
    req = urllib.request.Request(base.rstrip("/") + path, method="GET",
                                 headers={"Authorization": f"Key {auth}", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def _extract_video_url(results: dict | None) -> str | None:
    if not isinstance(results, dict):
        return None
    for k in ("raw", "min", "output", "video"):
        v = results.get(k)
        if isinstance(v, dict) and isinstance(v.get("url"), str):
            return v["url"]
    if isinstance(results.get("url"), str):
        return results["url"]
    for v in results.values():
        if isinstance(v, dict) and isinstance(v.get("url"), str):
            return v["url"]
    return None


def _poll_higgsfield_mp4(base: str, auth: str, job_set_id: str) -> str:
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        js = _hf_get(base, auth, f"/v1/job-sets/{job_set_id}")
        jobs = js.get("jobs") or []
        job = jobs[0] if jobs else {}
        status = str(job.get("status") or "").lower()
        if status == "completed":
            url = _extract_video_url(job.get("results"))
            if not url:
                raise RuntimeError(f"completed but no video url: {json.dumps(job)[:300]}")
            return url
        if status in ("failed", "error", "canceled", "cancelled", "nsfw"):
            raise RuntimeError(f"higgsfield job {status}: {json.dumps(job)[:300]}")
        time.sleep(6)
    raise RuntimeError(f"higgsfield poll timed out after {POLL_TIMEOUT}s")


def _post_json(url: str, obj: dict):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode()[:300]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]


def _put_bytes(url: str, data: bytes, content_type: str):
    req = urllib.request.Request(url, data=data, method="PUT", headers={
        "Content-Type": content_type, "x-upsert": "true", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status


def _produce_job(body: ProduceIn):
    cb = body.callback
    base = body.hf_base or HF_BASE_DEFAULT
    work = tempfile.mkdtemp(prefix="tp_")
    try:
        mp4_url = _poll_higgsfield_mp4(base, body.hf_auth, body.job_set_id)
        src = os.path.join(work, "in.mp4")
        _download(mp4_url, src)
        data, nframes, fps = _mp4_to_webm_alpha(src, work)
        _put_bytes(body.upload.url, data, "video/webm")
        _post_json(cb.url, {"asset_id": cb.asset_id, "token": cb.token, "status": "done",
                            "storage_path": body.upload.storage_path, "frames": nframes, "fps": fps})
    except Exception as e:  # noqa: BLE001
        try:
            _post_json(cb.url, {"asset_id": cb.asset_id, "token": cb.token,
                                "status": "error", "error": str(e)[:400]})
        except Exception:  # noqa: BLE001
            pass
    finally:
        shutil.rmtree(work, ignore_errors=True)


@app.post("/produce")
def produce(body: ProduceIn, authorization: str | None = Header(default=None)):
    if WORKER_TOKEN and authorization != f"Bearer {WORKER_TOKEN}":
        raise HTTPException(401, "unauthorized")
    if not body.job_set_id:
        raise HTTPException(400, "job_set_id required")
    threading.Thread(target=_produce_job, args=(body,), daemon=True).start()
    return Response(content=json.dumps({"accepted": True, "job_set_id": body.job_set_id}),
                    status_code=202, media_type="application/json")
