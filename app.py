"""
Transparent product-video worker (Higgsfield platform REST edition).

Endpoints:

1) GET  /health                                   -> 200 {ok}

2) POST /render   { "video_url": "<https mp4>" }   -> 200 video/webm (alpha)
   Turns an OPAQUE product mp4 into a browser-transparent WebM-VP9-alpha by removing the
   background per frame (true alpha) and re-encoding with an alpha channel. Synchronous.

3) POST /cutout   { "image_url": "<https img>" }   -> 200 image/png (alpha)
   Removes the background of a still product photo with rembg -> transparent PNG. Used as the
   canonical product image (OBRAZ A) for asset generation and as the video start frame.

4) POST /produce  { hf_auth, hf_base?, job_set_id, upload, callback }  -> 202
   Store-builder video stage, run OFF the Lovable edge function (short wall-clock budget) and
   OFF Higgsfield's slow DoP generation (~5-8 min). The edge function submits the Higgsfield
   image2video job (fast) and hands the job_set_id here; this worker — no time limit — polls
   Higgsfield's platform REST until the mp4 is ready, makes it transparent (same per-frame
   pipeline as /render), uploads the WebM to a Supabase signed upload URL, then calls back the
   edge function to finalize the asset.

Why per-frame: generative video returns opaque mp4/H.264. Real browser transparency needs WebM
VP9 alpha (yuva420p). rembg on each frame yields a true alpha PNG; ffmpeg re-encodes to WebM-alpha.
"""
import os, glob, json, shutil, subprocess, tempfile, threading, time, urllib.request, urllib.error
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import Response
from pydantic import BaseModel
from rembg import remove, new_session
from PIL import Image

REMBG_MODEL = os.environ.get("REMBG_MODEL", "u2netp")  # u2netp = fast; u2net = max quality
CRF = os.environ.get("VP9_CRF", "32")
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "600"))              # safety cap (~25s @ 24fps)
POLL_TIMEOUT = int(os.environ.get("PRODUCE_POLL_TIMEOUT", "1200"))  # Higgsfield DoP can take 5-8 min
HF_BASE_DEFAULT = os.environ.get("HF_BASE", "https://platform.higgsfield.ai")
UA = "curl/8.4.0"

app = FastAPI(title="transparent-video-worker")
_session = new_session(REMBG_MODEL)  # load model once at startup
WORKER_TOKEN = os.environ.get("WORKER_TOKEN")


class RenderIn(BaseModel):
    video_url: str


class CutoutIn(BaseModel):
    image_url: str


class UploadTarget(BaseModel):
    url: str            # full Supabase signed upload URL (PUT), content-type video/webm
    storage_path: str   # e.g. "<storeId>/pour-anim.webm" — echoed back in the callback


class Callback(BaseModel):
    url: str
    token: str
    asset_id: str


class ProduceIn(BaseModel):
    hf_auth: str                 # "KEY_ID:KEY_SECRET" — sent as `Authorization: Key ...`
    job_set_id: str              # already-submitted Higgsfield image2video job set
    upload: UploadTarget
    callback: Callback
    hf_base: str | None = None


def _run(cmd: list[str]):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd[:3])}... : {r.stderr[-500:]}")


def _download(url: str, dest: str):
    """Download to a file with a browser-ish UA — Higgsfield's CDN blocks the default
    python-urllib UA (Cloudflare error 1010), which urllib.urlretrieve would trip on."""
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


@app.get("/health")
def health():
    return {"ok": True, "model": REMBG_MODEL}


# ---------- transparency core (shared by /render and /produce) ----------

def _mp4_to_webm_alpha(src_mp4_path: str, work: str) -> tuple[bytes, int, str]:
    """mp4 file -> (webm-alpha bytes, frame_count, fps). Raises on failure."""
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
        out = remove(Image.open(f).convert("RGBA"), session=_session, post_process_mask=True)
        out.save(os.path.join(cdir, f"{i + 1:04d}.png"))

    webm = os.path.join(work, "out.webm")
    _run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", fps,
          "-i", os.path.join(cdir, "%04d.png"),
          "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", CRF,
          "-an", webm])
    return open(webm, "rb").read(), len(frames), fps


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


@app.post("/cutout")
def cutout(body: CutoutIn, authorization: str | None = Header(default=None)):
    """Still product photo -> transparent PNG (rembg). Canonical product cutout (OBRAZ A)."""
    if WORKER_TOKEN and authorization != f"Bearer {WORKER_TOKEN}":
        raise HTTPException(401, "unauthorized")
    if not body.image_url.startswith("https://"):
        raise HTTPException(400, "image_url must be https")
    work = tempfile.mkdtemp(prefix="cut_")
    try:
        src = os.path.join(work, "in")
        _download(body.image_url, src)
        out = remove(Image.open(src).convert("RGBA"), session=_session, post_process_mask=True)
        buf = os.path.join(work, "out.png")
        out.save(buf)
        return Response(content=open(buf, "rb").read(), media_type="image/png")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"cutout failed: {e}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------- Higgsfield platform REST (Key auth) ----------

def _hf_get(base: str, auth: str, path: str) -> dict:
    req = urllib.request.Request(base.rstrip("/") + path, method="GET",
                                 headers={"Authorization": f"Key {auth}", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def _extract_video_url(results: dict | None) -> str | None:
    """job.results -> best mp4 URL. Prefer full 'raw', fall back to 'min', else any nested url."""
    if not isinstance(results, dict):
        return None
    for k in ("raw", "min", "output", "video"):
        v = results.get(k)
        if isinstance(v, dict) and isinstance(v.get("url"), str):
            return v["url"]
    if isinstance(results.get("url"), str):
        return results["url"]
    # last resort: first nested {url:...} we can find
    for v in results.values():
        if isinstance(v, dict) and isinstance(v.get("url"), str):
            return v["url"]
    return None


def _poll_higgsfield_mp4(base: str, auth: str, job_set_id: str) -> str:
    """Poll GET /v1/job-sets/{id} until a job completes; return its mp4 URL. Raises on fail/timeout."""
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
    """Background: Higgsfield poll -> transparent WebM -> Supabase upload -> callback."""
    cb = body.callback
    base = body.hf_base or HF_BASE_DEFAULT
    work = tempfile.mkdtemp(prefix="tp_")
    try:
        mp4_url = _poll_higgsfield_mp4(base, body.hf_auth, body.job_set_id)
        print(f"[produce] {body.job_set_id} mp4 ready: {mp4_url[:80]}", flush=True)
        src = os.path.join(work, "in.mp4")
        _download(mp4_url, src)
        data, nframes, fps = _mp4_to_webm_alpha(src, work)
        print(f"[produce] {body.job_set_id} webm ready: {len(data)}B {nframes}f {fps}", flush=True)
        _put_bytes(body.upload.url, data, "video/webm")
        st, txt = _post_json(cb.url, {
            "asset_id": cb.asset_id, "token": cb.token, "status": "done",
            "storage_path": body.upload.storage_path, "frames": nframes, "fps": fps,
            "bytes": len(data)})
        print(f"[produce] {body.job_set_id} callback done -> {st} {txt}", flush=True)
    except Exception as e:  # noqa: BLE001 — report every failure to the callback
        msg = str(e)[:400]
        print(f"[produce] {body.job_set_id} ERROR: {msg}", flush=True)
        try:
            _post_json(cb.url, {"asset_id": cb.asset_id, "token": cb.token,
                                "status": "error", "error": msg})
        except Exception as e2:  # noqa: BLE001
            print(f"[produce] {body.job_set_id} callback failed: {e2}", flush=True)
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
