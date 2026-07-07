"""
Transparent product-video worker.

Two jobs:

1) POST /render  { "video_url": "<https mp4>" }  -> 200 video/webm (alpha)
   Turns an OPAQUE product mp4 (e.g. Kling image-to-video) into a browser-transparent
   WebM-VP9-alpha by removing the background per frame (true alpha) and re-encoding with
   an alpha channel. Synchronous — returns the bytes.

2) POST /produce  { higgsfield job + supabase upload target + callback }  -> 202
   Full store-builder video stage, run OFF the Lovable edge function (which has a short
   wall-clock budget). The edge function submits the Higgsfield generate_video job (fast)
   and hands the job_id here; this worker — with no time limit — polls Higgsfield until the
   mp4 is ready, makes it transparent (same per-frame pipeline as /render), uploads the WebM
   to a Supabase signed upload URL, then calls back the edge function to finalize the asset.

Why per-frame: generative video + Higgsfield's video background-remover return mp4/H.264 with
a BLACK background (no alpha) which browsers render opaque. Real browser transparency needs
WebM VP9 alpha (yuva420p). rembg on each frame yields a true alpha PNG; ffmpeg re-encodes the
PNG sequence to WebM-alpha.

GET /health -> 200
"""
import os, glob, json, shutil, subprocess, tempfile, threading, time, urllib.request, urllib.error
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import Response
from pydantic import BaseModel
from rembg import remove, new_session
from PIL import Image

REMBG_MODEL = os.environ.get("REMBG_MODEL", "u2netp")  # u2netp = fast; u2net = max quality
CRF = os.environ.get("VP9_CRF", "32")
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "600"))          # safety cap (~25s @ 24fps)
POLL_TIMEOUT = int(os.environ.get("PRODUCE_POLL_TIMEOUT", "900"))  # max seconds to wait on Higgsfield
UA = "curl/8.4.0"  # Higgsfield's Cloudflare blocks the default python-urllib UA (error 1010)

app = FastAPI(title="transparent-video-worker")
_session = new_session(REMBG_MODEL)  # load model once at startup
WORKER_TOKEN = os.environ.get("WORKER_TOKEN")


class RenderIn(BaseModel):
    video_url: str


class UploadTarget(BaseModel):
    url: str            # full Supabase signed upload URL (PUT), content-type video/webm
    storage_path: str   # e.g. "<storeId>/pour-anim.webm" — echoed back in the callback


class Callback(BaseModel):
    url: str
    token: str
    asset_id: str


class ProduceIn(BaseModel):
    mcp_url: str
    access_token: str
    job_id: str          # already-submitted Higgsfield generate_video job
    upload: UploadTarget
    callback: Callback


def _run(cmd: list[str]):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd[:3])}... : {r.stderr[-500:]}")


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
        urllib.request.urlretrieve(body.video_url, src)
        data, nframes, fps = _mp4_to_webm_alpha(src, work)
        return Response(content=data, media_type="video/webm",
                        headers={"X-Frames": str(nframes), "X-Fps": fps})
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------- Higgsfield MCP (JSON-RPC over StreamableHTTP) ----------

def _mcp(mcp_url: str, token: str, method: str, params: dict | None, notify: bool = False):
    payload = {"jsonrpc": "2.0", "method": method}
    if not notify:
        payload["id"] = 1
    if params is not None:
        payload["params"] = params
    req = urllib.request.Request(mcp_url, data=json.dumps(payload).encode(), method="POST", headers={
        "Authorization": f"Bearer {token}", "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode()
    obj = None
    if "data:" in raw:
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                try: obj = json.loads(line[5:].strip())
                except Exception: pass
    else:
        obj = json.loads(raw)
    return obj


def _mcp_handshake(mcp_url: str, token: str):
    _mcp(mcp_url, token, "initialize", {
        "protocolVersion": "2025-06-18", "capabilities": {},
        "clientInfo": {"name": "transparent-video-worker", "version": "1.0"}})
    _mcp(mcp_url, token, "notifications/initialized", {}, notify=True)


def _poll_higgsfield_mp4(mcp_url: str, token: str, job_id: str) -> str:
    """Poll job_status(sync) until terminal; return the mp4 rawUrl. Raises on failure/timeout."""
    _mcp_handshake(mcp_url, token)
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        o = _mcp(mcp_url, token, "tools/call",
                 {"name": "job_status", "arguments": {"jobId": job_id, "sync": True}})
        sc = ((o or {}).get("result") or {}).get("structuredContent") or {}
        gen = sc.get("generation") or sc
        status = (gen.get("status") or gen.get("state") or "").lower()
        if status in ("completed", "succeeded", "done"):
            raw = ((gen.get("results") or {}).get("rawUrl")
                   or (gen.get("results") or {}).get("url"))
            if not raw:
                raise RuntimeError(f"job done but no rawUrl: {json.dumps(sc)[:300]}")
            return raw
        if status in ("failed", "error", "canceled", "cancelled"):
            raise RuntimeError(f"higgsfield job {status}: {json.dumps(sc)[:300]}")
        time.sleep(int(sc.get("poll_after_seconds", 5)) or 5)
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
    """Background: Higgsfield mp4 -> transparent WebM -> Supabase upload -> callback."""
    cb = body.callback
    work = tempfile.mkdtemp(prefix="tp_")
    try:
        mp4_url = _poll_higgsfield_mp4(body.mcp_url, body.access_token, body.job_id)
        print(f"[produce] {body.job_id} mp4 ready: {mp4_url[:80]}", flush=True)
        src = os.path.join(work, "in.mp4")
        urllib.request.urlretrieve(mp4_url, src)
        data, nframes, fps = _mp4_to_webm_alpha(src, work)
        print(f"[produce] {body.job_id} webm ready: {len(data)}B {nframes}f {fps}", flush=True)
        _put_bytes(body.upload.url, data, "video/webm")
        st, txt = _post_json(cb.url, {
            "asset_id": cb.asset_id, "token": cb.token, "status": "done",
            "storage_path": body.upload.storage_path, "frames": nframes, "fps": fps,
            "bytes": len(data)})
        print(f"[produce] {body.job_id} callback done -> {st} {txt}", flush=True)
    except Exception as e:  # noqa: BLE001 — report every failure to the callback
        msg = str(e)[:400]
        print(f"[produce] {body.job_id} ERROR: {msg}", flush=True)
        try:
            _post_json(cb.url, {"asset_id": cb.asset_id, "token": cb.token,
                                "status": "error", "error": msg})
        except Exception as e2:  # noqa: BLE001
            print(f"[produce] {body.job_id} callback failed: {e2}", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)


@app.post("/produce")
def produce(body: ProduceIn, authorization: str | None = Header(default=None)):
    if WORKER_TOKEN and authorization != f"Bearer {WORKER_TOKEN}":
        raise HTTPException(401, "unauthorized")
    if not body.mcp_url.startswith("https://"):
        raise HTTPException(400, "mcp_url must be https")
    threading.Thread(target=_produce_job, args=(body,), daemon=True).start()
    return Response(content=json.dumps({"accepted": True, "job_id": body.job_id}),
                    status_code=202, media_type="application/json")
