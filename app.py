"""
Transparent product-video worker.

Turns an OPAQUE product video (e.g. Kling image-to-video from Higgsfield, mp4/H.264 with a
solid background) into a browser-transparent WebM-VP9-alpha, by removing the background
per frame (true alpha) and re-encoding with an alpha channel.

Why per-frame: generative video models + Higgsfield's video background-remover return an mp4
with a BLACK background (single H.264 track, no alpha) which browsers render opaque. Real
browser transparency needs WebM VP9/VP8 alpha (yuva420p). rembg on each frame yields a true
alpha PNG; ffmpeg re-encodes the PNG sequence to WebM-alpha.

POST /render  { "video_url": "<https mp4 url>" }  -> 200 video/webm (alpha)
GET  /health  -> 200
"""
import os, glob, shutil, subprocess, tempfile, urllib.request
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import Response
from pydantic import BaseModel
from rembg import remove, new_session
from PIL import Image

REMBG_MODEL = os.environ.get("REMBG_MODEL", "u2netp")  # u2netp = fast; u2net = max quality
CRF = os.environ.get("VP9_CRF", "32")
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "600"))  # safety cap (~25s @ 24fps)

app = FastAPI(title="transparent-video-worker")
_session = new_session(REMBG_MODEL)  # load model once at startup


class RenderIn(BaseModel):
    video_url: str


def _run(cmd: list[str]):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise HTTPException(500, f"cmd failed: {' '.join(cmd[:3])}... : {r.stderr[-500:]}")


@app.get("/health")
def health():
    return {"ok": True, "model": REMBG_MODEL}


WORKER_TOKEN = os.environ.get("WORKER_TOKEN")


@app.post("/render")
def render(body: RenderIn, authorization: str | None = Header(default=None)):
    if WORKER_TOKEN and authorization != f"Bearer {WORKER_TOKEN}":
        raise HTTPException(401, "unauthorized")
    if not body.video_url.startswith("https://"):
        raise HTTPException(400, "video_url must be https")
    work = tempfile.mkdtemp(prefix="tv_")
    fdir, cdir = os.path.join(work, "f"), os.path.join(work, "c")
    os.makedirs(fdir); os.makedirs(cdir)
    try:
        src = os.path.join(work, "in.mp4")
        urllib.request.urlretrieve(body.video_url, src)

        # 1) frames
        _run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, os.path.join(fdir, "%04d.png")])
        fps = subprocess.check_output(
            ["ffprobe", "-v", "0", "-of", "csv=p=0", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", src]).decode().strip() or "24/1"

        frames = sorted(glob.glob(os.path.join(fdir, "*.png")))
        if not frames:
            raise HTTPException(422, "no frames extracted")
        if len(frames) > MAX_FRAMES:
            raise HTTPException(422, f"too many frames ({len(frames)} > {MAX_FRAMES})")

        # 2) per-frame background removal -> RGBA (true alpha)
        for i, f in enumerate(frames):
            out = remove(Image.open(f).convert("RGBA"), session=_session, post_process_mask=True)
            out.save(os.path.join(cdir, f"{i + 1:04d}.png"))

        # 3) encode WebM VP9 with alpha (yuva420p) — browser-transparent
        webm = os.path.join(work, "out.webm")
        _run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", fps,
              "-i", os.path.join(cdir, "%04d.png"),
              "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", CRF,
              "-an", webm])
        data = open(webm, "rb").read()
        return Response(content=data, media_type="video/webm",
                        headers={"X-Frames": str(len(frames)), "X-Fps": fps})
    finally:
        shutil.rmtree(work, ignore_errors=True)
