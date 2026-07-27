# Deploying InclusiveAI

The whole site is one container (FastAPI serves both API and frontend). The
Dockerfile here works on all three platforms below.

## Option A — Hugging Face Spaces (free, recommended to start)

1. Create an account at huggingface.co (no card needed).
2. New Space → SDK: **Docker** → name it `inclusiveai` (public).
3. Push this folder to the Space's git repo:
   ```bash
   cd InclusiveAI
   git init && git add . && git commit -m "InclusiveAI v0.1"
   git remote add hf https://huggingface.co/spaces/<your-username>/inclusiveai
   git push hf main
   ```
   (First push asks for your HF username + access token from Settings → Tokens.)
4. The Space builds and serves at `https://<your-username>-inclusiveai.hf.space`.

Note: HF requires a `README.md` in the repo root with a YAML header. Add this
to the top of README.md before pushing:
```yaml
---
title: InclusiveAI
emoji: 📊
sdk: docker
app_port: 7860
---
```

## Option B — Render (free tier)

1. Push InclusiveAI to a GitHub repo.
2. render.com → New → Web Service → connect the repo → Runtime: Docker → Free plan.
3. Done; auto-redeploys on every push. Free instances sleep after ~15 min idle
   (first visit then takes ~1 min).

## Option C — Google Cloud Run (effectively $0, no sleep, custom domain)

```bash
gcloud run deploy inclusiveai --source . --region us-east1 \
  --allow-unauthenticated --memory 512Mi
```
Scale-to-zero; the always-free tier covers research-level traffic. Map a custom
domain in the Cloud Run console when ready.

## Test locally first

```bash
docker build -t inclusiveai . && docker run -p 7860:7860 inclusiveai
# open http://localhost:7860
```
(Or just `bash run.sh` without Docker.)
