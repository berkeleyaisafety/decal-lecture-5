"""Facilitator tool: launch one RunPod GPU pod per group, check they're ready, and shut them all down.

Usage (needs RUNPOD_API_KEY; students never need it):
    python pods.py up --n 8 --repo https://github.com/berkeleyaisafety/decal-lecture-5
    python pods.py status          # URL per group, and whether setup has finished
    python pods.py down            # TERMINATE every pod this script made. Stopped pods still bill for storage.

Each pod: clones the repo into /workspace, installs requirements, pre-downloads the model, then writes a READY file.
JupyterLab comes up right away; setup takes ~5 minutes in the background (log: /workspace/setup.log).
Students open the URL, then File > New > Terminal, then `cd /workspace/decal-lecture-5`.
"""

import argparse
import json
import os
import secrets
import sys
import urllib.error
import urllib.request

API = "https://rest.runpod.io/v1"
PREFIX = "decal5-group"
IMAGE = "runpod/pytorch:1.4.0-cu1290-torch291-ubuntu2404"
# 48 GB cards, in order of preference: room for 5 students each running the 4B model at once.
# RunPod's Cloudflare rejects urllib's default User-Agent (error 1010).
HEADERS = {"User-Agent": "decal-lecture-5/pods.py"}
GPUS = ["NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation"]

SETUP = """(set -x
cd /workspace
git clone {repo} decal-lecture-5
cd decal-lecture-5
pip install -q -r requirements.txt
python -c 'from interp import load; load()'
touch READY) > /workspace/setup.log 2>&1 &
exec /start.sh"""


def call(method, path, body=None):
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("Set RUNPOD_API_KEY first (runpod.io > Settings > API Keys).")
    request = urllib.request.Request(
        API + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={**HEADERS, "Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            text = response.read().decode()
            return json.loads(text) if text else None
    except urllib.error.HTTPError as e:
        sys.exit(f"RunPod API error {e.code} on {method} {path}: {e.read().decode()[:500]}")


def our_pods():
    return sorted((p for p in call("GET", "/pods") if p.get("name", "").startswith(PREFIX)), key=lambda p: p["name"])


def url(pod):
    token = pod.get("env", {}).get("JUPYTER_PASSWORD", "")
    return f"https://{pod['id']}-8888.proxy.runpod.net/lab?token={token}"


def is_ready(pod):
    """Ask the pod's Jupyter server whether setup has written the READY file."""
    token = pod.get("env", {}).get("JUPYTER_PASSWORD", "")
    check = f"https://{pod['id']}-8888.proxy.runpod.net/api/contents/workspace/decal-lecture-5/READY?token={token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(check, headers=HEADERS), timeout=10) as response:
            return "ready" if response.status == 200 else "setting up"
    except urllib.error.HTTPError as e:
        return "setting up" if e.code == 404 else f"jupyter {e.code}"
    except Exception:
        return "booting"


def up(args):
    existing = len(our_pods())
    for i in range(existing + 1, existing + args.n + 1):
        body = {
            "name": f"{PREFIX}{i}",
            "imageName": IMAGE,
            "gpuTypeIds": GPUS,
            "gpuTypePriority": "custom",
            "gpuCount": 1,
            "cloudType": args.cloud,
            "containerDiskInGb": 60,
            "volumeInGb": 0,
            "ports": ["8888/http"],
            "env": {"JUPYTER_PASSWORD": secrets.token_urlsafe(9), "HF_HOME": "/workspace/hf"},
            "dockerStartCmd": ["bash", "-c", SETUP.format(repo=args.repo)],
        }
        pod = call("POST", "/pods", body)
        gpu = (pod.get("gpu") or {}).get("displayName", "?")
        print(f"{body['name']}: {pod['id']}  {gpu}  ${pod.get('costPerHr', '?')}/hr")
    print("\nSetup takes ~5 minutes. Run `python pods.py status` until every pod says ready.")


def status(args):
    pods = our_pods()
    if not pods:
        print("No pods running.")
        return
    total = 0.0
    for pod in pods:
        total += float(pod.get("costPerHr") or 0)
        print(f"{pod['name']:<16} {pod.get('desiredStatus', '?'):<9} {is_ready(pod):<11} {url(pod)}")
    print(f"\n{len(pods)} pods, ${total:.2f}/hr total")


def down(args):
    pods = our_pods()
    for pod in pods:
        call("DELETE", f"/pods/{pod['id']}")
        print(f"terminated {pod['name']} ({pod['id']})")
    print(f"{len(pods)} pods terminated." if pods else "No pods to terminate.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_up = sub.add_parser("up", help="launch pods")
    p_up.add_argument("--n", type=int, required=True, help="number of pods (one per group)")
    p_up.add_argument("--repo", required=True, help="git URL of this repo")
    p_up.add_argument("--cloud", choices=["SECURE", "COMMUNITY"], default="SECURE")
    sub.add_parser("status", help="list pods, URLs, and readiness")
    sub.add_parser("down", help="terminate all pods")
    args = parser.parse_args()
    {"up": up, "status": status, "down": down}[args.command](args)


if __name__ == "__main__":
    main()
