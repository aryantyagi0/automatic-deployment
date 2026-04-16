
"""
AI Docker Deployment Agent — LangGraph Version
Converts the original main5.py into a proper LangGraph state machine.
Every function, every feature, every HITL checkpoint is preserved exactly.
"""

import requests
import subprocess
import os
import shutil
import stat
import time
import json
import sys
from typing import TypedDict, Optional, List
from openai import OpenAI
from dotenv import load_dotenv
import socket
from functools import wraps
# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


import shutil as _shutil

def _tool_ok(name):
    return _shutil.which(name) is not None

if not _tool_ok("git"):
    raise EnvironmentError("git not found in PATH. Install Git before running the agent.")
if not _tool_ok("docker"):
    print("[Agent] ⚠️  docker not found in PATH — Docker tests will fail.")


subprocess.run(["git", "config", "--global", "user.name", "AI-Agent"])
subprocess.run(["git", "config", "--global", "user.email", "ai-agent@example.com"])
# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM EXCEPTIONS
# ══════════════════════════════════════════════════════════════════════════════

class NetworkError(Exception):
    """Internet/connection problems"""
    pass

class GitHubError(Exception):
    """GitHub API problems"""
    pass

class DockerError(Exception):
    """Docker build/run problems"""
    pass

class ConfigError(Exception):
    """Missing config or credentials"""
    pass
STATE_FILE = "_agent_state.json"
DEPLOY_STATE_FILE = "_deploy_state.json"
# ══════════════════════════════════════════════════════════════════════════════
# RETRY DECORATOR
# ══════════════════════════════════════════════════════════════════════════════

def with_retry(max_attempts=3, delay=2, backoff=2, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        print(f"[Retry] ❌ '{func.__name__}' failed after {max_attempts} attempts")
                        raise
                    print(f"[Retry] ⚠️  Attempt {attempt}/{max_attempts} failed: {e}")
                    print(f"[Retry] ⏳ Waiting {current_delay}s before retry...")
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator


# ══════════════════════════════════════════════════════════════════════════════
# SAFE REQUEST — always has timeout, converts errors properly
# ══════════════════════════════════════════════════════════════════════════════

def safe_request(method, url, **kwargs):
    kwargs.setdefault("timeout", 30)
    try:
        return requests.request(method, url, **kwargs)
    except requests.exceptions.ConnectionError as e:
        raise NetworkError(f"Connection failed: {e}")
    except requests.exceptions.Timeout:
        raise NetworkError(f"Request timed out after {kwargs['timeout']}s")
    except requests.exceptions.RequestException as e:
        raise NetworkError(f"Request failed: {e}")


def safe_json(response, context=""):
    try:
        return response.json()
    except Exception:
        raise GitHubError(
            f"Expected JSON but got garbage {context}. "
            f"Status: {response.status_code}, "
            f"Body: {response.text[:200]}"
        )

# ══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH STATE DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    repo_url:        str
    token:           str
    openai_api_key:  str
    fork_owner:      str
    default_branch:  str
    fork_url:        str
    folder:          str
    context:         dict
    dockerfile:      str
    test_passed:     bool
    deploy_targets:  List[str]
    app_name:        str
    deploy_results:  dict
    pr_approved:     bool
    pr_url:          str
    deploy_approved: bool
    env_vars:        dict
    paused:          bool
    error:           Optional[str]
    current_step:    str


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def safe_rmtree(path):
    shutil.rmtree(path, onexc=_remove_readonly)

def make_github_headers(token):
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

SENSITIVE_KEYS = {
    "token", "openai_api_key", "secret_key", "access_key",
    "password", "api_key", "client_secret", "dockerhub_pass"
}

SENSITIVE_ENV_MAP = {
    "token":          "GITHUB_TOKEN",
    "openai_api_key": "OPENAI_API_KEY",
    "secret_key":     "AWS_SECRET_ACCESS_KEY",
    "access_key":     "AWS_ACCESS_KEY_ID",
    "dockerhub_pass": "DOCKERHUB_PASSWORD",
}

def save_state(data: dict):
    safe_data = {}
    for key, value in data.items():
        if key in SENSITIVE_KEYS:
            env_var_name = SENSITIVE_ENV_MAP.get(key, key.upper())
            safe_data[f"__ref__{key}"] = env_var_name
        else:
            safe_data[key] = value
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(safe_data, f, indent=2)
        print(f"[Agent] 💾 State saved (tokens/keys NOT written to disk)")
    except OSError as e:
        print(f"[Agent] ⚠️  Could not save state: {e}")

def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[Agent] ❌ Could not read state file: {e}")
        return None

    restored = {}
    for key, value in data.items():
        if key.startswith("__ref__"):
            original_key = key[7:]       # "__ref__token" → "token"
            env_var_name = value         # e.g. "GITHUB_TOKEN"
            env_value    = os.getenv(env_var_name, "").strip()
            if env_value:
                print(f"[Agent] ✅ {env_var_name} loaded from environment")
                restored[original_key] = env_value
            else:
                print(f"[Agent] ⚠️  {env_var_name} not found in environment")
                env_value = input(f"  Enter {original_key}: ").strip()
                if not env_value:
                    raise ConfigError(f"{original_key} is required but was not provided")
                restored[original_key] = env_value
        else:
            restored[key] = value
    return restored

def save_deploy_state(data: dict):
    SENSITIVE = {"token", "access_key", "secret_key", "client_secret",
                 "api_key", "dockerhub_pass", "password"}
    safe_creds = {}
    for platform, cred_dict in data.get("creds", {}).items():
        safe_creds[platform] = {
            k: ("__REDACTED__" if k in SENSITIVE else v)
            for k, v in cred_dict.items()
        }
    safe_data          = dict(data)
    safe_data["creds"] = safe_creds
    try:
        with open(DEPLOY_STATE_FILE, "w") as f:
            json.dump(safe_data, f, indent=2)
        print(f"[Deploy] 💾 Deploy state saved (credentials redacted)")
    except OSError as e:
        print(f"[Deploy] ⚠️  Could not save deploy state: {e}")


def load_deploy_state() -> dict:
    if not os.path.exists(DEPLOY_STATE_FILE):
        return None
    try:
        with open(DEPLOY_STATE_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[Deploy] ❌ Could not read deploy state: {e}")
        return None


def clear_deploy_state():
    if os.path.exists(DEPLOY_STATE_FILE):
        os.remove(DEPLOY_STATE_FILE)
        print(f"[Deploy] 🗑️  Deploy state cleared")


def _is_credential_error(msg: str) -> bool:
    kws = ["unauthorized", "authentication", "invalid token", "access denied",
           "forbidden", "401", "403", "invalid api key", "incorrect credentials",
           "invalid credentials", "not authorized", "docker login failed",
           "denied: requested access", "invalidsignature", "no such access key"]
    return any(k in msg.lower() for k in kws)


def _is_quota_error(msg: str) -> bool:
    kws = ["free tier", "quota exceeded", "limit exceeded", "instancelimitexceeded",
           "billing", "payment", "upgrade", "plan limit", "rate limit",
           "insufficient capacity", "vcpu"]
    return any(k in msg.lower() for k in kws)


def _is_network_error(msg: str) -> bool:
    kws = ["connection", "timeout", "network", "could not resolve",
           "max retries", "remotedisconnected", "ssl", "timed out"]
    return any(k in msg.lower() for k in kws)

def _check_docker_running() -> bool:
    """Check if Docker daemon is running before attempting any Docker operation."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def _wait_for_docker(max_wait_seconds=120) -> bool:
    """
    Wait for Docker to start — useful if user just opened Docker Desktop.
    Checks every 5 seconds for up to max_wait_seconds.
    """
    print(f"[Docker] ⏳ Waiting for Docker to start (up to {max_wait_seconds}s)...")
    for i in range(0, max_wait_seconds, 5):
        if _check_docker_running():
            print(f"[Docker] ✅ Docker is running!")
            return True
        print(f"[Docker] ⏳ Still waiting... ({i}s elapsed)")
        time.sleep(5)
    return False


def _print_credential_hint(platform: str, error_msg: str):
    hints = {
        "aws": [
            "  • Go to AWS Console → IAM → Users → Security credentials",
            "  • Generate a new Access Key ID and Secret Access Key",
            "  • Make sure IAM user has EC2, ECR, and STS permissions",
        ],
        "azure": [
            "  • Go to Azure Portal → Azure Active Directory → App registrations",
            "  • Check Client ID, Tenant ID, and Client Secret are correct",
            "  • Client secrets expire — create a new one if needed",
        ],
        "render": [
            "  • Go to Render Dashboard → Account Settings → API Keys",
            "  • Create a new API key and use it here",
        ],
        "railway": [
            "  • Go to Railway Dashboard → Account Settings → Tokens",
            "  • Create a new token",
            "  • For Docker Hub: verify username/password at hub.docker.com",
        ],
    }
    print(f"\n[Deploy] 💡 How to fix {platform.upper()} credentials:")
    for hint in hints.get(platform, ["  • Check your credentials and try again"]):
        print(hint)


def _print_quota_hint(platform: str):
    hints = {
        "aws": [
            "  1. Check EC2 instance limits in your region (AWS Console → EC2 → Limits)",
            "  2. Request a limit increase or switch to a different region",
            "  3. Or terminate unused EC2 instances to free up capacity",
        ],
        "azure": [
            "  1. Check Azure subscription spending limits",
            "  2. Upgrade to Pay-As-You-Go if free tier is exhausted",
        ],
        "render": [
            "  1. Render free tier allows 1 free web service",
            "  2. Consider upgrading to Starter ($7/month) for always-on",
        ],
        "railway": [
            "  1. Railway Hobby plan: $5/month credit, pay per use after",
            "  2. Add a payment method to continue deploying",
        ],
    }
    for hint in hints.get(platform, ["  • Check your plan limits and try again"]):
        print(hint)


def _prompt_recredential(platform: str, app_name: str, env_vars: dict) -> dict:
    print(f"\n[Deploy] Enter new credentials for {platform.upper()}:")
    print(f"[Deploy] (Press Enter to keep current value)")

    def ask(label, env_key="", secret=False):
        env_val = os.getenv(env_key, "").strip() if env_key else ""
        masked  = f"[current: {'***' if secret and env_val else env_val or 'empty'}]"
        val     = input(f"  {label} {masked}: ").strip()
        return val if val else env_val

    if platform == "aws":
        return {
            "access_key": ask("AWS_ACCESS_KEY_ID",     "AWS_ACCESS_KEY_ID"),
            "secret_key": ask("AWS_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY", secret=True),
            "region":     ask("AWS_REGION",            "AWS_REGION"),
            "app_name":   app_name,
            "env_vars":   env_vars,
        }
    elif platform == "azure":
        return {
            "client_id":       ask("AZURE_CLIENT_ID",       "AZURE_CLIENT_ID"),
            "client_secret":   ask("AZURE_CLIENT_SECRET",   "AZURE_CLIENT_SECRET",   secret=True),
            "tenant_id":       ask("AZURE_TENANT_ID",       "AZURE_TENANT_ID"),
            "subscription_id": ask("AZURE_SUBSCRIPTION_ID", "AZURE_SUBSCRIPTION_ID"),
            "resource_group":  ask("AZURE_RESOURCE_GROUP",  "AZURE_RESOURCE_GROUP"),
            "dockerhub_user":  ask("Docker Hub Username",   "DOCKERHUB_USERNAME"),
            "dockerhub_pass":  ask("Docker Hub Password",   "DOCKERHUB_PASSWORD",    secret=True),
            "app_name":        app_name,
            "env_vars":        env_vars,
        }
    elif platform == "render":
        return {
            "api_key":  ask("RENDER_API_KEY", "RENDER_API_KEY", secret=True),
            "app_name": app_name,
            "env_vars": env_vars,
        }
    elif platform == "railway":
        return {
            "token":          ask("RAILWAY_TOKEN",       "RAILWAY_TOKEN",      secret=True),
            "dockerhub_user": ask("Docker Hub Username", "DOCKERHUB_USERNAME"),
            "dockerhub_pass": ask("Docker Hub Password", "DOCKERHUB_PASSWORD", secret=True),
            "app_name":       app_name,
            "env_vars":       env_vars,
        }
    return {}
# ══════════════════════════════════════════════════════════════════════════════
# GITHUB FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

@with_retry(max_attempts=3, delay=2, backoff=2, exceptions=(NetworkError,))
def get_authenticated_user(token):
    if not token or not token.strip():
        raise ConfigError("GitHub token is empty. Set GITHUB_TOKEN in .env")

    response = safe_request(
        "GET",
        "https://api.github.com/user",
        headers=make_github_headers(token)
    )

    if response.status_code == 401:
        raise ConfigError(
            "GitHub token is invalid or expired.\n"
            "Go to github.com/settings/tokens and create a new one."
        )
    if response.status_code == 403:
        raise ConfigError(
            "GitHub token missing permissions.\n"
            "Make sure it has repo and workflow scopes."
        )
    if response.status_code != 200:
        raise NetworkError(f"GitHub returned {response.status_code} — will retry")

    data  = safe_json(response, "while getting user info")
    login = data.get("login")
    if not login:
        raise GitHubError("GitHub response had no login field")

    print(f"[Agent] Authenticated as: {login}")
    return login

@with_retry(max_attempts=3, delay=3, backoff=2, exceptions=(NetworkError,))
def get_default_branch(repo_url, token):
    repo     = repo_url.replace("https://github.com/", "").rstrip("/")
    response = safe_request(
        "GET",
        f"https://api.github.com/repos/{repo}",
        headers=make_github_headers(token)
    )

    if response.status_code == 404:
        raise ConfigError(
            f"Repo not found: {repo_url}\n"
            f"Check: 1) URL is correct  2) Token has access"
        )
    if response.status_code == 403:
        raise ConfigError(
            "No access to this repo.\n"
            "Token needs repo scope for private repos."
        )
    if response.status_code != 200:
        raise NetworkError(f"GitHub returned {response.status_code} — will retry")

    data   = safe_json(response, "while getting repo info")
    branch = data.get("default_branch")
    if not branch:
        raise GitHubError("Response had no default_branch field")

    print(f"[Agent] Default branch: {branch}")
    return branch
    

@with_retry(max_attempts=3, delay=5, backoff=2, exceptions=(NetworkError,))
def fork_repo(repo_url, token):
    print("[Agent] Forking repository...")
    repo     = repo_url.replace("https://github.com/", "").rstrip("/")
    response = safe_request(
        "POST",
        f"https://api.github.com/repos/{repo}/forks",
        headers=make_github_headers(token)
    )

    if response.status_code == 403 and "already exists" in response.text.lower():
        print("[Agent] Already forked — using existing fork")
        login = get_authenticated_user(token)
        return f"https://github.com/{login}/{repo.split('/')[-1]}.git"

    if response.status_code == 403:
        raise ConfigError(f"Cannot fork this repo: {response.text[:200]}")

    if response.status_code not in (200, 202):
        raise NetworkError(f"Fork returned {response.status_code} — will retry")

    data     = safe_json(response, "while forking")
    fork_url = data.get("clone_url")
    if not fork_url:
        raise GitHubError("No clone_url in fork response")

    original_owner      = repo.split("/")[0].lower()
    fork_owner_returned = data.get("owner", {}).get("login", "").lower()
    if fork_owner_returned == original_owner:
        print("[Agent] You own this repo — working directly, no fork needed")

    print(f"[Agent] Fork requested: {fork_url}")

    # Wait for fork to be ready
    print("[Agent] Waiting for fork", end="", flush=True)
    for attempt in range(20):
        print(".", end="", flush=True)
        time.sleep(3)
        try:
            check = safe_request(
                "GET", data["url"],
                headers=make_github_headers(token)
            )
            if check.status_code == 200:
                print(f" ready!")
                break
        except NetworkError:
            pass
    else:
        print()
        raise GitHubError("Fork did not become ready in time")

    return fork_url

def download_repo(repo_url, fork_url, default_branch):
    print(f"[Agent] Cloning latest upstream default branch: {default_branch}")
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    if os.path.exists(repo_name):
        safe_rmtree(repo_name)
    subprocess.run(
        ["git", "clone", "--branch", default_branch, "--single-branch", repo_url, repo_name],
        check=True
    )
    subprocess.run(["git", "remote", "rename", "origin", "upstream"], cwd=repo_name, check=True)
    subprocess.run(["git", "remote", "add", "origin", fork_url], cwd=repo_name, check=True)
    print("[Agent] Repo cloned from upstream:", repo_name)
    print("[Agent] Remotes configured: upstream=original repo, origin=fork")
    return repo_name


# ══════════════════════════════════════════════════════════════════════════════
# DEEP SCAN REPO
# ══════════════════════════════════════════════════════════════════════════════

def deep_scan_repo(folder):
    print("[Agent] Deep scanning repository...")
    context = {}

    all_files = os.listdir(folder)
    context["all_files"] = all_files
    print(f"[Agent] Root files: {all_files}")

    JUNK_FILES = {
        ".git", ".github", ".gitignore", ".gitattributes",
        "README.md", "readme.md", "LICENSE", "license",
        ".DS_Store", "Thumbs.db", ".env.example",
    }
    ENTRY_SIGNALS = {
        "requirements.txt", "setup.py", "setup.cfg", "pyproject.toml",
        "Pipfile", "environment.yml", "environment.yaml",
        "app.py", "main.py", "server.py", "manage.py",
        "streamlit_app.py", "gradio_app.py", "run.py", "api.py",
        "package.json", "go.mod", "pom.xml", "build.gradle",
        "Gemfile", "Cargo.toml", "composer.json",
        "Dockerfile", "docker-compose.yml", "Makefile", "Procfile",
    }

    def find_project_root(base, max_depth=3):
        current = base
        for _ in range(max_depth):
            entries = os.listdir(current)
            real    = [e for e in entries if e not in JUNK_FILES]
            if any(e in ENTRY_SIGNALS for e in entries):
                return current
            if len(real) > 1:
                return current
            if len(real) == 1 and os.path.isdir(os.path.join(current, real[0])):
                print(f"[Agent] 📦 Diving into subfolder: {real[0]}/")
                current = os.path.join(current, real[0])
                continue
            break
        return current

    project_root = find_project_root(folder)

    if project_root != folder:
        print(f"[Agent] 📦 Project root detected at: {project_root}")
        print(f"[Agent] Promoting files to repo root...")
        for item in os.listdir(project_root):
            src = os.path.join(project_root, item)
            dst = os.path.join(folder, item)
            if os.path.exists(dst):
                print(f"[Agent] ⚠️  Skipping (exists at root): {item}")
                continue
            try:
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            except Exception as e:
                print(f"[Agent] ⚠️  Could not promote {item}: {e}")
        print(f"[Agent] ✅ Promotion done")
        all_files = os.listdir(folder)
        context["all_files"] = all_files
        print(f"[Agent] Root files after promotion: {all_files}")
    else:
        print(f"[Agent] ✅ Project already at root — no promotion needed")

    for subdir in ["src", "public", "app", "pages", "components",
                   "models", "notebooks", "data", "scripts", "api",
                   "lib", "utils", "training", "inference", "pipeline"]:
        path = os.path.join(folder, subdir)
        if os.path.isdir(path):
            context[f"subdir_{subdir}"] = os.listdir(path)

    dep_files = [
        "requirements.txt", "Pipfile", "pyproject.toml", "setup.py", "setup.cfg",
        "environment.yml", "environment.yaml", "conda.yml",
        "package.json", "yarn.lock", "package-lock.json",
        "pom.xml", "build.gradle", "go.mod", "Gemfile",
        "composer.json", "Cargo.toml",
        "runtime.txt", ".python-version", ".nvmrc",
        "vite.config.js", "vite.config.ts",
        "next.config.js", "next.config.ts",
        "nuxt.config.js", "angular.json",
        "Makefile", "config.yaml", "config.yml",
        "params.yaml", "dvc.yaml", "MLproject", "bentofile.yaml",
    ]
    for fname in dep_files:
        path = os.path.join(folder, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                context[f"dep_file_{fname}"] = f.read(5000)
            print(f"[Agent] Read: {fname}")

    print("[Agent] 🔍 Content-scanning all .py files for framework signals...")

    UI_SIGNALS = {
        "import streamlit": 100, "from streamlit": 100,
        "st.title": 90, "st.write": 80, "st.sidebar": 80,
        "st.button": 80, "st.selectbox": 80, "st.text_input": 80,
        "st.chat_message": 90, "st.chat_input": 90,
        "import gradio": 100, "from gradio": 100,
        "gr.interface": 90, "gr.blocks": 90, "gr.chatinterface": 90,
        "from fastapi": 90, "import fastapi": 90,
        "fastapi()": 95, "@app.get": 80, "@app.post": 80, "@router.get": 80,
        "from flask import": 90, "flask(__name__)": 95, "@app.route": 85,
        "import django": 85, "from django": 85,
        "uvicorn": 70, "starlette": 70,
    }
    BACKEND_PENALTIES = {
        "train.py": -60, "predict.py": -50, "inference.py": -50,
        "score.py": -50, "utils.py": -40, "helpers.py": -40,
        "config.py": -30, "settings.py": -30,
    }

    scored_files = {}

    for walk_root, walk_dirs, walk_files in os.walk(folder):
        walk_dirs[:] = [d for d in walk_dirs if d not in
                        (".git", "__pycache__", "venv", ".venv",
                         "node_modules", ".mypy_cache", "site-packages")]
        for fname in walk_files:
            if not fname.endswith(".py"):
                continue
            if any(fname.startswith(p) for p in ["test_", "__init__"]):
                continue
            fpath = os.path.join(walk_root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                    fcontent = fh.read(3000)
            except Exception:
                continue
            flower = fcontent.lower()
            score = sum(pts for sig, pts in UI_SIGNALS.items() if sig.lower() in flower)
            score += BACKEND_PENALTIES.get(fname, 0)
            if "if __name__" in fcontent:
                score += 5
            rel = os.path.relpath(fpath, folder).replace("\\", "/")
            # ── CHANGE 1: use rel path as key — no collisions for same-named files ──
            scored_files[rel] = (score, fcontent, rel)

    for fname in ["index.js", "server.js", "app.js", "index.ts", "index.html"]:
        fpath = os.path.join(folder, fname)
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                scored_files[fname] = (50, fh.read(3000), fname)

    # ── CHANGE 2: store entrypoint context by both rel path and basename ──
    for rel_key, (score, fcontent, rel) in scored_files.items():
        basename = os.path.basename(rel_key)
        context[f"entrypoint_{basename}"] = fcontent
        context[f"entrypoint_{rel_key}"] = fcontent

    # ── CHANGE 3: boost by basename, penalize depth ──
    ENTRY_PRIORITY = {
        "main.py": 200, "app.py": 190, "server.py": 180,
        "run.py": 170, "api.py": 160, "wsgi.py": 150, "asgi.py": 150,
        "manage.py": 140, "streamlit_app.py": 200, "gradio_app.py": 200,
    }
    boosted = {}
    for rel_key, (score, content, rel) in scored_files.items():
        basename      = os.path.basename(rel_key)
        boost         = ENTRY_PRIORITY.get(basename, 0)
        depth_penalty = rel_key.count("/") * 10
        boosted[rel_key] = (score + boost - depth_penalty, content, rel)

    sorted_files = sorted(boosted.items(), key=lambda x: x[1][0], reverse=True)
    print(f"[Agent] 📊 File scores (with boost): { {f: s for f, (s,_,_) in sorted_files[:8]} }")

    entry_points_found = []
    # ── CHANGE 4: ui_candidates uses rel_key ──
    ui_candidates = [rel_key for rel_key, (score, _, _) in sorted_files if score > 0]

    if ui_candidates:
        entry_points_found = ui_candidates
        print(f"[Agent] ✅ Content-detected entry points: {entry_points_found}")
    else:
        print("[Agent] ⚠️  No UI signals found — falling back to filename matching")
        # ── CHANGE 5: fallback matching uses basename ──
        for target in ["app.py", "main.py", "server.py", "run.py", "api.py",
                       "manage.py", "wsgi.py", "asgi.py"]:
            match = next((k for k in scored_files if os.path.basename(k) == target), None)
            if match:
                entry_points_found.append(match)
        if not entry_points_found and scored_files:
            entry_points_found = [list(scored_files.keys())[0]]
            print(f"[Agent] 📄 Last resort entry point: {entry_points_found}")

    if not entry_points_found:
        print("[Agent] ⚠️  No standard entry point found — using LLM to detect...")
        extension_map = {
            ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
            ".rb": "Ruby", ".go": "Go", ".java": "Java",
            ".php": "PHP", ".rs": "Rust", ".sh": "Shell",
        }
        special_files  = ["Makefile", "Procfile"]
        all_relevant   = []
        for f in all_files:
            ext = os.path.splitext(f)[1].lower()
            skip_prefixes = ["test_", "conf", "setup", "config", "__init__"]
            if ext in extension_map and not any(f.startswith(s) for s in skip_prefixes):
                all_relevant.append((f, extension_map[ext]))
            elif f in special_files:
                all_relevant.append((f, "Special"))

        file_snippets = {}
        for fname, lang_label in all_relevant[:20]:
            fpath = os.path.join(folder, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    file_snippets[fname] = (lang_label, f.read(500))
            except Exception:
                pass

        if file_snippets:
            snippet_text = "\n\n".join(
                f"--- {fname} ({lang_label}) ---\n{content}"
                for fname, (lang_label, content) in file_snippets.items()
            )
            client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"""
Given these files from a repository, identify which ONE file is the main entry point.
Files:\n{snippet_text}
Return ONLY the filename, nothing else.
"""}],
                temperature=0,
            )
            detected_entry = response.choices[0].message.content.strip()
            all_filenames  = [f for f, _ in all_relevant]
            if detected_entry in all_filenames:
                fpath = os.path.join(folder, detected_entry)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read(3000)
                    context[f"entrypoint_{detected_entry}"] = content
                    entry_points_found.append(detected_entry)
                    print(f"[Agent] 🤖 LLM detected entry point: {detected_entry}")
                except Exception:
                    pass
            else:
                fallback = next((f for f, _ in all_relevant if f != "__init__.py"), None)
                if fallback:
                    entry_points_found.append(fallback)
                    print(f"[Agent] 📄 Fallback entry point: {fallback}")

    context["entry_points_found"] = entry_points_found
    print(f"[Agent] Entry points: {entry_points_found}")

    notebooks   = [f for f in all_files if f.endswith(".ipynb")]
    context["notebooks_found"] = notebooks

    model_files = [f for f in all_files if any(
        f.endswith(ext) for ext in [
            ".pkl", ".joblib", ".h5", ".keras", ".pt", ".pth",
            ".onnx", ".pb", ".bin", ".safetensors", ".ckpt", ".model"
        ]
    )]
    context["model_files_found"] = model_files

    detected_lang      = "unknown"
    detected_framework = "unknown"
    app_variable       = "app"
    python_version     = "3.11"
    node_version       = "18"
    is_frontend        = False
    frontend_type      = "unknown"
    is_ml              = False
    ml_type            = "unknown"
    ml_frameworks      = []
    uses_conda         = False
    uses_gpu           = False
    build_output_dir   = "dist"

    if (any(f.endswith(".py") for f in all_files)
            or "requirements.txt" in all_files
            or "environment.yml" in all_files
            or "environment.yaml" in all_files):
        detected_lang = "python"

    if "environment.yml" in all_files or "environment.yaml" in all_files:
        uses_conda = True

    all_content = ""
    for key, val in context.items():
        if key.startswith("dep_file_") or key.startswith("entrypoint_"):
            all_content += val.lower() + "\n"

    ml_lib_map = {
        "tensorflow":  ["tensorflow", "tf.", "keras"],
        "pytorch":     ["torch", "torchvision", "torchaudio"],
        "sklearn":     ["sklearn", "scikit-learn"],
        "xgboost":     ["xgboost", "xgb."],
        "lightgbm":    ["lightgbm", "lgbm"],
        "catboost":    ["catboost"],
        "huggingface": ["transformers", "huggingface", "datasets", "diffusers"],
        "langchain":   ["langchain"],
        "openai":      ["openai"],
        "anthropic":   ["anthropic"],
        "spacy":       ["spacy"],
        "nltk":        ["nltk"],
        "pandas":      ["pandas"],
        "numpy":       ["numpy"],
        "matplotlib":  ["matplotlib"],
        "seaborn":     ["seaborn"],
        "plotly":      ["plotly"],
        "mlflow":      ["mlflow"],
        "bentoml":     ["bentoml"],
        "fastai":      ["fastai"],
        "opencv":      ["cv2", "opencv"],
        "streamlit":   ["streamlit"],
        "gradio":      ["gradio"],
    }
    for lib, keywords in ml_lib_map.items():
        if any(kw in all_content for kw in keywords):
            ml_frameworks.append(lib)

    if ml_frameworks:
        is_ml = True

    gpu_keywords = ["cuda", "torch.cuda", "device('cuda')", "tensorflow-gpu", ".to('cuda')"]
    if any(kw in all_content for kw in gpu_keywords):
        uses_gpu = True

    if detected_lang == "python" and is_ml:
        if "streamlit" in ml_frameworks or any(
            "streamlit" in context.get(f"entrypoint_{os.path.basename(e)}", "").lower() for e in entry_points_found
        ):
            detected_framework = "streamlit"
            ml_type = "streamlit"
            for c in ["streamlit_app.py", "app.py", "dashboard.py", "dashbord.py", "demo.py", "main.py"]:
                match = next((e for e in entry_points_found if os.path.basename(e) == c), None)
                if match:
                    if "streamlit" in context.get(f"entrypoint_{c}", "").lower() or c == "streamlit_app.py":
                        context["streamlit_entry_file"] = match
                        break
            if "streamlit_entry_file" not in context:
                context["streamlit_entry_file"] = entry_points_found[0] if entry_points_found else "app.py"

        elif "gradio" in ml_frameworks or any(
            "gradio" in context.get(f"entrypoint_{os.path.basename(e)}", "").lower() for e in entry_points_found
        ):
            detected_framework = "gradio"
            ml_type = "gradio"
            for c in ["app.py", "demo.py", "gradio_app.py", "main.py", "interface.py"]:
                match = next((e for e in entry_points_found if os.path.basename(e) == c), None)
                if match:
                    context["gradio_entry_file"] = match
                    break
            if "gradio_entry_file" not in context:
                context["gradio_entry_file"] = entry_points_found[0] if entry_points_found else "app.py"

        elif notebooks and not entry_points_found:
            detected_framework = "jupyter"
            ml_type = "jupyter"

        elif any("fastapi" in context.get(f"entrypoint_{os.path.basename(e)}", "").lower() for e in entry_points_found):
            detected_framework = "fastapi_ml"
            ml_type = "fastapi_ml"
            for e in entry_points_found:
                if "fastapi" in context.get(f"entrypoint_{os.path.basename(e)}", "").lower():
                    for line in context.get(f"entrypoint_{os.path.basename(e)}", "").splitlines():
                        if "fastapi()" in line.lower() and "=" in line:
                            app_variable = line.split("=")[0].strip()
                    context["fastapi_entry_file"] = e
                    break

        elif any("flask" in context.get(f"entrypoint_{os.path.basename(e)}", "").lower() for e in entry_points_found):
            detected_framework = "flask_ml"
            ml_type = "flask_ml"
            for e in entry_points_found:
                if "flask" in context.get(f"entrypoint_{os.path.basename(e)}", "").lower():
                    context["flask_entry_file"] = e
                    break

        elif any(os.path.basename(e) in ["train.py","predict.py","inference.py","score.py","main.py"]
                 for e in entry_points_found):
            detected_framework = "ml_script"
            ml_type = "ml_script"
            context["ml_script_entry"] = next(
                (e for e in entry_points_found
                 if os.path.basename(e) in ["train.py","predict.py","inference.py","score.py","main.py"]),
                entry_points_found[0]
            )

        elif "mlflow" in ml_frameworks or "MLproject" in all_files:
            detected_framework = "mlflow"
            ml_type = "mlflow"

        elif "bentoml" in ml_frameworks or "bentofile.yaml" in all_files:
            detected_framework = "bentoml"
            ml_type = "bentoml"

    if detected_lang == "python" and detected_framework == "unknown":

        # ── CHANGE 6: priority file detection uses basename lookup ──
        for priority_basename in ["main.py", "app.py", "server.py", "run.py", "api.py"]:
            priority_file = next(
                (k for k in boosted if os.path.basename(k) == priority_basename), None
            )
            if not priority_file:
                continue
            content = context.get(f"entrypoint_{priority_basename}", "").lower()
            if not content:
                continue
            if "fastapi" in content:
                detected_framework = "fastapi"
                context["fastapi_entry_file"] = priority_file
                for line in context.get(f"entrypoint_{priority_basename}", "").splitlines():
                    if "fastapi()" in line.lower() and "=" in line:
                        app_variable = line.split("=")[0].strip()
                        break
                print(f"[Agent] ✅ Detected FastAPI in {priority_file} directly")
                break
            elif "flask" in content:
                detected_framework = "flask"
                context["flask_entry_file"] = priority_file
                for line in context.get(f"entrypoint_{priority_basename}", "").splitlines():
                    if "flask(" in line.lower() and "=" in line:
                        app_variable = line.split("=")[0].strip()
                        break
                print(f"[Agent] ✅ Detected Flask in {priority_file} directly")
                break
            elif "django" in content or priority_basename == "manage.py":
                detected_framework = "django"
                print(f"[Agent] ✅ Detected Django in {priority_file} directly")
                break
            elif "uvicorn" in content or "starlette" in content:
                detected_framework = "fastapi"
                context["fastapi_entry_file"] = priority_file
                print(f"[Agent] ✅ Detected FastAPI (uvicorn) in {priority_file} directly")
                break

        # ── Fallback: scan all entry points ───────────────────────────
        if detected_framework == "unknown":
            for e in entry_points_found:
                basename = os.path.basename(e)
                content = context.get(f"entrypoint_{basename}", "").lower()
                if "fastapi" in content:
                    detected_framework = "fastapi"
                    for line in context.get(f"entrypoint_{basename}", "").splitlines():
                        if "fastapi()" in line.lower() and "=" in line:
                            app_variable = line.split("=")[0].strip()
                    context["fastapi_entry_file"] = e
                    break
                elif "flask" in content:
                    detected_framework = "flask"
                    for line in context.get(f"entrypoint_{basename}", "").splitlines():
                        if "flask(" in line.lower() and "=" in line:
                            app_variable = line.split("=")[0].strip()
                    context["flask_entry_file"] = e
                    break
                elif "django" in content or basename == "manage.py":
                    detected_framework = "django"
                    break
                elif "uvicorn" in content or "starlette" in content:
                    detected_framework = "fastapi"
                    context["fastapi_entry_file"] = e
                    break

    if detected_framework == "unknown" and entry_points_found:
        print("[Agent] ⚠️  Framework unknown — asking LLM to detect...")
        all_context_text = ""
        for e in entry_points_found:
            basename = os.path.basename(e)
            all_context_text += f"\n--- {e} ---\n{context.get(f'entrypoint_{basename}', '')}\n"
        for key, val in context.items():
            if key.startswith("dep_file_"):
                fname = key.replace("dep_file_", "")
                all_context_text += f"\n--- {fname} ---\n{val[:1000]}\n"

        client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"""
Analyze these files and detect the framework/type of this project.
Files:\n{all_context_text[:4000]}
Return ONLY valid JSON:
{{"framework":"fastapi/flask/streamlit/etc","language":"python/nodejs/go/etc","cmd":"exact start command","port":"8000"}}
"""}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            lines = [l for l in raw.splitlines() if not l.strip().startswith("```")]
            raw   = "\n".join(lines).strip()
        try:
            detected           = json.loads(raw)
            detected_framework = detected.get("framework", "unknown")
            if detected_lang == "unknown":
                detected_lang = detected.get("language", "unknown")
            context["llm_detected_cmd"]  = detected.get("cmd", "")
            context["llm_detected_port"] = detected.get("port", "8000")
            print(f"[Agent] 🤖 LLM detected: {detected_framework} | cmd: {context['llm_detected_cmd']}")
        except Exception as e:
            print(f"[Agent] ⚠️  Could not parse LLM framework response: {e}")

    if "package.json" in all_files:
        if detected_lang == "unknown":
            detected_lang = "nodejs"
        pkg_lower = context.get("dep_file_package.json", "").lower()
        if '"next"'            in pkg_lower: detected_framework="nextjs";  is_frontend=True; frontend_type="nextjs";  build_output_dir=".next"
        elif '"react"'         in pkg_lower: detected_framework="react";   is_frontend=True; frontend_type="react";   build_output_dir="build"
        elif '"vue"'           in pkg_lower:
            if '"nuxt"'        in pkg_lower: detected_framework="nuxt";    is_frontend=True; frontend_type="nuxt";    build_output_dir=".output"
            else:                            detected_framework="vue";     is_frontend=True; frontend_type="vue";     build_output_dir="dist"
        elif '"@angular/core"' in pkg_lower: detected_framework="angular"; is_frontend=True; frontend_type="angular"; build_output_dir="dist"
        elif '"svelte"'        in pkg_lower: detected_framework="svelte";  is_frontend=True; frontend_type="svelte";  build_output_dir="build"
        elif '"vite"'          in pkg_lower: detected_framework="vite";    is_frontend=True; frontend_type="vite";    build_output_dir="dist"
        elif '"express"'       in pkg_lower: detected_framework="express"
        elif '"fastify"'       in pkg_lower: detected_framework="fastify"
        for key in ["dep_file_.nvmrc", "dep_file_.node-version"]:
            val = context.get(key, "").strip()
            if val:
                node_version = val.replace("v", "").split(".")[0]

    if "pom.xml" in all_files or "build.gradle" in all_files: detected_lang = "java"
    if "go.mod"        in all_files: detected_lang = "go"
    if "Gemfile"       in all_files:
        detected_lang = "ruby"
        detected_framework = "rails" if "rails" in context.get("dep_file_Gemfile", "").lower() else "ruby"
    if "composer.json" in all_files: detected_lang = "php"
    if "Cargo.toml"    in all_files: detected_lang = "rust"
    # Fix: also catch javascript projects that are actually static HTML
# A static site has index.html but NO package.json
    if "index.html" in all_files and "package.json" not in all_files:
        if detected_lang in ("unknown", "javascript"):
            detected_lang      = "html"
            detected_framework = "static"
            is_frontend        = True
            frontend_type      = "static_html"

    if detected_lang == "python":
        for key in ["dep_file_runtime.txt", "dep_file_.python-version"]:
            val = context.get(key, "")
            for v in ["3.8", "3.9", "3.10", "3.11", "3.12"]:
                if v in val:
                    python_version = v; break
        if "dep_file_requirements.txt" not in context:
            context["missing_requirements_warning"] = (
                "No requirements.txt. Scan entry points for all imports and install them."
            )

    context.update({
        "detected_language":  detected_lang,
        "detected_framework": detected_framework,
        "app_variable_name":  app_variable,
        "python_version":     python_version,
        "node_version":       node_version,
        "is_frontend":        is_frontend,
        "frontend_type":      frontend_type,
        "is_ml":              is_ml,
        "ml_type":            ml_type,
        "ml_frameworks":      ml_frameworks,
        "uses_conda":         uses_conda,
        "uses_gpu":           uses_gpu,
        "build_output_dir":   build_output_dir,
    })

    print(f"[Agent] Lang={detected_lang} | Framework={detected_framework} | "
          f"ML={is_ml} | MLType={ml_type} | GPU={uses_gpu} | Conda={uses_conda}")
    return context


# ══════════════════════════════════════════════════════════════════════════════
# DOCKERFILE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_dockerfile_with_openai(folder, openai_api_key):
    context      = deep_scan_repo(folder)
    lang         = context["detected_language"]
    framework    = context["detected_framework"]
    app_var      = context["app_variable_name"]
    py_ver       = context["python_version"]
    node_ver     = context["node_version"]
    fe_type      = context["frontend_type"]
    is_ml        = context["is_ml"]
    ml_type      = context["ml_type"]
    ml_libs      = context["ml_frameworks"]
    uses_gpu     = context["uses_gpu"]
    entries      = context["entry_points_found"]
    entry        = entries[0] if entries else "app.py"
    entry_base   = os.path.basename(entry)
    entry_module = entry.replace("\\", "/").replace("/", ".").replace(".py", "")

    context_text = f"""
=== REPO ANALYSIS ===
All root files: {context['all_files']}
Language: {lang} | Framework: {framework}
ML project: {is_ml} | ML type: {ml_type} | ML libs: {ml_libs}
GPU: {uses_gpu} | Conda: {context['uses_conda']}
Frontend: {context['is_frontend']} | Frontend type: {fe_type}
Python version: {py_ver} | Node version: {node_ver}
Entry points: {entries}
Entry file (basename): {entry_base}
Entry module: {entry_module}
Model files: {context.get('model_files_found', [])}
Notebooks: {context.get('notebooks_found', [])}
"""
    for key, value in context.items():
        if key.startswith("dep_file_") or key.startswith("entrypoint_"):
            fname = key.replace("dep_file_", "").replace("entrypoint_", "")
            context_text += f"\n=== {fname} ===\n{value}\n"

    if "missing_requirements_warning" in context:
        context_text += f"\n⚠️  {context['missing_requirements_warning']}\n"

    notes_path = os.path.join(folder, "_agent_notes.txt")
    if os.path.exists(notes_path):
        with open(notes_path) as f:
            user_notes = f.read().strip()
        if user_notes:
            context_text += f"\n=== USER NOTES ===\n{user_notes}\n"
            print(f"[Agent] 📝 Using user notes in Dockerfile generation")

    if context.get("llm_detected_cmd"):
        context_text += f"\n=== LLM DETECTED ===\nCmd: {context['llm_detected_cmd']}\nPort: {context.get('llm_detected_port', '8000')}\n"

    gpu_base   = "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04"
    base_image = gpu_base if uses_gpu else f"python:{py_ver}-slim"

    specific_instructions = ""

    if ml_type == "streamlit":
        e = context.get("streamlit_entry_file", entry_base)
        specific_instructions = f"""
PROJECT: Streamlit App
FROM {base_image}
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y build-essential libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
RUN pip install --no-cache-dir streamlit
COPY . .
EXPOSE 8501
CMD ["sh", "-c", "streamlit run {e} --server.port=${{PORT:-8501}} --server.address=0.0.0.0 --server.headless=true"]
ML libs: {ml_libs}
"""
    elif ml_type == "gradio":
        e = context.get("gradio_entry_file", entry_base)
        specific_instructions = f"""
PROJECT: Gradio App
FROM {base_image}
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
RUN apt-get update && apt-get install -y build-essential libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
RUN pip install --no-cache-dir gradio
COPY . .
EXPOSE 7860
CMD ["sh", "-c", "python {e}"]
ML libs: {ml_libs}
"""
    elif ml_type == "jupyter":
        specific_instructions = f"""
PROJECT: Jupyter Notebook Server
FROM python:{py_ver}-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt* ./
RUN pip install --no-cache-dir jupyter notebook jupyterlab
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
COPY . .
EXPOSE 8888
CMD ["sh", "-c", "jupyter notebook --ip=0.0.0.0 --port=${{PORT:-8888}} --no-browser --allow-root --NotebookApp.token=''"]
"""
    elif ml_type in ("fastapi_ml", "flask_ml"):
        e     = context.get("fastapi_entry_file") or context.get("flask_entry_file", entry)
        mod   = e.replace("\\", "/").replace("/", ".").replace(".py", "")
        is_fa = ml_type == "fastapi_ml"
        cmd   = (f"uvicorn {mod}:{app_var} --host 0.0.0.0 --port ${{PORT:-8000}}"
                 if is_fa else "flask run --host=0.0.0.0 --port=${PORT:-5000}")
        specific_instructions = f"""
PROJECT: {'FastAPI' if is_fa else 'Flask'} ML API
FROM {base_image}
WORKDIR /app
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y build-essential libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["sh", "-c", "{cmd}"]
ML libs: {ml_libs}
"""
    elif ml_type == "ml_script":
        e = context.get("ml_script_entry", entry_base)
        specific_instructions = f"""
PROJECT: Plain ML Script
FROM {base_image}
WORKDIR /app
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y build-essential libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
COPY . .
CMD ["python", "{e}"]
ML libs: {ml_libs}
"""
    elif framework == "fastapi":
        e   = context.get("fastapi_entry_file", entry)
        mod = e.replace("\\", "/").replace("/", ".").replace(".py", "")
        specific_instructions = f"""
PROJECT: FastAPI
FROM python:{py_ver}-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
COPY . .
CMD ["sh", "-c", "uvicorn {mod}:{app_var} --host 0.0.0.0 --port ${{PORT:-8000}}"]
"""
    elif framework == "flask":
        e = context.get("flask_entry_file", entry)
        specific_instructions = f"""
PROJECT: Flask
FROM python:{py_ver}-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP={e}
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
COPY . .
CMD ["sh", "-c", "flask run --host=0.0.0.0 --port=${{PORT:-5000}}"]
"""
    elif framework == "django":
        specific_instructions = f"""
PROJECT: Django
FROM python:{py_ver}-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
COPY . .
CMD ["sh", "-c", "python manage.py runserver 0.0.0.0:${{PORT:-8000}}"]
"""
    elif fe_type in ("react", "vue", "angular", "vite", "svelte"):
        specific_instructions = f"""
PROJECT: {fe_type.title()} SPA
Multi-stage: node:{node_ver}-alpine builder + nginx:alpine
RUN npm ci && npm run build
COPY build output to /usr/share/nginx/html
CMD ["nginx", "-g", "daemon off;"]
Use try_files $uri /index.html for SPA routing
"""
    elif fe_type == "nextjs":
        specific_instructions = f"""
PROJECT: Next.js
Multi-stage: node:{node_ver}-alpine builder + runner
RUN npm ci && npm run build
CMD ["sh", "-c", "npm start -- --port ${{PORT:-3000}}"]
"""
    elif fe_type == "static_html":
        specific_instructions = """
PROJECT: Static HTML
FROM nginx:alpine
COPY . /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
"""
    elif lang == "java":
        specific_instructions = """
PROJECT: Java
Multi-stage: maven:3.9-eclipse-temurin-17 + eclipse-temurin:17-jre-slim
RUN mvn package -DskipTests
CMD ["java", "-jar", "target/app.jar"]
"""
    elif lang == "go":
        specific_instructions = """
PROJECT: Go
Multi-stage: golang:1.21-alpine + alpine:3.18
RUN go build -o main .
CMD ["./main"]
"""
    elif lang == "ruby":
        specific_instructions = f"""
PROJECT: Ruby {'Rails' if framework == 'rails' else ''}
FROM ruby:3.2-slim
RUN bundle install
CMD rails server or ruby {entry_base} with $PORT
"""
    else:
        cmd  = context.get("llm_detected_cmd", f"python {entry_base}")
        port = context.get("llm_detected_port", "8000")
        specific_instructions = f"""
PROJECT: Custom ({framework or 'unknown'}) — Language: {lang}
FROM python:{py_ver}-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
COPY . .
EXPOSE {port}
CMD ["sh", "-c", "{cmd}"]
NOTE: Use ${{PORT:-{port}}} pattern. Entry file is {entry_base}.
"""
        print(f"[Agent] 🤖 Using cmd: {cmd} on port {port}")

    prompt = f"""You are a Docker expert. Generate a production-ready Dockerfile.

{context_text}

INSTRUCTIONS:
{specific_instructions}

CRITICAL RULES:
1. NEVER hardcode ports — always use ${{PORT:-DEFAULT}}
2. Python: always set ENV PYTHONUNBUFFERED=1 and PYTHONDONTWRITEBYTECODE=1
3. Install ALL dependencies BEFORE COPY . . (layer caching)
4. The entry module is: {entry_module} — CMD must use this exact module path for uvicorn
5. WORKDIR must be /app — all files are copied there via COPY . .
6. Streamlit: --server.address=0.0.0.0 --server.headless=true --server.port=${{PORT:-8501}}
7. Gradio: ENV GRADIO_SERVER_NAME=0.0.0.0
8. XGBoost/LightGBM: apt-get install -y libgomp1
9. OpenCV: apt-get install -y libgl1-mesa-glx libglib2.0-0
10. HuggingFace: use python:3.11-slim not alpine
11. No requirements.txt: scan imports from entry file and RUN pip install each one
12. Output ONLY the raw Dockerfile — no markdown, no backticks, no explanation
"""

    client   = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"Output ONLY raw Dockerfile. No markdown. No backticks. Entry module is {entry_module}. Always use ${{PORT:-DEFAULT}}."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.1,
    )

    dockerfile_content = response.choices[0].message.content.strip()

    if dockerfile_content.startswith("```"):
        lines = [l for l in dockerfile_content.splitlines() if not l.strip().startswith("```")]
        dockerfile_content = "\n".join(lines).strip()

    for port in ["8000", "8080", "5000", "3000", "8501", "7860", "8888"]:
        if f"--port {port}" in dockerfile_content and "${PORT" not in dockerfile_content:
            dockerfile_content = dockerfile_content.replace(
                f"--port {port}", f"--port ${{PORT:-{port}}}")

    if lang == "python" and "PYTHONUNBUFFERED" not in dockerfile_content:
        dockerfile_content = dockerfile_content.replace(
            "WORKDIR /app",
            "WORKDIR /app\n\nENV PYTHONUNBUFFERED=1\nENV PYTHONDONTWRITEBYTECODE=1")

    if ml_type == "streamlit":
        if "--server.headless" not in dockerfile_content:
            dockerfile_content = dockerfile_content.replace(
                "streamlit run", "streamlit run --server.headless=true")
        if "--server.address" not in dockerfile_content:
            dockerfile_content = dockerfile_content.replace(
                "streamlit run", "streamlit run --server.address=0.0.0.0")

    path = os.path.join(folder, "Dockerfile")
    with open(path, "w", encoding="utf-8") as f:
        f.write(dockerfile_content)

    print(f"\n[Agent] ── Dockerfile ({lang}/{framework or ml_type}) ──")
    print(dockerfile_content)
    print("[Agent] ──────────────────────────────────────────────────\n")

    # ── Auto-generate docker-compose.yml for DB-dependent apps ────────
    DB_DRIVERS = ["psycopg2", "asyncpg", "pymysql", "mysqlclient",
                  "pymongo", "motor", "aiomysql", "aiopg"]
    req_content = context.get("dep_file_requirements.txt", "").lower()
    db_needed   = any(kw in req_content for kw in DB_DRIVERS)

    if db_needed:
        # ── Detect DB type ────────────────────────────────────────────
        if any(kw in req_content for kw in ["psycopg2", "asyncpg", "aiopg"]):
            db_type = "postgres"
        elif any(kw in req_content for kw in ["pymysql", "mysqlclient", "aiomysql"]):
            db_type = "mysql"
        elif "pymongo" in req_content or "motor" in req_content:
            db_type = "mongo"
        else:
            db_type = "postgres"

        # ── Read existing .env first ──────────────────────────────────
        env_file = os.path.join(folder, ".env")
        env_vars = {}
        if os.path.exists(env_file):
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    env_vars[k.strip()] = v.strip().strip('"').strip("'")

        # ── Collect DB credentials ────────────────────────────────────
        print(f"\n[Agent] 🗄️  Database detected ({db_type}) — collecting credentials for docker-compose...")
        # ── Parse credentials from existing DATABASE_URL if present ───
        existing_url = env_vars.get("DATABASE_URL", "")
        if existing_url:
            try:
                # Parse postgresql://user:pass@host:port/dbname
                import re
                match = re.match(
                    r"(?:postgresql|mysql|mongodb)(?:\+\w+)?://([^:]+):([^@]+)@[^:]+:\d+/(\w+)",
                    existing_url
                )
                if match:
                    parsed_user = match.group(1)
                    parsed_pass = match.group(2)
                    parsed_db   = match.group(3)
                    # Inject into env_vars so ask() picks them up
                    env_vars.setdefault("POSTGRES_USER",     parsed_user)
                    env_vars.setdefault("POSTGRES_PASSWORD", parsed_pass)
                    env_vars.setdefault("POSTGRES_DB",       parsed_db)
                    env_vars.setdefault("MYSQL_ROOT_PASSWORD", parsed_pass)
                    env_vars.setdefault("MYSQL_DATABASE",      parsed_db)
                    print(f"[Agent] 🔑 Parsed credentials from DATABASE_URL")
            except Exception:
                pass

        print(f"[Agent] ℹ️  Docker will auto-create the DB — no local install needed!")
        print(f"[Agent] ℹ️  Just press Enter to accept defaults.\n")

        def ask(label, env_key, default):
            if env_key and env_key in env_vars:
                print(f"  ✅ {label}: loaded from .env ({env_vars[env_key]})")
                return env_vars[env_key]
            val = input(f"  {label} [{default}]: ").strip()
            return val if val else default

        if db_type == "postgres":
            db_user    = ask("Postgres user",     "POSTGRES_USER",     "postgres")
            db_pass    = ask("Postgres password", "POSTGRES_PASSWORD", "password")
            db_name    = ask("Postgres DB name",  "POSTGRES_DB",       "app")
            db_service = f"""  db:
    image: postgres:15
    restart: always
    environment:
      POSTGRES_USER: {db_user}
      POSTGRES_PASSWORD: {db_pass}
      POSTGRES_DB: {db_name}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data"""
            db_url  = f"postgresql://{db_user}:{db_pass}@db:5432/{db_name}"
            volumes = "\nvolumes:\n  postgres_data:"

        elif db_type == "mysql":
            db_pass    = ask("MySQL root password", "MYSQL_ROOT_PASSWORD", "password")
            db_name    = ask("MySQL DB name",       "MYSQL_DATABASE",      "app")
            db_service = f"""  db:
    image: mysql:8
    restart: always
    environment:
      MYSQL_ROOT_PASSWORD: {db_pass}
      MYSQL_DATABASE: {db_name}
    ports:
      - "3306:3306"
    volumes:
      - mysql_data:/var/lib/mysql"""
            db_url  = f"mysql+pymysql://root:{db_pass}@db:3306/{db_name}"
            volumes = "\nvolumes:\n  mysql_data:"

        elif db_type == "mongo":
            db_name    = ask("MongoDB DB name", "MONGO_DB", "app")
            db_service = f"""  db:
    image: mongo:6
    restart: always
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db"""
            db_url  = f"mongodb://db:27017/{db_name}"
            volumes = "\nvolumes:\n  mongo_data:"

        # ── Use DATABASE_URL from .env only if it's a hosted URL ──────
        existing_db_url  = env_vars.get("DATABASE_URL", "")
        is_localhost_url = any(h in existing_db_url for h in
                               ["localhost", "127.0.0.1", "0.0.0.0"])

        if existing_db_url and not is_localhost_url:
            final_db_url = existing_db_url
            print(f"\n[Agent] ℹ️  Using hosted DATABASE_URL from .env")
        else:
            final_db_url = db_url
            if existing_db_url and is_localhost_url:
                print(f"\n[Agent] ℹ️  .env has localhost DATABASE_URL — replacing with docker-compose internal URL")

        app_port = detect_port_from_dockerfile(folder, fallback="8000")

        compose_content = f"""version: "3.8"
services:
{db_service}

  app:
    build: .
    ports:
      - "{app_port}:{app_port}"
    environment:
      PORT: "{app_port}"
      DATABASE_URL: "{final_db_url}"
    depends_on:
      - db
    restart: always
{volumes}
"""
        compose_path = os.path.join(folder, "docker-compose.yml")
        with open(compose_path, "w", encoding="utf-8") as f:
            f.write(compose_content)

        print(f"\n[Agent] ✅ docker-compose.yml generated ({db_type} + app)")
        print(f"[Agent] ℹ️  Run locally with: docker-compose up --build")
        print(f"[Agent] ℹ️  App will be at:   http://localhost:{app_port}")
        print(f"[Agent] ℹ️  DB URL in compose: {final_db_url}\n")

    return dockerfile_content, context

# ══════════════════════════════════════════════════════════════════════════════
# DOCKER TESTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_test_port(ml_type, framework, folder=None):
    if folder:
        dockerfile_path = os.path.join(folder, "Dockerfile")
        if os.path.exists(dockerfile_path):
            with open(dockerfile_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.upper().startswith("EXPOSE"):
                        parts = line.split()
                        if len(parts) >= 2:
                            port = parts[1].strip()
                            if "${PORT" in port:
                                import re
                                match = re.search(r'\$\{PORT:-(\d+)\}', port)
                                if match:
                                    return match.group(1)
                            elif port.isdigit():
                                return port

    port_map = {
        "streamlit": "8501", "gradio": "7860", "jupyter": "8888",
        "fastapi_ml": "8000", "flask_ml": "5000", "ml_script": None,
        "fastapi": "8000", "flask": "5000", "django": "8000",
        "nextjs": "3000", "nuxt": "3000", "react": "80", "vue": "80",
        "angular": "80", "vite": "80", "svelte": "80",
        "static_html": "80", "static": "80", "none": "80",
        "express": "3000", "fastify": "3000",
    }
    key = ml_type if ml_type and ml_type != "unknown" else framework
    if key in ("unknown", "none", "", None):
        key = "static_html"
    return port_map.get(key, "8000")

def detect_port_from_dockerfile(folder, fallback="8000"):
    """Read EXPOSE port from Dockerfile dynamically — works for any project."""
    import re
    dockerfile_path = os.path.join(folder, "Dockerfile")
    if not os.path.exists(dockerfile_path):
        return fallback
    with open(dockerfile_path, "r") as f:
        content = f.read()
    for line in content.splitlines():
        line = line.strip()
        if line.upper().startswith("EXPOSE"):
            parts = line.split()
            if len(parts) >= 2:
                port = parts[1].strip()
                match = re.search(r'\$\{PORT:-(\d+)\}', port)
                if match:
                    return match.group(1)
                elif port.isdigit():
                    return port
    return fallback

def get_startup_wait(ml_type, framework):
    wait_map = {
        "streamlit": 15, "gradio": 15, "jupyter": 15,
        "fastapi_ml": 10, "flask_ml": 10, "huggingface": 30,
        "fastapi": 8, "flask": 8, "django": 8,
        "nextjs": 15, "react": 5, "vue": 5, "angular": 5,
    }
    key = ml_type if ml_type and ml_type != "unknown" else framework
    return wait_map.get(key, 10)

def get_container_logs(container_name):
    result = subprocess.run(
        ["docker", "logs", "--tail", "100", container_name],
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    return result.stdout + result.stderr

def cleanup_test_container(container_name, image_tag):
    print(f"[Test] Cleaning up test container and image...")
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
    subprocess.run(["docker", "rmi", "-f", image_tag],     capture_output=True)
    print(f"[Test] Cleanup done")

def fix_dockerfile_with_llm(dockerfile_path, error_output, error_type, context, openai_api_key):
    with open(dockerfile_path, "r", encoding="utf-8", errors="replace") as f:
        current_dockerfile = f.read()

    # ── Get actual entry file — never let LLM guess ───────────────────
    entries      = context.get("entry_points_found", [])
    entry        = entries[0] if entries else "app.py"
    entry_base   = os.path.basename(entry)
    # ── CHANGE 8: convert rel path to Python module notation ──
    entry_module = entry.replace("\\", "/").replace("/", ".").replace(".py", "")
    app_var      = context.get("app_variable_name", "app")
    framework    = context.get("detected_framework", "unknown")
    ml_type      = context.get("ml_type", "unknown")

    # ── Build correct CMD so LLM cannot get it wrong ──────────────────
    if framework == "fastapi" or ml_type == "fastapi_ml":
        e   = context.get("fastapi_entry_file", entry)
        mod = e.replace("\\", "/").replace("/", ".").replace(".py", "")
        correct_cmd = f'CMD ["sh", "-c", "uvicorn {mod}:{app_var} --host 0.0.0.0 --port ${{PORT:-8000}}"]'
    elif framework == "flask" or ml_type == "flask_ml":
        correct_cmd = f'CMD ["sh", "-c", "flask run --host=0.0.0.0 --port=${{PORT:-5000}}"]'
    elif framework == "django":
        correct_cmd = f'CMD ["sh", "-c", "python manage.py runserver 0.0.0.0:${{PORT:-8000}}"]'
    elif ml_type == "streamlit":
        e = context.get("streamlit_entry_file", entry_base)
        correct_cmd = f'CMD ["sh", "-c", "streamlit run {e} --server.port=${{PORT:-8501}} --server.address=0.0.0.0 --server.headless=true"]'
    elif ml_type == "gradio":
        e = context.get("gradio_entry_file", entry_base)
        correct_cmd = f'CMD ["sh", "-c", "python {e}"]'
    else:
        correct_cmd = f'CMD ["sh", "-c", "python {entry_base}"]'

    error_descriptions = {
        "build":        "The Docker image failed to BUILD with this error",
        "runtime":      "The Docker container failed to START with this error",
        "runtime_exit": "The Docker container STARTED but then EXITED immediately with this error",
        "no_response":  "The Docker container is running but NOT RESPONDING to HTTP requests",
    }
    error_desc = error_descriptions.get(error_type, "There was an error")

    prompt = f"""You are a Docker expert. {error_desc}:

ERROR OUTPUT:
{error_output[-3000:]}

CURRENT DOCKERFILE:
{current_dockerfile}

PROJECT INFO:
- Language:     {context.get("detected_language", "unknown")}
- Framework:    {framework}
- ML type:      {ml_type}
- ML libraries: {context.get("ml_frameworks", [])}
- Entry file:   {entry_base}        ← ACTUAL FILENAME, USE THIS EXACTLY
- Module name:  {entry_module}      ← USE FOR uvicorn (e.g. app.main for app/main.py)
- App variable: {app_var}           ← USE FOR uvicorn app variable

CORRECT CMD FOR THIS PROJECT — USE EXACTLY THIS:
{correct_cmd}

CRITICAL RULES — NEVER VIOLATE:
1. CMD MUST use sh -c shell form:
   ✅ CMD ["sh", "-c", "uvicorn {entry_module}:{app_var} --host 0.0.0.0 --port ${{PORT:-8000}}"]
   ❌ CMD ["uvicorn", "...", "${{PORT:-8000}}"]  ← shell vars do NOT expand in JSON array
2. Module name MUST be "{entry_module}" — if file is app/main.py use "app.main" not "main"
3. App variable MUST be "{app_var}"
4. Port MUST use ${{PORT:-DEFAULT}} inside sh -c

COMMON FIXES:
- "Could not import module": wrong module name — use "{entry_module}"
- "${{PORT:-8000}} is not a valid integer": switch to sh -c form immediately
- ModuleNotFoundError: add missing pip install
- libgomp not found: RUN apt-get install -y libgomp1
- libGL not found: RUN apt-get install -y libgl1-mesa-glx libglib2.0-0
- Container exits: check CMD references correct module "{entry_module}"

Output ONLY the fixed raw Dockerfile. No markdown, no backticks, no explanation.
"""

    client   = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"Output ONLY the fixed raw Dockerfile. No markdown. No backticks. Entry module is '{entry_module}', app variable is '{app_var}'. CMD must use sh -c form."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
    )

    fixed_content = response.choices[0].message.content.strip()
    if fixed_content.startswith("```"):
        lines = [l for l in fixed_content.splitlines() if not l.strip().startswith("```")]
        fixed_content = "\n".join(lines).strip()

    if not fixed_content.startswith("FROM"):
        print("[Test] ⚠️  LLM response doesn't look like a Dockerfile, skipping fix")
        return False

    # ── Safety net: force convert JSON array PORT to sh -c ────────────
    import re
    if re.search(r'CMD \[.*?\$\{PORT', fixed_content, re.DOTALL):
        print("[Test] ⚠️  Post-fix: JSON array CMD with ${PORT} detected — forcing sh -c")
        fixed_content = re.sub(r'CMD \[.*?\]', correct_cmd, fixed_content, flags=re.DOTALL)

    with open(dockerfile_path, "w", encoding="utf-8") as f:
        f.write(fixed_content)

    print(f"[Test] ✅ Dockerfile updated by GPT-4o fix")
    print(f"[Test] Fixed Dockerfile:\n{fixed_content}\n")
    return True


def test_docker_image(folder, app_name, context, openai_api_key, max_retries=3):
    ml_type        = context.get("ml_type", "unknown")
    framework      = context.get("detected_framework", "unknown")
    lang           = context.get("detected_language", "unknown")
    test_port      = detect_port_from_dockerfile(folder)
    image_tag      = f"{app_name}-test:latest"
    container_name = f"{app_name}-test-container"

    print(f"\n[Test] ══════════════════════════════════════════════════")
    print(f"[Test] Starting Docker image test for: {app_name}")

    # ── Check Docker is running before doing anything ─────────────
    if not _check_docker_running():
        print(f"\n[Test] ❌ Docker is NOT running!")
        print(f"[Test] 👉 Please open Docker Desktop and wait for it to fully start")
        print(f"[Test] ⏳ Waiting up to 2 minutes for Docker to start...")
        if _wait_for_docker(max_wait_seconds=120):
            print(f"[Test] ✅ Docker started — continuing test")
        else:
            print(f"\n[Test] ❌ Docker did not start in time")
            print(f"[Test] 💡 Steps to fix:")
            print(f"[Test]    1. Open Docker Desktop")
            print(f"[Test]    2. Wait for the whale icon to stop animating")
            print(f"[Test]    3. Run --resume to retry")
            return False
    print(f"[Test] Project type: {lang}/{framework or ml_type}")
    print(f"[Test] Test port: {test_port}")
    print(f"[Test] ══════════════════════════════════════════════════\n")

    dockerfile_path = os.path.join(folder, "Dockerfile")
    # ── Build-only test for database-dependent apps ───────────────────
    DB_DRIVERS = ["psycopg2", "asyncpg", "pymysql", "mysqlclient",
                  "pymongo", "motor", "redis", "aiomysql", "aiopg"]
    db_needed = any(
        kw in context.get("dep_file_requirements.txt", "").lower()
        for kw in DB_DRIVERS
    )
    if db_needed:
        print(f"[Test] ℹ️  Database dependency detected — switching to build-only test mode")
        print(f"[Test] ℹ️  Runtime skipped — app needs external DB (connects at deploy time via DATABASE_URL)")
        for attempt in range(1, max_retries + 1):
            print(f"\n[Test] ── Build attempt {attempt}/{max_retries} ──────────────────────")
            build_result = subprocess.run(
                ["docker", "build", "-t", image_tag, "."],
                cwd=folder, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
            )
            if build_result.returncode == 0:
                print(f"[Test] ✅✅✅ DOCKER BUILD PASSED ✅✅✅")
                print(f"[Test] ℹ️  Dockerfile is correct — DB will connect when env vars are set at deploy time")
                subprocess.run(["docker", "rmi", "-f", image_tag], capture_output=True)
                return True
            else:
                print(f"[Test] ❌ Build FAILED on attempt {attempt}")
                print(f"[Test] Build error:\n{build_result.stderr[-2000:]}")
                if attempt < max_retries:
                    print(f"[Test] 🔧 Asking GPT-4o to fix the Dockerfile...")
                    fix_dockerfile_with_llm(
                        dockerfile_path, error_output=build_result.stderr,
                        error_type="build", context=context, openai_api_key=openai_api_key,
                    )
                else:
                    print(f"[Test] ❌ All {max_retries} build attempts failed")
                    return False
                
    for attempt in range(1, max_retries + 1):
        print(f"\n[Test] ── Attempt {attempt}/{max_retries} ──────────────────────")
        print(f"[Test] Building image: {image_tag}")
        build_result = subprocess.run(
            ["docker", "build", "-t", image_tag, "."],
            cwd=folder, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )

        if build_result.returncode != 0:
            print(f"[Test] ❌ Build FAILED on attempt {attempt}")
            print(f"[Test] Build error:\n{build_result.stderr[-3000:]}")
            if attempt < max_retries:
                print(f"[Test] 🔧 Asking GPT-4o to fix the Dockerfile...")
                fixed = fix_dockerfile_with_llm(
                    dockerfile_path, error_output=build_result.stderr,
                    error_type="build", context=context, openai_api_key=openai_api_key,
                )
                if fixed:
                    print(f"[Test] ✅ Dockerfile updated, retrying build...")
                    continue
                else:
                    print(f"[Test] ❌ Could not auto-fix Dockerfile")
                    break
            else:
                print(f"[Test] ❌ All {max_retries} build attempts failed")
                return False

        print(f"[Test] ✅ Image built successfully: {image_tag}")
        test_port = detect_port_from_dockerfile(folder)
        print(f"[Test] Using port: {test_port}")

        if test_port is None:
            print(f"[Test] ℹ️  ML script project — no web server to test")
            print(f"[Test] ✅ Build passed — skipping runtime test")
            cleanup_test_container(container_name, image_tag)
            return True

        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
        print(f"[Test] Starting container on port {test_port}...")
        run_result = subprocess.run(
            ["docker", "run", "-d", "--name", container_name,
             "-p", f"{test_port}:{test_port}", "-e", f"PORT={test_port}", image_tag],
            capture_output=True, text=True,
        )

        if run_result.returncode != 0:
            print(f"[Test] ❌ Container failed to start")
            logs = get_container_logs(container_name)
            print(f"[Test] Container logs:\n{logs}")
            if attempt < max_retries:
                fix_dockerfile_with_llm(dockerfile_path, error_output=logs or run_result.stderr,
                                        error_type="runtime", context=context, openai_api_key=openai_api_key)
                cleanup_test_container(container_name, image_tag)
                continue
            else:
                cleanup_test_container(container_name, image_tag)
                return False

        startup_wait = get_startup_wait(ml_type, framework)
        print(f"[Test] Waiting {startup_wait}s for app to start...")
        time.sleep(startup_wait)

        if container_name not in subprocess.run(
            ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            capture_output=True, text=True
        ).stdout:
            print(f"[Test] ❌ Container exited unexpectedly")
            logs = get_container_logs(container_name)
            print(f"[Test] Container logs:\n{logs}")
            if attempt < max_retries:
                fix_dockerfile_with_llm(dockerfile_path, error_output=logs,
                                        error_type="runtime_exit", context=context, openai_api_key=openai_api_key)
                cleanup_test_container(container_name, image_tag)
                continue
            else:
                cleanup_test_container(container_name, image_tag)
                return False

        print(f"[Test] Checking if app responds on http://localhost:{test_port} ...")
        health_ok = False
        for check_attempt in range(5):
            try:
                import urllib.request
                req = urllib.request.urlopen(f"http://localhost:{test_port}", timeout=10)
                print(f"[Test] ✅ HTTP {req.getcode()} — app is responding!")
                health_ok = True
                break
            except Exception as e:
                print(f"[Test] HTTP check {check_attempt+1}/5 failed: {e}")
                time.sleep(5)

        if health_ok:
            print(f"\n[Test] ✅✅✅ DOCKER TEST PASSED ✅✅✅")
            cleanup_test_container(container_name, image_tag)
            return True
        else:
            logs = get_container_logs(container_name)
            print(f"[Test] ❌ App not responding. Logs:\n{logs}")
            if attempt < max_retries:
                fix_dockerfile_with_llm(dockerfile_path, error_output=logs,
                                        error_type="no_response", context=context, openai_api_key=openai_api_key)
                cleanup_test_container(container_name, image_tag)
                continue
            else:
                cleanup_test_container(container_name, image_tag)
                return False

    return False


def _ensure_requirements(folder: str, context: dict, openai_api_key: str):
    """Auto-generate dependency file if missing — works for all languages/frameworks."""
    lang      = context.get("detected_language", "unknown")

    if lang in ("python", "unknown"):
        dep_file = "requirements.txt"
    elif lang == "nodejs":
        dep_file = "package.json"
    elif lang == "ruby":
        dep_file = "Gemfile"
    elif lang == "go":
        dep_file = "go.mod"
    elif lang == "rust":
        dep_file = "Cargo.toml"
    elif lang == "php":
        dep_file = "composer.json"
    elif lang == "java":
        dep_file = "pom.xml"
    elif lang in ("html", "static"):
        print(f"[Agent] ℹ️  Static HTML project — no dependency file needed")
        return
    else:
        print(f"[Agent] ℹ️  Language '{lang}' — skipping dependency check")
        return

    dep_path = os.path.join(folder, dep_file)

    if os.path.exists(dep_path):
        print(f"[Agent] ✅ {dep_file} already exists — skipping generation")
        return

    print(f"\n[Agent] ⚠️  No {dep_file} found for {lang} project")
    answer = input(f"[Agent] Auto-generate {dep_file}? (y/n): ").strip().lower()
    if answer not in ("y", "yes"):
        print(f"[Agent] ℹ️  Skipping {dep_file} generation")
        return

    if lang in ("python", "unknown"):
        _generate_python_requirements(folder, context, openai_api_key, dep_path)
    elif lang == "nodejs":
        _generate_package_json(folder, context, openai_api_key, dep_path)
    elif lang == "ruby":
        _generate_gemfile(folder, context, openai_api_key, dep_path)
    elif lang == "go":
        _generate_go_mod(folder, context, dep_path)
    elif lang == "rust":
        _generate_cargo_toml(folder, context, openai_api_key, dep_path)
    elif lang == "php":
        _generate_composer_json(folder, context, openai_api_key, dep_path)
    elif lang == "java":
        _generate_pom_xml(folder, context, openai_api_key, dep_path)


# ══════════════════════════════════════════════════════════════════════════════
# PYTHON
# ══════════════════════════════════════════════════════════════════════════════

def _generate_python_requirements(folder, context, openai_api_key, dep_path):
    import ast
    import re

    print(f"[Agent] 🔍 Scanning all .py files for imports...")

    stdlib_modules = {
        "os", "sys", "re", "json", "time", "datetime", "math", "random",
        "collections", "itertools", "functools", "pathlib", "shutil",
        "subprocess", "threading", "multiprocessing", "logging", "warnings",
        "typing", "abc", "io", "copy", "enum", "dataclasses", "contextlib",
        "hashlib", "hmac", "base64", "urllib", "http", "email", "html",
        "xml", "csv", "sqlite3", "pickle", "struct", "socket", "ssl",
        "uuid", "string", "textwrap", "traceback", "inspect", "ast",
        "unittest", "argparse", "configparser", "tempfile", "glob",
        "fnmatch", "stat", "platform", "gc", "weakref", "signal",
        "builtins", "types", "operator", "dis", "tokenize", "token",
        "importlib", "pkgutil", "site", "sysconfig", "distutils",
    }

    found_imports = set()

    for walk_root, walk_dirs, walk_files in os.walk(folder):
        walk_dirs[:] = [d for d in walk_dirs if d not in
                        (".git", "__pycache__", "venv", ".venv",
                         "node_modules", "_test_venv")]
        for fname in walk_files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(walk_root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    source = f.read()
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            found_imports.add(alias.name.split(".")[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            found_imports.add(node.module.split(".")[0])
            except Exception:
                for match in re.finditer(
                    r'^(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                    source, re.MULTILINE
                ):
                    found_imports.add(match.group(1))

    third_party = sorted([
        imp for imp in found_imports
        if imp not in stdlib_modules
        and not imp.startswith("_")
        and imp != ""
    ])

    print(f"[Agent] 📦 Found imports: {third_party}")

    if not third_party:
        print(f"[Agent] ℹ️  No third-party imports found — skipping")
        return

    IMPORT_TO_PIP = {
        "cv2":             "opencv-python",
        "PIL":             "Pillow",
        "sklearn":         "scikit-learn",
        "bs4":             "beautifulsoup4",
        "yaml":            "PyYAML",
        "dotenv":          "python-dotenv",
        "googleapiclient": "google-api-python-client",
        "jwt":             "PyJWT",
        "dateutil":        "python-dateutil",
        "attr":            "attrs",
        "pkg_resources":   "setuptools",
        "magic":           "python-magic",
        "serial":          "pyserial",
        "Crypto":          "pycryptodome",
        "Image":           "Pillow",
        "telegram":        "python-telegram-bot",
        "discord":         "discord.py",
        "tweepy":          "tweepy",
        "instaloader":     "instaloader",
        "usaddress":       "usaddress",
        "gi":              "PyGObject",
        "wx":              "wxPython",
    }

    pip_packages = [IMPORT_TO_PIP.get(imp, imp) for imp in third_party]

    _refine_with_llm(
        dep_path=dep_path,
        lang="python",
        pip_packages=pip_packages,
        context=context,
        folder=folder,
        openai_api_key=openai_api_key,
        system_prompt=(
            "You are a Python packaging expert. Output ONLY a valid requirements.txt "
            "— one package per line with minimum versions. No comments, no markdown."
        ),
        user_prompt=f"""
Generate a requirements.txt for this Python project.

Detected imports (raw): {pip_packages}

Rules:
- Remove any stdlib modules that sneaked in
- Use correct pip package names (cv2 → opencv-python, PIL → Pillow, sklearn → scikit-learn etc)
- Add realistic minimum versions (e.g. streamlit>=1.28.0)
- Include framework itself if detected (streamlit, fastapi, flask etc)
- Output ONLY requirements.txt content, nothing else
""",
    )


# ══════════════════════════════════════════════════════════════════════════════
# NODE.JS
# ══════════════════════════════════════════════════════════════════════════════

def _generate_package_json(folder, context, openai_api_key, dep_path):
    import re

    print(f"[Agent] 🔍 Scanning .js/.ts files for require/import statements...")

    framework = context.get("detected_framework", "unknown")
    entries   = context.get("entry_points_found", [])
    found     = set()

    for walk_root, walk_dirs, walk_files in os.walk(folder):
        walk_dirs[:] = [d for d in walk_dirs if d not in
                        (".git", "node_modules", ".next", "dist", "build")]
        for fname in walk_files:
            if not any(fname.endswith(ext) for ext in (".js", ".ts", ".jsx", ".tsx", ".mjs")):
                continue
            fpath = os.path.join(walk_root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                for match in re.finditer(
                    r"""(?:require\(['"]|from\s+['"])([^./'"@][^'"]*?)['"]""",
                    content
                ):
                    pkg = match.group(1).split("/")[0]
                    found.add(pkg)
            except Exception:
                pass

    print(f"[Agent] 📦 Found packages: {sorted(found)}")

    _refine_with_llm(
        dep_path=dep_path,
        lang="nodejs",
        pip_packages=sorted(found),
        context=context,
        folder=folder,
        openai_api_key=openai_api_key,
        system_prompt=(
            "You are a Node.js expert. Output ONLY valid package.json content. "
            "No markdown, no explanation."
        ),
        user_prompt=f"""
Generate a package.json for this Node.js project.

Framework detected: {framework}
Detected packages: {sorted(found)}
Entry points: {entries}

Rules:
- Include correct name, version, scripts (start, build, dev)
- Add all detected packages as dependencies with realistic versions
- Add framework-specific scripts (e.g. next dev, react-scripts start)
- Include engines.node if detectable
- Output ONLY valid package.json, nothing else
""",
    )


# ══════════════════════════════════════════════════════════════════════════════
# RUBY
# ══════════════════════════════════════════════════════════════════════════════

def _generate_gemfile(folder, context, openai_api_key, dep_path):
    import re

    print(f"[Agent] 🔍 Scanning .rb files for require statements...")

    framework = context.get("detected_framework", "unknown")
    found     = set()

    for walk_root, walk_dirs, walk_files in os.walk(folder):
        walk_dirs[:] = [d for d in walk_dirs if d not in (".git", "vendor")]
        for fname in walk_files:
            if not fname.endswith(".rb"):
                continue
            fpath = os.path.join(walk_root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                for match in re.finditer(r"""require\s+['"]([^'"./][^'"]*?)['"]""", content):
                    found.add(match.group(1))
            except Exception:
                pass

    print(f"[Agent] 📦 Found gems: {sorted(found)}")

    _refine_with_llm(
        dep_path=dep_path,
        lang="ruby",
        pip_packages=sorted(found),
        context=context,
        folder=folder,
        openai_api_key=openai_api_key,
        system_prompt=(
            "You are a Ruby expert. Output ONLY valid Gemfile content. "
            "No markdown, no explanation."
        ),
        user_prompt=f"""
Generate a Gemfile for this Ruby project.

Framework: {framework}
Detected requires: {sorted(found)}

Rules:
- Start with source 'https://rubygems.org'
- Include ruby version if detectable
- Add rails and all detected gems with realistic versions
- Output ONLY valid Gemfile content, nothing else
""",
    )


# ══════════════════════════════════════════════════════════════════════════════
# GO
# ══════════════════════════════════════════════════════════════════════════════

def _generate_go_mod(folder, context, dep_path):
    import re

    print(f"[Agent] 🔍 Scanning .go files for imports...")

    found    = set()
    mod_name = os.path.basename(folder).lower().replace(" ", "-") or "myapp"

    for walk_root, walk_dirs, walk_files in os.walk(folder):
        walk_dirs[:] = [d for d in walk_dirs if d not in (".git", "vendor")]
        for fname in walk_files:
            if not fname.endswith(".go"):
                continue
            fpath = os.path.join(walk_root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                for match in re.finditer(r'"([a-zA-Z][^"]*\.[^"]+/[^"]+)"', content):
                    pkg = match.group(1)
                    parts = pkg.split("/")
                    if "." in parts[0]:
                        found.add("/".join(parts[:3]))
            except Exception:
                pass

    print(f"[Agent] 📦 Found packages: {sorted(found)}")

    go_version = "1.21"
    content    = f"module {mod_name}\n\ngo {go_version}\n"

    if found:
        content += "\nrequire (\n"
        for pkg in sorted(found):
            content += f"\t{pkg} v0.0.0\n"
        content += ")\n"

    with open(dep_path, "w") as f:
        f.write(content)

    print(f"\n[Agent] ✅ go.mod generated:")
    print(f"{'─'*40}")
    print(content)
    print(f"{'─'*40}")
    print(f"[Agent] ⚠️  Run 'go mod tidy' to fix versions\n")


# ══════════════════════════════════════════════════════════════════════════════
# RUST
# ══════════════════════════════════════════════════════════════════════════════

def _generate_cargo_toml(folder, context, openai_api_key, dep_path):
    import re

    print(f"[Agent] 🔍 Scanning .rs files for extern crate / use statements...")

    found    = set()
    app_name = os.path.basename(folder).lower().replace(" ", "-") or "myapp"

    for walk_root, walk_dirs, walk_files in os.walk(folder):
        walk_dirs[:] = [d for d in walk_dirs if d not in (".git", "target")]
        for fname in walk_files:
            if not fname.endswith(".rs"):
                continue
            fpath = os.path.join(walk_root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                for match in re.finditer(r'extern crate ([a-zA-Z_][a-zA-Z0-9_]*)', content):
                    found.add(match.group(1))
                for match in re.finditer(r'^use ([a-zA-Z_][a-zA-Z0-9_]*)::', content, re.MULTILINE):
                    found.add(match.group(1))
            except Exception:
                pass

    RUST_STDLIB = {"std", "core", "alloc", "proc_macro", "test"}
    found = found - RUST_STDLIB

    print(f"[Agent] 📦 Found crates: {sorted(found)}")

    _refine_with_llm(
        dep_path=dep_path,
        lang="rust",
        pip_packages=sorted(found),
        context=context,
        folder=folder,
        openai_api_key=openai_api_key,
        system_prompt=(
            "You are a Rust expert. Output ONLY valid Cargo.toml content. "
            "No markdown, no explanation."
        ),
        user_prompt=f"""
Generate a Cargo.toml for this Rust project.

App name: {app_name}
Detected crates: {sorted(found)}

Rules:
- Include [package] with name, version = "0.1.0", edition = "2021"
- Add all detected crates as [dependencies] with realistic versions
- Output ONLY valid Cargo.toml content, nothing else
""",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PHP
# ══════════════════════════════════════════════════════════════════════════════

def _generate_composer_json(folder, context, openai_api_key, dep_path):
    import re

    print(f"[Agent] 🔍 Scanning .php files for use/require statements...")

    framework = context.get("detected_framework", "unknown")
    found     = set()

    for walk_root, walk_dirs, walk_files in os.walk(folder):
        walk_dirs[:] = [d for d in walk_dirs if d not in (".git", "vendor")]
        for fname in walk_files:
            if not fname.endswith(".php"):
                continue
            fpath = os.path.join(walk_root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                for match in re.finditer(r'use\s+([A-Z][a-zA-Z\\]+)\\', content):
                    ns = match.group(1)
                    found.add(ns)
            except Exception:
                pass

    print(f"[Agent] 📦 Found namespaces: {sorted(found)}")

    _refine_with_llm(
        dep_path=dep_path,
        lang="php",
        pip_packages=sorted(found),
        context=context,
        folder=folder,
        openai_api_key=openai_api_key,
        system_prompt=(
            "You are a PHP/Composer expert. Output ONLY valid composer.json content. "
            "No markdown, no explanation."
        ),
        user_prompt=f"""
Generate a composer.json for this PHP project.

Framework: {framework}
Detected namespaces: {sorted(found)}

Rules:
- Include name, description, require with php version and detected packages
- Use correct packagist package names
- Add autoload psr-4 if applicable
- Output ONLY valid composer.json content, nothing else
""",
    )


# ══════════════════════════════════════════════════════════════════════════════
# JAVA
# ══════════════════════════════════════════════════════════════════════════════

def _generate_pom_xml(folder, context, openai_api_key, dep_path):
    import re

    print(f"[Agent] 🔍 Scanning .java files for import statements...")

    framework = context.get("detected_framework", "unknown")
    found     = set()
    app_name  = os.path.basename(folder).lower().replace(" ", "-") or "myapp"

    for walk_root, walk_dirs, walk_files in os.walk(folder):
        walk_dirs[:] = [d for d in walk_dirs if d not in (".git", "target", ".gradle")]
        for fname in walk_files:
            if not fname.endswith(".java"):
                continue
            fpath = os.path.join(walk_root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                for match in re.finditer(r'^import\s+([\w.]+);', content, re.MULTILINE):
                    pkg = match.group(1)
                    if not any(pkg.startswith(p) for p in ("java.", "javax.", "sun.", "com.sun.")):
                        found.add(pkg.split(".")[0] + "." + pkg.split(".")[1] if len(pkg.split(".")) > 1 else pkg)
            except Exception:
                pass

    print(f"[Agent] 📦 Found packages: {sorted(found)}")

    _refine_with_llm(
        dep_path=dep_path,
        lang="java",
        pip_packages=sorted(found),
        context=context,
        folder=folder,
        openai_api_key=openai_api_key,
        system_prompt=(
            "You are a Java/Maven expert. Output ONLY valid pom.xml content. "
            "No markdown, no explanation."
        ),
        user_prompt=f"""
Generate a pom.xml for this Java project.

App name: {app_name}
Framework: {framework}
Detected imports: {sorted(found)}

Rules:
- Include groupId, artifactId, version, packaging
- Add spring-boot-starter-parent if Spring detected
- Add all detected third-party dependencies with realistic versions
- Include maven-compiler-plugin with Java 17
- Output ONLY valid pom.xml content, nothing else
""",
    )


# ══════════════════════════════════════════════════════════════════════════════
# SHARED LLM REFINER
# ══════════════════════════════════════════════════════════════════════════════

def _refine_with_llm(dep_path, lang, pip_packages, context,
                     folder, openai_api_key, system_prompt, user_prompt):
    """Call GPT-4o to refine detected packages and write the dependency file."""
    from openai import OpenAI

    print(f"[Agent] 🤖 Asking GPT-4o to generate {os.path.basename(dep_path)}...")

    entries  = context.get("entry_points_found", [])
    snippets = ""
    for e in entries[:3]:
        fpath = os.path.join(folder, e)
        if os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    snippets += f"\n--- {e} ---\n{f.read(2000)}\n"
            except Exception:
                pass

    full_prompt = user_prompt
    if snippets:
        full_prompt += f"\n\nProject file snippets:\n{snippets}"

    client   = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": full_prompt},
        ],
        temperature=0.1,
    )

    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        lines   = [l for l in content.splitlines() if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()

    with open(dep_path, "w") as f:
        f.write(content + "\n")

    print(f"\n[Agent] ✅ {os.path.basename(dep_path)} generated:")
    print(f"{'─'*40}")
    print(content)
    print(f"{'─'*40}")
    print(f"[Agent] ℹ️  Review it in VS Code and edit if needed before continuing\n")


def run_project_locally(folder, context, openai_api_key=None, max_retries=1):
    import threading

    lang      = context.get("detected_language", "unknown")
    framework = context.get("detected_framework", "unknown")
    ml_type   = context.get("ml_type", "unknown")
    entries   = context.get("entry_points_found", [])
    entry     = entries[0] if entries else ""

    print(f"\n[LocalTest] ══════════════════════════════════════════════════")
    print(f"[LocalTest] 🧪 Starting local test")
    print(f"[LocalTest] 📁 Folder:    {folder}")
    print(f"[LocalTest] 🐍 Language:  {lang}")
    print(f"[LocalTest] 🔧 Framework: {framework}")
    print(f"[LocalTest] 🤖 ML type:   {ml_type}")
    print(f"[LocalTest] 📄 Entry:     {entry}")
    print(f"[LocalTest] ══════════════════════════════════════════════════\n")

    FRAMEWORK_TIMEOUTS = {
        "streamlit":  120,
        "gradio":     120,
        "fastapi_ml":  60,
        "flask_ml":    60,
        "fastapi":     30,
        "flask":       30,
        "django":      30,
        "nextjs":      60,
        "express":     20,
        "fastify":     20,
        "default":     60,
    }

    def get_startup_timeout():
        key  = ml_type if ml_type not in ("unknown", "none", "") else framework
        base = FRAMEWORK_TIMEOUTS.get(key, FRAMEWORK_TIMEOUTS["default"])
        ml_libs = context.get("ml_frameworks", [])
        heavy   = {"pytorch", "tensorflow", "huggingface", "transformers",
                   "opencv", "fastai", "xgboost", "lightgbm"}
        if heavy & set(ml_libs):
            base += 60
            print(f"[LocalTest] ⚠️  Heavy ML libs detected {list(heavy & set(ml_libs))} — adding 60s buffer")
        print(f"[LocalTest] ⏱️  Startup timeout: {base}s (key: {key})")
        return base

    venv_dir = os.path.join(folder, "_test_venv")
    if lang in ("python", "unknown"):
        print(f"[LocalTest] 🔨 Creating isolated venv at {venv_dir}...")
        venv_result = subprocess.run(
            [sys.executable, "-m", "venv", venv_dir],
            capture_output=True, text=True
        )
        if venv_result.returncode != 0:
            print(f"[LocalTest] ⚠️  Venv creation failed: {venv_result.stderr}")
            print(f"[LocalTest] ⚠️  Falling back to system Python")
            venv_python = sys.executable
        else:
            print(f"[LocalTest] ✅ Venv created")
            venv_python = (
                os.path.join(venv_dir, "Scripts", "python.exe")
                if os.name == "nt"
                else os.path.join(venv_dir, "bin", "python")
            )
    else:
        venv_python = sys.executable

    req_path = os.path.join(folder, "requirements.txt")
    if os.path.exists(req_path):
        print(f"[LocalTest] 📦 Installing requirements into venv...")
        result = subprocess.run(
            [venv_python, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"],
            cwd=folder, capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"[LocalTest] ✅ Requirements installed")
        else:
            print(f"[LocalTest] ⚠️  Some requirements failed:\n{result.stderr[-300:]}")
    else:
        print(f"[LocalTest] ⚠️  No requirements.txt found")

    pkg_path = os.path.join(folder, "package.json")
    if os.path.exists(pkg_path):
        node_check = subprocess.run(["node", "--version"], capture_output=True)
        if node_check.returncode == 0:
            print(f"[LocalTest] 📦 Running npm install...")
            subprocess.run(["npm", "install", "--silent"], cwd=folder, capture_output=True)
            print(f"[LocalTest] ✅ npm install done")
        else:
            print(f"[LocalTest] ⚠️  Node.js not found — skipping npm install")

    cmd             = None
    success_signals = []
    is_server       = False

    if lang in ("python", "unknown"):

        if ml_type == "streamlit" or framework == "streamlit":
            print(f"[LocalTest] 🔧 Installing streamlit into venv...")
            subprocess.run(
                [venv_python, "-m", "pip", "install", "streamlit", "--quiet"],
                capture_output=True
            )
            e   = context.get("streamlit_entry_file", entry)
            cmd = [
                venv_python, "-m", "streamlit", "run", e,
                "--server.headless=true",
                "--server.port=8501",
                "--server.address=0.0.0.0",
            ]
            success_signals = [
                "you can now view",
                "network url",
                "local url",
                "http://",
                "started server",
            ]
            is_server = True
            print(f"[LocalTest] 📋 Command: {' '.join(cmd)}")

        elif ml_type == "gradio" or framework == "gradio":
            print(f"[LocalTest] 🔧 Installing gradio into venv...")
            subprocess.run(
                [venv_python, "-m", "pip", "install", "gradio", "--quiet"],
                capture_output=True
            )
            e   = context.get("gradio_entry_file", entry)
            cmd = [venv_python, e]
            success_signals = ["running on", "local url", "gradio"]
            is_server       = True
            print(f"[LocalTest] 📋 Command: {' '.join(cmd)}")

        elif framework == "fastapi" or ml_type == "fastapi_ml":
            print(f"[LocalTest] 🔧 Installing uvicorn into venv...")
            subprocess.run(
                [venv_python, "-m", "pip", "install", "uvicorn", "--quiet"],
                capture_output=True
            )
            e       = context.get("fastapi_entry_file", entry)
            # Use path-to-module conversion for uvicorn
            mod     = e.replace("\\", "/").replace("/", ".").replace(".py", "")
            app_var = context.get("app_variable_name", "app")
            cmd     = [venv_python, "-m", "uvicorn", f"{mod}:{app_var}",
                       "--host", "0.0.0.0", "--port", "8000"]
            success_signals = ["application startup complete", "uvicorn running"]
            is_server       = True
            print(f"[LocalTest] 📋 Command: {' '.join(cmd)}")

        elif framework == "flask" or ml_type == "flask_ml":
            print(f"[LocalTest] 🔧 Installing flask into venv...")
            subprocess.run(
                [venv_python, "-m", "pip", "install", "flask", "--quiet"],
                capture_output=True
            )
            e   = context.get("flask_entry_file", entry)
            cmd = [venv_python, e]
            success_signals = ["running on", "serving flask", "debugger"]
            is_server       = True
            print(f"[LocalTest] 📋 Command: {' '.join(cmd)}")

        elif framework == "django":
            print(f"[LocalTest] 🔧 Running Django check...")
            cmd       = [venv_python, "manage.py", "check"]
            is_server = False
            print(f"[LocalTest] 📋 Command: {' '.join(cmd)}")

        elif ml_type == "ml_script":
            e         = context.get("ml_script_entry", entry)
            cmd       = [venv_python, "-m", "py_compile", e]
            is_server = False
            print(f"[LocalTest] 📋 Syntax check: {' '.join(cmd)}")

        elif entry and entry.endswith(".py"):
            cmd       = [venv_python, "-m", "py_compile", entry]
            is_server = False
            print(f"[LocalTest] 📋 Syntax check: {' '.join(cmd)}")

        else:
            print(f"[LocalTest] ℹ️  No Python entry point found — skipping")
            _cleanup_venv(venv_dir)
            return True

    elif lang == "nodejs":
        node_check = subprocess.run(["node", "--version"], capture_output=True)
        if node_check.returncode != 0:
            print(f"[LocalTest] ⚠️  Node.js not found — skipping")
            return True
        if framework in ("react", "vue", "vite", "svelte", "angular", "nextjs"):
            cmd       = ["npm", "run", "build"]
            is_server = False
        elif framework in ("express", "fastify"):
            cmd             = ["node", entry or "index.js"]
            is_server       = True
            success_signals = ["listening", "started", "running"]
        else:
            print(f"[LocalTest] ℹ️  Unknown Node framework — skipping")
            return True

    else:
        print(f"[LocalTest] ℹ️  Language '{lang}' — skipping local test")
        return True

    if not cmd:
        print(f"[LocalTest] ℹ️  No run command — skipping")
        _cleanup_venv(venv_dir)
        return True

    if not is_server:
        print(f"[LocalTest] ▶️  Running command...")
        result = subprocess.run(cmd, cwd=folder, capture_output=True, text=True)
        _cleanup_venv(venv_dir)
        if result.returncode == 0:
            print(f"[LocalTest] ✅ Project OK locally")
            return True
        else:
            print(f"[LocalTest] ❌ Failed:\n{result.stderr[-500:]}")
            return False

    startup_timeout = get_startup_timeout()
    print(f"[LocalTest] ▶️  Starting server ({startup_timeout}s timeout)...")

    collected_lines = []
    lock            = threading.Lock()

    def stream_reader(stream, label):
        try:
            for raw_line in iter(stream.readline, ""):
                line = raw_line.rstrip()
                if line:
                    print(f"[LocalTest/{label}]   {line}")
                    with lock:
                        collected_lines.append(line.lower())
        except Exception:
            pass

    proc = subprocess.Popen(
        cmd,
        cwd=folder,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    t_out = threading.Thread(target=stream_reader, args=(proc.stdout, "OUT"), daemon=True)
    t_err = threading.Thread(target=stream_reader, args=(proc.stderr, "ERR"), daemon=True)
    t_out.start()
    t_err.start()

    FATAL_KEYWORDS = [
        "modulenotfounderror", "importerror", "no module named",
        "syntaxerror", "traceback (most recent", "error:",
        "address already in use",
    ]

    startup_ok  = False
    start_time  = time.time()
    last_report = start_time

    while time.time() - start_time < startup_timeout:
        with lock:
            lines_snapshot = list(collected_lines)

        if any(sig in line for line in lines_snapshot for sig in success_signals):
            startup_ok = True
            break

        if any(kw in line for line in lines_snapshot for kw in FATAL_KEYWORDS):
            print(f"[LocalTest] ❌ Fatal error detected — stopping early")
            break

        if proc.poll() is not None:
            time.sleep(0.5)
            with lock:
                lines_snapshot = list(collected_lines)
            if any(sig in line for line in lines_snapshot for sig in success_signals):
                startup_ok = True
            break

        now = time.time()
        if now - last_report >= 15:
            elapsed   = int(now - start_time)
            remaining = int(startup_timeout - elapsed)
            print(f"[LocalTest] ⏳ Still waiting for startup... ({elapsed}s elapsed, {remaining}s remaining)")
            last_report = now

        time.sleep(0.3)

    if startup_ok:
        elapsed = int(time.time() - start_time)
        print(f"\n[LocalTest] ✅ Server started successfully (took ~{elapsed}s)")
        print(f"[LocalTest] 🌐 Open in browser:")
        port_map = {
            "streamlit": "8501", "gradio": "7860", "jupyter": "8888",
            "fastapi": "8000", "fastapi_ml": "8000",
            "flask": "5000", "flask_ml": "5000",
            "django": "8000", "express": "3000", "fastify": "3000",
        }
        key          = ml_type if ml_type not in ("unknown", "none", "") else framework
        browser_port = port_map.get(key, "8000")
        print(f"[LocalTest]    → http://localhost:{browser_port}")
        print(f"[LocalTest]    → http://127.0.0.1:{browser_port}")
        print(f"\n[LocalTest] Server is running. Press Enter to stop and continue deployment...")

        try:
            input()
        except KeyboardInterrupt:
            print(f"\n[LocalTest] Ctrl+C received — stopping server...")

        print(f"[LocalTest] 🛑 Stopping server...")
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass

        _cleanup_venv(venv_dir)
        print(f"[LocalTest] ✅ Server stopped — continuing deployment")
        return True

    else:
        elapsed = int(time.time() - start_time)
        print(f"[LocalTest] ⚠️  No startup signal after {elapsed}s — proceeding anyway")
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass
        _cleanup_venv(venv_dir)
        return False


def _cleanup_venv(venv_dir):
    if venv_dir and os.path.exists(venv_dir):
        print(f"[LocalTest] 🧹 Cleaning up venv...")
        try:
            safe_rmtree(venv_dir)
            print(f"[LocalTest] ✅ Venv removed")
        except Exception as e:
            print(f"[LocalTest] ⚠️  Could not remove venv: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# GIT OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════



def _handle_large_files(folder):
    """Detect files >100 MB in the last commit and handle them via LFS or .gitignore."""
    LIMIT = 100 * 1024 * 1024  # 100 MB

    # Check files in the last commit (already committed, not just staged)
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
        cwd=folder, capture_output=True, text=True
    )
    if result.returncode != 0:
        # Fallback: first commit — list all files tracked in HEAD
        result = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", "HEAD"],
            cwd=folder, capture_output=True, text=True
        )
    committed = [f.strip() for f in result.stdout.splitlines() if f.strip()]

    large_files = []
    for rel_path in committed:
        abs_path = os.path.join(folder, rel_path)
        if os.path.isfile(abs_path) and os.path.getsize(abs_path) > LIMIT:
            large_files.append(rel_path)

    if not large_files:
        return

    print(f"[Agent] ⚠️  Found {len(large_files)} file(s) exceeding GitHub's 100 MB limit:")
    for f in large_files:
        size_mb = os.path.getsize(os.path.join(folder, f)) / (1024 * 1024)
        print(f"         • {f} ({size_mb:.1f} MB)")

    # Check if git-lfs is available
    lfs_available = subprocess.run(
        ["git", "lfs", "version"], capture_output=True
    ).returncode == 0

    if lfs_available:
        print("[Agent] 📦 Git LFS available — tracking large files with LFS")
        subprocess.run(["git", "lfs", "install"], cwd=folder, capture_output=True)

        gitattributes_path = os.path.join(folder, ".gitattributes")
        existing_patterns = set()
        if os.path.exists(gitattributes_path):
            with open(gitattributes_path, "r") as f:
                for line in f:
                    existing_patterns.add(line.strip())

        for rel_path in large_files:
            pattern = f"{rel_path} filter=lfs diff=lfs merge=lfs -text"
            if pattern not in existing_patterns:
                subprocess.run(
                    ["git", "lfs", "track", rel_path],
                    cwd=folder, capture_output=True
                )
                print(f"[Agent] ✅ LFS tracking: {rel_path}")
            else:
                print(f"[Agent] ℹ️  {rel_path} already covered by LFS pattern — skipping")

        # Remove large files from the last commit and re-add via LFS, then amend
        subprocess.run(["git", "add", ".gitattributes"], cwd=folder, capture_output=True)
        for rel_path in large_files:
            subprocess.run(["git", "rm", "--cached", rel_path], cwd=folder, capture_output=True)
            subprocess.run(["git", "add", rel_path], cwd=folder, capture_output=True)
        subprocess.run(["git", "commit", "--amend", "--no-edit"], cwd=folder, capture_output=True)
        print("[Agent] ✅ Amended commit — large files now tracked via LFS")
    else:
        # No LFS — add large files to .gitignore and amend commit to exclude them
        print("[Agent] ℹ️  Git LFS not available — adding large files to .gitignore")
        gitignore_path = os.path.join(folder, ".gitignore")
        existing = open(gitignore_path).read() if os.path.exists(gitignore_path) else ""
        with open(gitignore_path, "a") as f:
            for rel_path in large_files:
                entry = os.path.basename(rel_path)
                if entry not in existing:
                    f.write(f"\n{entry}")
                    print(f"[Agent] ✅ Added to .gitignore: {entry}")

        for rel_path in large_files:
            subprocess.run(["git", "rm", "--cached", rel_path], cwd=folder, capture_output=True)

        subprocess.run(["git", "add", ".gitignore"], cwd=folder, capture_output=True)
        subprocess.run(["git", "commit", "--amend", "--no-edit"], cwd=folder, capture_output=True)
        print("[Agent] ✅ Amended commit — large files excluded from push")


@with_retry(max_attempts=3, delay=5, backoff=2, exceptions=(NetworkError,))
>>>>>>> Stashed changes
def push_branch(folder, fork_url, token):
    auth_url = fork_url.replace("https://", f"https://{token}@")

    try:
        result = subprocess.run(
            ["git", "push", auth_url, "ai-docker-setup", "--force"],
            cwd=folder,
            capture_output=True,
            text=True,
            timeout=120,
            encoding="utf-8",
            errors="replace"
        )
    except subprocess.TimeoutExpired:
        raise NetworkError("Git push timed out after 2 minutes — check internet")

    if result.returncode != 0:
        stderr = result.stderr.lower()
        if any(w in stderr for w in ["could not resolve", "connection", "timeout", "network"]):
            raise NetworkError(f"Git push network error: {result.stderr[:200]}")
        raise GitHubError(f"Git push failed: {result.stderr[:300]}")

    print("[Agent] ✅ Branch pushed successfully")

def create_pull_request(repo_url, token, fork_owner, default_branch):
    repo      = repo_url.replace("https://github.com/", "").rstrip("/")
    headers   = make_github_headers(token)
    check_url = f"https://api.github.com/repos/{repo}/pulls"

    repo_owner = repo.split("/")[0]
    head_ref   = "ai-docker-setup" if fork_owner == repo_owner else f"{fork_owner}:ai-docker-setup"

    existing = requests.get(check_url, headers=headers,
                            params={"head": head_ref, "state": "open"})
    if existing.status_code == 200 and existing.json():
        url = existing.json()[0]["html_url"]
        print("[Agent] PR already exists (open):", url)
        return url

    closed = requests.get(check_url, headers=headers,
                          params={"head": head_ref, "state": "closed"})
    if closed.status_code == 200 and closed.json():
        pr = closed.json()[0]
        if pr.get("merged_at"):
            # ── Check if this is a NEW PR (has new commits) or old ────
            # Don't reuse old merged PRs — always create fresh ones
            print("[Agent] ℹ️  Found old merged PR — creating new PR for latest changes")
            # Fall through to create a new PR below
        else:
            reopen = requests.patch(
                f"https://api.github.com/repos/{repo}/pulls/{pr['number']}",
                headers=headers, json={"state": "open"}
            )
            if reopen.status_code == 200:
                url = reopen.json()["html_url"]
                print(f"[Agent] ♻️  Reopened existing PR: {url}")
                return url

    data = {
        "title": "AI Generated Docker Setup (via OpenAI)",
        "head":  head_ref,
        "base":  default_branch,
        "body":  "Auto-generated Dockerfile by AI agent using OpenAI GPT-4o.",
    }
    r        = requests.post(check_url, headers=headers, json=data)
    response = r.json()

    if r.status_code == 201:
        url = response["html_url"]
        print("[Agent] PR created:", url)
        return url
    elif r.status_code == 422:
        for err in response.get("errors", []):
            print(f"[Agent] GitHub validation error: {err}")
        all_resp = requests.get(check_url, headers=headers,
                                params={"head": head_ref})
        if all_resp.status_code == 200 and all_resp.json():
            url = all_resp.json()[0]["html_url"]
            print(f"[Agent] Existing PR: {url}")
            return url
        raise RuntimeError(f"PR failed: {response.get('message')}")
    else:
        raise RuntimeError(f"PR failed: {r.status_code}")


def get_pr_details(repo_url, token, pr_number, retries=5, retry_delay=2):
    repo    = repo_url.replace("https://github.com/", "").rstrip("/")
    headers = make_github_headers(token)
    pr_url  = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"

    last_payload = None
    for attempt in range(retries):
        r = requests.get(pr_url, headers=headers)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to fetch PR details: {r.status_code} {r.text}")

        payload = r.json()
        last_payload = payload

        # GitHub may return mergeable=null briefly while it computes mergeability.
        if payload.get("mergeable") is not None:
            return payload

        if attempt < retries - 1:
            time.sleep(retry_delay)

    return last_payload


def _extract_pr_number(pr_url):
    if not pr_url:
        return None
    try:
        return int(str(pr_url).rstrip("/").split("/")[-1])
    except (TypeError, ValueError):
        return None


def get_pr_by_number(repo_url, token, pr_number):
    if not pr_number:
        return None
    try:
        return get_pr_details(repo_url, token, pr_number, retries=1, retry_delay=0)
    except Exception:
        return None


def check_upstream_merge_conflicts(folder, repo_url, token, default_branch):
    auth_repo_url = repo_url.replace("https://", f"https://{token}@")
    fetch_result = subprocess.run(
        ["git", "fetch", "--no-tags", "--depth", "1", auth_repo_url, default_branch],
        cwd=folder, capture_output=True, text=True
    )
    if fetch_result.returncode != 0:
        return {
            "has_conflicts": False,
            "conflict_files": [],
            "conflict_text": fetch_result.stdout + fetch_result.stderr,
            "error": f"Failed to fetch upstream base branch '{default_branch}'",
        }

    merge_result = subprocess.run(
        ["git", "merge", "--no-commit", "--no-ff", "FETCH_HEAD"],
        cwd=folder, capture_output=True, text=True
    )

    diff_result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=U"],
        cwd=folder, capture_output=True, text=True
    )
    conflict_files = [f.strip() for f in diff_result.stdout.splitlines() if f.strip()]
    conflict_text = merge_result.stdout + merge_result.stderr

    abort_result = subprocess.run(
        ["git", "merge", "--abort"],
        cwd=folder, capture_output=True, text=True
    )
    if abort_result.returncode != 0:
        subprocess.run(["git", "reset", "--merge"], cwd=folder, capture_output=True, text=True)

    return {
        "has_conflicts": bool(conflict_files),
        "conflict_files": conflict_files,
        "conflict_text": conflict_text,
        "error": None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DEPLOY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def parse_deploy_targets(user_input, openai_api_key):
    client   = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"""
Extract deployment platforms from this input: "{user_input}"

SUPPORTED PLATFORMS (only these, nothing else):
- aws
- azure  
- render
- railway

RULES:
- Only return platforms explicitly mentioned
- "aws" means ONLY ["aws"], NOT azure/render/railway
- Return ONLY a JSON array, nothing else
- Examples:
  "aws" → ["aws"]
  "deploy to railway" → ["railway"]
  "render and aws" → ["render", "aws"]
  "azure" → ["azure"]

Return ONLY the JSON array:
"""}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        lines = [l for l in raw.splitlines() if not l.strip().startswith("```")]
        raw   = "\n".join(lines).strip()
    try:
        targets = json.loads(raw)
        valid   = {"aws", "azure", "render", "railway"}
        targets = [t for t in targets if t in valid]
        print(f"[Agent] Targets: {targets}")
        return targets
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# PLATFORM DEPLOYERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_free_tier_instance(ec2_client):
    try:
        resp = ec2_client.describe_instance_types(
            Filters=[{"Name": "free-tier-eligible", "Values": ["true"]}]
        )
        types = [i["InstanceType"] for i in resp.get("InstanceTypes", [])]
        print(f"[AWS] ℹ️  Free tier eligible types in this region: {types}")

        for preferred in ["t2.micro", "t3.micro", "t4g.micro", "t2.small"]:
            if preferred in types:
                print(f"[AWS] ✅ Using free tier instance: {preferred}")
                return preferred

        if types:
            print(f"[AWS] ✅ Using free tier instance: {types[0]}")
            return types[0]

    except Exception as e:
        print(f"[AWS] ⚠️  Could not detect free tier instance: {e}")

    print(f"[AWS] ⚠️  Falling back to t3.micro")
    return "t3.micro"

def deploy_to_aws(folder, creds):
    import boto3
    import base64

    app_name   = creds["app_name"]
    region     = creds["region"]
    access_key = creds["access_key"]
    secret_key = creds["secret_key"]
    env_vars   = creds.get("env_vars", {})
    port       = detect_port_from_dockerfile(folder, fallback="8080")

    ec2 = boto3.client(
        "ec2",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )
    ecr = boto3.client(
        "ecr",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )

    print(f"[AWS] 🔧 Setting up ECR repository: {app_name}...")
    try:
        repo_resp = ecr.create_repository(repositoryName=app_name)
        repo_uri  = repo_resp["repository"]["repositoryUri"]
        print(f"[AWS] ✅ ECR repo created: {repo_uri}")
    except ecr.exceptions.RepositoryAlreadyExistsException:
        repo_resp = ecr.describe_repositories(repositoryNames=[app_name])
        repo_uri  = repo_resp["repositories"][0]["repositoryUri"]
        print(f"[AWS] ℹ️  ECR repo exists: {repo_uri}")

    print(f"[AWS] 🔐 Logging Docker into ECR...")
    token_resp   = ecr.get_authorization_token()
    auth_data    = token_resp["authorizationData"][0]
    auth_token   = base64.b64decode(auth_data["authorizationToken"]).decode()
    ecr_user, ecr_pass = auth_token.split(":", 1)
    registry_url = auth_data["proxyEndpoint"]

    subprocess.run(
        ["docker", "login", "--username", ecr_user, "--password-stdin", registry_url],
        input=ecr_pass.encode(), capture_output=True, check=True
    )
    print(f"[AWS] ✅ Docker logged into ECR")

    image_tag = f"{repo_uri}:latest"
    local_tag = f"{app_name}:latest"

    print(f"[AWS] 🔨 Building image...")
    subprocess.run(["docker", "build", "-t", local_tag, "."],
                   cwd=folder, check=True)
    subprocess.run(["docker", "tag", local_tag, image_tag], check=True)

    print(f"[AWS] 📤 Pushing to ECR...")
    subprocess.run(["docker", "push", image_tag], check=True)
    print(f"[AWS] ✅ Image pushed: {image_tag}")

    print(f"[AWS] 🔧 Setting up security group...")
    sg_name = f"{app_name}-sg"
    try:
        sg_resp = ec2.create_security_group(
            GroupName=sg_name,
            Description=f"Security group for {app_name}",
        )
        sg_id = sg_resp["GroupId"]
        ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[
                {"IpProtocol": "tcp", "FromPort": int(port), "ToPort": int(port),
                 "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
                {"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
                 "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
            ]
        )
        print(f"[AWS] ✅ Security group created: {sg_id}")
    except ec2.exceptions.ClientError as e:
        if "InvalidGroup.Duplicate" in str(e):
            sgs = ec2.describe_security_groups(GroupNames=[sg_name])
            sg_id = sgs["SecurityGroups"][0]["GroupId"]
            print(f"[AWS] ℹ️  Security group exists: {sg_id}")
        else:
            raise

    env_exports = "\n".join(
        f'export {k}="{v}"' for k, v in env_vars.items()
    )
    account_id = boto3.client(
        "sts",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    ).get_caller_identity()["Account"]

    user_data = f"""#!/bin/bash
yum update -y
yum install -y docker
service docker start
usermod -aG docker ec2-user

export AWS_ACCESS_KEY_ID={access_key}
export AWS_SECRET_ACCESS_KEY={secret_key}
export AWS_DEFAULT_REGION={region}

aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com

{env_exports}
export PORT={port}

docker pull {image_tag}
docker run -d --restart always \\
  -p {port}:{port} \\
  -e PORT={port} \\
  {" ".join(f'-e {k}="{v}"' for k, v in env_vars.items())} \\
  --name {app_name} \\
  {image_tag}
"""

    ami_resp = ec2.describe_images(
        Filters=[
            {"Name": "name",        "Values": ["amzn2-ami-hvm-*-x86_64-gp2"]},
            {"Name": "state",       "Values": ["available"]},
            {"Name": "owner-alias", "Values": ["amazon"]},
        ],
        Owners=["amazon"],
    )
    ami_id = sorted(
        ami_resp["Images"], key=lambda x: x["CreationDate"], reverse=True
    )[0]["ImageId"]
    print(f"[AWS] ℹ️  Using AMI: {ami_id}")

    free_tier_instance = _get_free_tier_instance(ec2)
    print(f"[AWS] 🚀 Launching {free_tier_instance} EC2 instance (free tier)...")

    instance_resp = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=free_tier_instance,
        MinCount=1,
        MaxCount=1,
        SecurityGroupIds=[sg_id],
        UserData=user_data,
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [{"Key": "Name", "Value": app_name}],
        }],
    )

    instance_id = instance_resp["Instances"][0]["InstanceId"]
    print(f"[AWS] ✅ Instance launched: {instance_id}")
    print(f"[AWS] ⏳ Waiting for instance to get public IP (30s)...")
    time.sleep(30)

    desc      = ec2.describe_instances(InstanceIds=[instance_id])
    public_ip = desc["Reservations"][0]["Instances"][0].get("PublicIpAddress", "")

    if public_ip:
        url = f"http://{public_ip}:{port}"
        print(f"[AWS] ✅ Instance running at: {url}")
        print(f"[AWS] ⏳ App may take 5-7 minutes to start (install Docker + pull image + run)")
        print(f"[AWS] ℹ️  Instance ID: {instance_id} — stop it from AWS console to avoid charges")
        return url
    else:
        url = f"http://check-aws-console-for-ip:{port}"
        print(f"[AWS] ⚠️  Could not get public IP — check AWS console for instance: {instance_id}")
        return url


def deploy_to_azure(folder, creds):
    from azure.identity import ClientSecretCredential
    from azure.mgmt.containerregistry import ContainerRegistryManagementClient
    from azure.mgmt.appcontainers import ContainerAppsAPIClient

    app_name  = creds["app_name"]
    reg_name  = f"{app_name}registry".replace("-", "")[:50]

    cred = ClientSecretCredential(
        tenant_id=creds["tenant_id"],
        client_id=creds["client_id"],
        client_secret=creds["client_secret"],
    )

    acr      = ContainerRegistryManagementClient(cred, creds["subscription_id"])
    result   = acr.registries.begin_create(
        creds["resource_group"], reg_name,
        {"location": "eastus", "sku": {"name": "Basic"}, "admin_user_enabled": True}
    ).result()
    login_server = result.login_server
    acr_creds    = acr.registries.list_credentials(creds["resource_group"], reg_name)
    acr_user     = acr_creds.username
    acr_pass     = acr_creds.passwords[0].value
    image_tag    = f"{login_server}/{app_name}:latest"

    env_vars = creds.get("env_vars", {})
    env_list = [{"name": k, "value": v} for k, v in env_vars.items()]

    aca = ContainerAppsAPIClient(cred, creds["subscription_id"])
    res = aca.container_apps.begin_create_or_update(
        creds["resource_group"], app_name,
        {
            "location": "eastus",
            "properties": {
                "configuration": {
                    "ingress": {"external": True, "targetPort": 8080},
                    "registries": [{"server": login_server, "username": acr_user, "passwordSecretRef": "acr-pass"}],
                    "secrets": [{"name": "acr-pass", "value": acr_pass}],
                },
                "template": {
                    "containers": [{"name": app_name, "image": image_tag,
                                    "resources": {"cpu": 0.5, "memory": "1Gi"}, "env": env_list}],
                    "scale": {"minReplicas": 1, "maxReplicas": 3},
                },
            },
        }
    ).result()

    url = f"https://{res.properties.configuration.ingress.fqdn}"
    print(f"[Azure] ✅ {url}")
    return url


def deploy_to_render(fork_url, creds, folder=""):
    app_name = creds["app_name"]
    headers  = {"Authorization": f"Bearer {creds['api_key']}", "Content-Type": "application/json"}

    owner_id = requests.get(
        "https://api.render.com/v1/owners?limit=1", headers=headers
    ).json()[0]["owner"]["id"]

    services = requests.get(
        "https://api.render.com/v1/services?limit=50", headers=headers
    ).json()
    existing = next((s["service"] for s in services
                     if s["service"]["name"] == app_name), None)

    if existing:
        svc_id = existing["id"]
        print(f"[Render] ℹ️  Service exists — updating env vars and redeploying...")
        env_vars = creds.get("env_vars", {})
        if env_vars:
            env_pairs = [{"key": k, "value": v} for k, v in env_vars.items()]
            requests.put(f"https://api.render.com/v1/services/{svc_id}/env-vars",
                         headers=headers, json=env_pairs)
            print(f"[Render] ✅ Updated {len(env_pairs)} env vars")
        requests.post(f"https://api.render.com/v1/services/{svc_id}/deploys",
                      headers=headers, json={"clearCache": "do_not_clear"})
        raw_url = existing['serviceDetails']['url']
        url = raw_url if raw_url.startswith("http") else f"https://{raw_url}"
        print(f"[Render] ✅ Redeployed: {url}")
        return url

    env_vars_list = [{"key": "PORT", "value": "10000"}]
    for k, v in creds.get("env_vars", {}).items():
        env_vars_list.append({"key": k, "value": v})
    if env_vars_list:
        print(f"[Render] 📦 Using {len(env_vars_list)-1} env vars")

    resp = requests.post("https://api.render.com/v1/services", headers=headers, json={
        "type":    "web_service",
        "name":    app_name,
        "ownerId": owner_id,
        "repo":    fork_url.replace(".git", ""),
        "branch":  "ai-docker-setup",
        "serviceDetails": {
            "env":    "docker",
            "plan":   "free",
            "region": "oregon",
            "envVars": env_vars_list,
            "pullRequestPreviewsEnabled": "no",
        },
    })
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Render failed: {resp.text}")

    svc     = resp.json()["service"]
    raw_url = svc['serviceDetails']['url']
    url     = raw_url if raw_url.startswith("http") else f"https://{raw_url}"
    print(f"[Render] ✅ Service created from GitHub branch ai-docker-setup")
    print(f"[Render] ✅ {url}")
    return url


# def deploy_to_railway(folder, creds):
#     app_name       = creds["app_name"]
#     dockerhub_user = creds["dockerhub_user"]
#     dockerhub_pass = creds["dockerhub_pass"]
#     token          = creds["token"]

#     # subprocess.run(["docker", "login", "--username", dockerhub_user, "--password-stdin"],
#     #                input=dockerhub_pass.encode(), check=True)
#     # print("[Railway] ✅ Docker Hub login")
#     if dockerhub_user and dockerhub_pass:
#         login = subprocess.run(
#             ["docker", "login", "--username", dockerhub_user, "--password-stdin"],
#             input=dockerhub_pass,
#             text=True,
#             capture_output=True
#         )

#         if login.returncode != 0:
#             print("[Railway] ❌ Docker login failed")
#             print(login.stderr)
#             raise Exception("Docker login failed")
#         else:
#             print("[Railway] ✅ Docker Hub login successful")
#     else:
#         print("[Railway] ⚠️ Skipping Docker login (no credentials provided)")

#     image_name = f"{dockerhub_user}/{app_name}:latest"
#     subprocess.run(["docker", "build", "-t", image_name, "."], cwd=folder, check=True)
#     subprocess.run(["docker", "push", image_name], check=True)
#     print(f"[Railway] ✅ Pushed: {image_name}")

#     headers     = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
#     graphql_url = "https://backboard.railway.app/graphql/v2"

#     def gql(query):
#         return requests.post(graphql_url, headers=headers, json={"query": query}).json()

#     ws = gql("query { me { workspaces { id name } } }")
#     if "errors" in ws:
#         raise RuntimeError(f"Workspace error: {ws['errors']}")
#     workspace_id = ws["data"]["me"]["workspaces"][0]["id"]

#     proj = gql("""mutation { projectCreate(input:{name:"%s",workspaceId:"%s"}){
#         id environments{edges{node{id name}}}}}""" % (app_name, workspace_id))
#     if "errors" in proj:
#         raise RuntimeError(f"Project error: {proj['errors']}")
#     project_id     = proj["data"]["projectCreate"]["id"]
#     environment_id = proj["data"]["projectCreate"]["environments"]["edges"][0]["node"]["id"]

#     svc = gql("""mutation { serviceCreate(input:{projectId:"%s",name:"%s",
#         source:{image:"%s"}}){id name}}""" % (project_id, app_name, image_name))
#     if "errors" in svc:
#         raise RuntimeError(f"Service error: {svc['errors']}")
#     service_id = svc["data"]["serviceCreate"]["id"]

#     railway_port = detect_port_from_dockerfile(folder, fallback="8000")
#     gql("""mutation { variableUpsert(input:{projectId:"%s",environmentId:"%s",
#         serviceId:"%s",name:"PORT",value:"%s"})}""" % (
#         project_id, environment_id, service_id, railway_port))
#     print(f"[Railway] ✅ PORT={railway_port} set")

#     env_vars = creds.get("env_vars", {})
#     if env_vars:
#         print("[Railway] 📦 Pushing env vars to Railway...")
#         for key, value in env_vars.items():
#             resp = gql("""mutation { variableUpsert(input:{projectId:"%s",environmentId:"%s",
#                 serviceId:"%s",name:"%s",value:"%s"})}""" % (
#                 project_id, environment_id, service_id, key, value))
#             if "errors" in resp:
#                 print(f"[Railway] ⚠️  Failed to set {key}: {resp['errors']}")
#             else:
#                 print(f"[Railway] ✅ Set: {key}")
#         print("[Railway] ✅ All env vars pushed to Railway")
#     else:
#         print("[Railway] ℹ️  No env vars provided — skipping")

#     time.sleep(20)
#     domain_q    = """mutation { serviceDomainCreate(input:{serviceId:"%s",environmentId:"%s"}){domain}}""" % (service_id, environment_id)
#     domain_resp = gql(domain_q)
#     if "errors" in domain_resp or not domain_resp.get("data", {}).get("serviceDomainCreate"):
#         print("[Railway] Retrying domain creation in 15 seconds...")
#         time.sleep(15)
#         domain_resp = gql(domain_q)

#     try:
#         url = f"https://{domain_resp['data']['serviceDomainCreate']['domain']}"
#     except Exception:
#         url = f"https://railway.app/project/{project_id}"
#         print("[Railway] ⚠️  Get domain manually from Railway dashboard")

#     print(f"[Railway] ✅ {url}")
#     return url


def deploy_to_railway(folder, creds):
    import subprocess, requests, time

    app_name       = creds.get("app_name")
    dockerhub_user = creds.get("dockerhub_user")
    dockerhub_pass = creds.get("dockerhub_pass")
    token          = creds.get("token")

    if not app_name:
        raise Exception("App name is required")

    if not token:
        raise Exception("Railway token is required")

    # ─────────────────────────────────────────────
    # 1. Docker Login (Optional)
    # ─────────────────────────────────────────────
    if dockerhub_user and dockerhub_pass:
        print("[Railway] 🔐 Logging into Docker Hub...")

        login = subprocess.run(
            ["docker", "login", "--username", dockerhub_user, "--password-stdin"],
            input=dockerhub_pass,
            text=True,
            capture_output=True
        )

        if login.returncode != 0:
            print("[Railway] ❌ Docker login failed")
            print(login.stderr)
            raise Exception("Docker login failed")
        else:
            print("[Railway] ✅ Docker Hub login successful")
    else:
        print("[Railway] ⚠️ Skipping Docker login (no credentials provided)")

    # ─────────────────────────────────────────────
    # 2. Build Image
    # ─────────────────────────────────────────────
    image_name = f"{dockerhub_user}/{app_name}:latest" if dockerhub_user else f"{app_name}:latest"

    print(f"[Railway] 🏗️ Building Docker image: {image_name}")

    build = subprocess.run(
        ["docker", "build", "-t", image_name, "."],
        cwd=folder,
        capture_output=True,
        text=True
    )

    if build.returncode != 0:
        print("[Railway] ❌ Docker build failed")
        print(build.stderr)
        raise Exception("Docker build failed")

    print("[Railway] ✅ Docker build successful")

    # ─────────────────────────────────────────────
    # 3. Push Image (ONLY if logged in)
    # ─────────────────────────────────────────────
    if dockerhub_user and dockerhub_pass:
        print("[Railway] 🚀 Pushing image to Docker Hub...")

        push = subprocess.run(
            ["docker", "push", image_name],
            capture_output=True,
            text=True
        )

        if push.returncode != 0:
            print("[Railway] ❌ Docker push failed")
            print(push.stderr)
            raise Exception("Docker push failed")

        print(f"[Railway] ✅ Pushed: {image_name}")
    else:
        raise Exception("Docker Hub credentials required for Railway deploy (image must be public or pushed)")

    # ─────────────────────────────────────────────
    # 4. Railway GraphQL Setup
    # ─────────────────────────────────────────────
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    graphql_url = "https://backboard.railway.app/graphql/v2"

    def gql(query):
        resp = requests.post(graphql_url, headers=headers, json={"query": query})
        data = resp.json()
        if "errors" in data:
            raise RuntimeError(data["errors"])
        return data

    print("[Railway] 🔗 Connecting to Railway...")

    ws = gql("query { me { workspaces { id name } } }")
    workspace_id = ws["data"]["me"]["workspaces"][0]["id"]

    # ─────────────────────────────────────────────
    # 5. Create Project
    # ─────────────────────────────────────────────
    print("[Railway] 📦 Creating project...")

    proj = gql(f"""
    mutation {{
        projectCreate(input:{{
            name:"{app_name}",
            workspaceId:"{workspace_id}"
        }}) {{
            id
            environments {{ edges {{ node {{ id name }} }} }}
        }}
    }}
    """)

    project_id     = proj["data"]["projectCreate"]["id"]
    environment_id = proj["data"]["projectCreate"]["environments"]["edges"][0]["node"]["id"]

    # ─────────────────────────────────────────────
    # 6. Create Service (Docker Image)
    # ─────────────────────────────────────────────
    print("[Railway] 🚀 Creating service...")

    svc = gql(f"""
    mutation {{
        serviceCreate(input:{{
            projectId:"{project_id}",
            name:"{app_name}",
            source:{{ image:"{image_name}" }}
        }}) {{
            id
            name
        }}
    }}
    """)

    service_id = svc["data"]["serviceCreate"]["id"]

    # ─────────────────────────────────────────────
    # 7. Set PORT
    # ─────────────────────────────────────────────
    railway_port = detect_port_from_dockerfile(folder, fallback="8000")

    gql(f"""
    mutation {{
        variableUpsert(input:{{
            projectId:"{project_id}",
            environmentId:"{environment_id}",
            serviceId:"{service_id}",
            name:"PORT",
            value:"{railway_port}"
        }})
    }}
    """)

    print(f"[Railway] ✅ PORT={railway_port} set")

    # ─────────────────────────────────────────────
    # 8. Env Vars
    # ─────────────────────────────────────────────
    env_vars = creds.get("env_vars", {})

    if env_vars:
        print("[Railway] 📦 Setting environment variables...")

        for key, value in env_vars.items():
            try:
                gql(f"""
                mutation {{
                    variableUpsert(input:{{
                        projectId:"{project_id}",
                        environmentId:"{environment_id}",
                        serviceId:"{service_id}",
                        name:"{key}",
                        value:"{value}"
                    }})
                }}
                """)
                print(f"[Railway] ✅ {key}")
            except Exception as e:
                print(f"[Railway] ⚠️ Failed {key}: {e}")
    else:
        print("[Railway] ℹ️ No env vars provided")


    # ─────────────────────────────────────────────
    # 9. Trigger Deployment
    # ─────────────────────────────────────────────
    print("[Railway] 🚀 Triggering deployment...")
    
    gql(f"""
    mutation {{
        serviceInstanceDeploy(
            serviceId: "{service_id}",
            environmentId: "{environment_id}"
        )
    }}
    """)

    # ─────────────────────────────────────────────
    # 10. Create Domain & Return URL
    # ─────────────────────────────────────────────
    print("[Railway] 🌐 Creating domain...")
    time.sleep(20)  # Wait for service to be ready

    domain_q = f"""
    mutation {{
        serviceDomainCreate(input:{{
            serviceId: "{service_id}",
            environmentId: "{environment_id}"
        }}) {{
            domain
        }}
    }}
    """
    domain_resp = gql(domain_q)

    try:
        domain = domain_resp["data"]["serviceDomainCreate"]["domain"]
        url = f"https://{domain}"
    except Exception:
        print("[Railway] ⚠️  Could not get domain — retrying in 15s...")
        time.sleep(15)
        try:
            domain_resp = gql(domain_q)
            domain = domain_resp["data"]["serviceDomainCreate"]["domain"]
            url = f"https://{domain}"
        except Exception:
            url = f"https://railway.app/project/{project_id}"
            print("[Railway] ⚠️  Using project URL — check dashboard for actual domain")

    print(f"[Railway] ✅ Deployed: {url}")
    print(f"[Railway] ⏳ App may take 1-2 minutes to fully start")
    return url



def deploy_to_platforms(targets, folder, fork_url, creds):
    results = {}
    for platform in targets:
        print(f"\n{'='*50}\n[Agent] Deploying: {platform.upper()}\n{'='*50}")
        try:
            if platform == "aws":
                results["aws"]     = deploy_to_aws(folder, creds["aws"])
            elif platform == "azure":
                results["azure"]   = deploy_to_azure(folder, creds["azure"])
            elif platform == "render":
                results["render"]  = deploy_to_render(fork_url, creds["render"], folder=folder)
            elif platform == "railway":
                results["railway"] = deploy_to_railway(folder, creds["railway"])
        except Exception as e:
            print(f"[Agent] ❌ {platform}: {e}")
            results[platform] = f"FAILED: {e}"

    print(f"\n{'='*50}\n[Agent] 🚀 SUMMARY\n{'='*50}")
    for p, url in results.items():
        icon = "✅" if not str(url).startswith("FAILED") else "❌"
        print(f"  {icon} {p.upper():<10} -> {url}")
    print(f"{'='*50}\n")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CHECK MODE
# ══════════════════════════════════════════════════════════════════════════════

def check_mode(folder):
    print(f"\n[Agent] ── Check Mode ────────────────────────────────────")
    print(f"[Agent] Scanning: {os.path.abspath(folder)}\n")

    context = deep_scan_repo(folder)

    print(f"\n[Agent] ── Detection Results ─────────────────────────────")
    print(f"  Language:    {context['detected_language']}")
    print(f"  Framework:   {context['detected_framework']}")
    print(f"  ML type:     {context['ml_type']}")
    print(f"  ML libs:     {context['ml_frameworks']}")
    print(f"  Entry pts:   {context['entry_points_found']}")
    print(f"  GPU:         {context['uses_gpu']}")
    print(f"  Python ver:  {context['python_version']}")

    env_path = os.path.join(folder, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            keys = [l.split("=")[0].strip() for l in f
                    if l.strip() and not l.startswith("#") and "=" in l]
        print(f"  .env keys:   {keys} ✅")
    else:
        print(f"  .env:        ⚠️  NOT FOUND — add if app needs API keys")

    all_files = [f for f in os.listdir(folder) if f != ".git"]
    print(f"  Root files:  {all_files}")

    print(f"\n[Agent] ── Dockerfile preview ────────────────────────────")
    ml_type   = context.get("ml_type", "unknown")
    framework = context.get("detected_framework", "unknown")
    entries   = context.get("entry_points_found", [])
    entry     = entries[0] if entries else "unknown"
    entry_mod = entry.replace("\\", "/").replace("/", ".").replace(".py", "")
    py_ver    = context.get("python_version", "3.11")

    if ml_type == "streamlit":
        print(f"  Base:  python:{py_ver}-slim")
        print(f"  CMD:   streamlit run {entry} --server.port=${{PORT:-8501}} --server.address=0.0.0.0")
    elif ml_type == "gradio":
        print(f"  Base:  python:{py_ver}-slim")
        print(f"  CMD:   python {entry}")
    elif ml_type == "fastapi_ml" or framework == "fastapi":
        print(f"  Base:  python:{py_ver}-slim")
        print(f"  CMD:   uvicorn {entry_mod}:app --host 0.0.0.0 --port ${{PORT:-8000}}")
    elif ml_type == "flask_ml" or framework == "flask":
        print(f"  Base:  python:{py_ver}-slim")
        print(f"  CMD:   flask run --host=0.0.0.0 --port=${{PORT:-5000}}")
    elif framework == "django":
        print(f"  Base:  python:{py_ver}-slim")
        print(f"  CMD:   python manage.py runserver 0.0.0.0:${{PORT:-8000}}")
    elif framework in ("react", "vue", "angular", "svelte", "vite"):
        print(f"  Base:  node:18-alpine + nginx:alpine (multi-stage)")
        print(f"  CMD:   nginx -g 'daemon off;'")
    elif framework == "nextjs":
        print(f"  Base:  node:18-alpine (multi-stage)")
        print(f"  CMD:   npm start -- --port ${{PORT:-3000}}")
    elif context.get("detected_language") == "go":
        print(f"  Base:  golang:1.21-alpine + alpine:3.18 (multi-stage)")
        print(f"  CMD:   ./main")
    else:
        print(f"  Type:  {framework or ml_type} — GPT-4o will generate Dockerfile")

    print(f"\n[Agent] ── What to do ────────────────────────────────────")
    print(f"  If detection looks wrong:")
    print(f"    • Edit files in the cloned folder")
    print(f"    • Run script --check again to verify")
    print(f"  If detection looks correct:")
    print(f"    • Run script --resume to continue\n")


# ══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH NODES
# ══════════════════════════════════════════════════════════════════════════════

def node_authenticate(state: AgentState) -> AgentState:
    print("\n[Agent] ── Node: Authenticate ───────────────────────────")
    if not state.get("token", "").strip():
        return {**state,
                "error": "GitHub token missing. Set GITHUB_TOKEN in .env",
                "current_step": "authenticate"}
    try:
        fork_owner = get_authenticated_user(state["token"])
        return {**state, "fork_owner": fork_owner, "current_step": "authenticate", "error": None}
    except ConfigError as e:
        return {**state, "error": f"Config error: {e}", "current_step": "authenticate"}
    except NetworkError as e:
        return {**state, "error": f"Network error: {e}\nCheck internet connection.", "current_step": "authenticate"}
    except Exception as e:
        return {**state, "error": f"Unexpected error: {e}", "current_step": "authenticate"}


def node_get_default_branch(state: AgentState) -> AgentState:
    print("\n[Agent] ── Node: Get Default Branch ─────────────────────")
    if not state.get("repo_url", "").strip():
        return {**state, "error": "repo_url is empty", "current_step": "get_branch"}
    try:
        branch = get_default_branch(state["repo_url"], state["token"])
        return {**state, "default_branch": branch, "current_step": "get_branch", "error": None}
    except ConfigError as e:
        return {**state, "error": str(e), "current_step": "get_branch"}
    except NetworkError as e:
        return {**state, "error": f"Network error: {e}", "current_step": "get_branch"}
    except Exception as e:
        return {**state, "error": f"Unexpected error: {e}", "current_step": "get_branch"}


def node_fork_repo(state: AgentState) -> AgentState:
    print("\n[Agent] ── Node: Fork Repo ───────────────────────────────")
    try:
        fork_url = fork_repo(state["repo_url"], state["token"])
        return {**state, "fork_url": fork_url, "current_step": "fork_repo", "error": None}
    except ConfigError as e:
        return {**state, "error": str(e), "current_step": "fork_repo"}
    except NetworkError as e:
        return {**state, "error": f"Network error while forking: {e}", "current_step": "fork_repo"}
    except Exception as e:
        return {**state, "error": f"Unexpected error: {e}", "current_step": "fork_repo"}


def node_clone_repo(state: AgentState) -> AgentState:
    print("\n[Agent] ── Node: Clone Repo ──────────────────────────────")

    # ── Check Docker before doing anything else ───────────────────
    if not _check_docker_running():
        print(f"\n[Agent] ❌ Docker is NOT running!")
        print(f"[Agent] 👉 Please open Docker Desktop and wait for it to fully start")
        print(f"[Agent] ⏳ Waiting up to 2 minutes for Docker to start...")
        if _wait_for_docker(max_wait_seconds=120):
            print(f"[Agent] ✅ Docker started — continuing")
        else:
            print(f"\n[Agent] ❌ Docker did not start in time")
            print(f"[Agent] 💡 Steps to fix:")
            print(f"[Agent]    1. Open Docker Desktop")
            print(f"[Agent]    2. Wait for the whale icon to stop animating")
            print(f"[Agent]    3. Run the script again")
            return {**state,
                    "error": "Docker is not running. Open Docker Desktop and try again.",
                    "current_step": "clone_repo"}

    try:
        folder = download_repo(state["repo_url"], state["fork_url"], state["default_branch"])
        return {**state, "folder": folder, "current_step": "clone_repo", "error": None}
    except subprocess.CalledProcessError as e:
        return {**state, "error": f"Git clone failed: {e.stderr or e}", "current_step": "clone_repo"}
    except OSError as e:
        return {**state, "error": f"File system error: {e}", "current_step": "clone_repo"}
    except Exception as e:
        return {**state, "error": f"Unexpected error: {e}", "current_step": "clone_repo"}

def node_pause_for_user(state: AgentState) -> AgentState:
    print("\n[Agent] ── Node: Pause For User ──────────────────────────")
    folder      = state["folder"]
    folder_path = os.path.abspath(folder)

    save_state({
        "folder":         state["folder"],
        "fork_url":       state["fork_url"],
        "token":          state["token"],
        "fork_owner":     state["fork_owner"],
        "default_branch": state["default_branch"],
        "repo_url":       state["repo_url"],
        "openai_api_key": state["openai_api_key"],
        "paused":         True,
    })

    print(f"\n{'='*55}")
    print(f"[Agent] ⏸️  PAUSED — Repo cloned and ready for your changes!")
    print(f"{'='*55}")
    print(f"[Agent] 📁 Location: {folder_path}")
    print(f"{'='*55}\n")

    try:
        subprocess.Popen(["code", folder_path])
        print(f"[Agent] ✅ VS Code opened at: {folder_path}")
    except FileNotFoundError:
        print(f"[Agent] ⚠️  VS Code not found — open manually: code {folder_path}")

    print()

    env_file = os.path.join(folder, ".env")
    env_vars = {}

    if os.path.exists(env_file):
        print(f"[Agent] 📦 Found existing .env file — loading...")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                env_vars[k.strip()] = v.strip().strip('"').strip("'")
        print(f"[Agent] ✅ Loaded {len(env_vars)} vars: {list(env_vars.keys())}")

        add_more = input("[Agent] Want to add/update any env vars? (y/n): ").strip().lower()
        if add_more in ("y", "yes"):
            env_vars = _collect_env_vars(env_vars, folder)
    else:
        needs_env = _detect_env_var_needs(folder)

        if needs_env:
            print(f"\n[Agent] 🔍 Detected your app likely needs these env vars:")
            for var in needs_env:
                print(f"         • {var}")
            print()

        answer = input("[Agent] Does your app need environment variables / API keys? (y/n): ").strip().lower()
        if answer in ("y", "yes"):
            env_vars = _collect_env_vars({}, folder)
        else:
            print(f"[Agent] ℹ️  No env vars — continuing without .env")

    if env_vars:
        with open(env_file, "w") as f:
            for k, v in env_vars.items():
                f.write(f"{k}={v}\n")
        print(f"\n[Agent] ✅ .env written with {len(env_vars)} vars: {list(env_vars.keys())}")
        print(f"[Agent] ℹ️  .env is local only — will NOT be committed to GitHub")

        gitignore_path = os.path.join(folder, ".gitignore")
        if os.path.exists(gitignore_path):
            with open(gitignore_path, "r") as f:
                content = f.read()
            if ".env" not in content:
                with open(gitignore_path, "a") as f:
                    f.write("\n.env\n")
                print(f"[Agent] ✅ Added .env to .gitignore")
        else:
            with open(gitignore_path, "w") as f:
                f.write(".env\n")
            print(f"[Agent] ✅ Created .gitignore with .env")

    print()

    _ensure_requirements(folder, deep_scan_repo(folder), state["openai_api_key"])

    print(f"[Agent] Make any other changes you want in VS Code:")
    print(f"[Agent]   • Edit source files")
    print(f"[Agent]   • Fix requirements.txt")
    print(f"[Agent]   • Add missing data files")
    print()

    while True:
        answer = input("[Agent] Are you done making changes? (y/n): ").strip().lower()
        if answer in ("y", "yes"):
            print("[Agent] ▶️  Continuing deployment...\n")
            break
        elif answer in ("n", "no"):
            print("[Agent] ⏳ Take your time. Edit files, then type y when ready.")
        else:
            print("[Agent] Please type y or n.")

    return {**state, "current_step": "pause_for_user"}


def _collect_env_vars(existing: dict, folder: str) -> dict:
    env_vars = dict(existing)
    print(f"\n[Agent] Enter env vars one by one.")
    print(f"[Agent] Press Enter with empty key to finish.\n")
    while True:
        key = input("  KEY (or Enter to finish): ").strip()
        if not key:
            break
        if key in env_vars:
            val = input(f"  {key} [{env_vars[key]}] (Enter to keep): ").strip()
            if not val:
                val = env_vars[key]
        else:
            val = input(f"  {key}=: ").strip()
        env_vars[key] = val
        print(f"  ✅ {key} saved")
    if env_vars:
        print(f"\n[Agent] ✅ Total env vars collected: {len(env_vars)}")
    return env_vars


def _detect_env_var_needs(folder: str) -> list:
    import re

    COMMON_PATTERNS = [
        (r'openai',           ["OPENAI_API_KEY"]),
        (r'anthropic',        ["ANTHROPIC_API_KEY"]),
        (r'huggingface|hf_',  ["HUGGINGFACE_TOKEN"]),
        (r'cohere',           ["COHERE_API_KEY"]),
        (r'postgres|psycopg', ["DATABASE_URL"]),
        (r'mysql',            ["DATABASE_URL"]),
        (r'mongodb|pymongo',  ["MONGODB_URI"]),
        (r'redis',            ["REDIS_URL"]),
        (r'supabase',         ["SUPABASE_URL", "SUPABASE_KEY"]),
        (r'boto3|aws',        ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]),
        (r'stripe',           ["STRIPE_SECRET_KEY"]),
        (r'twilio',           ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"]),
        (r'sendgrid',         ["SENDGRID_API_KEY"]),
        (r'firebase',         ["FIREBASE_API_KEY"]),
        (r'pinecone',         ["PINECONE_API_KEY"]),
        (r'weaviate',         ["WEAVIATE_URL"]),
        (r'os\.getenv|os\.environ', ["(custom env vars detected)"]),
        (r'dotenv',           ["(uses dotenv — likely needs .env)"]),
    ]

    detected = []
    seen     = set()

    for walk_root, walk_dirs, walk_files in os.walk(folder):
        walk_dirs[:] = [d for d in walk_dirs if d not in
                        (".git", "__pycache__", "venv", ".venv", "node_modules")]
        for fname in walk_files:
            if not any(fname.endswith(ext) for ext in [".py", ".js", ".ts", ".env.example"]):
                continue
            fpath = os.path.join(walk_root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().lower()
            except Exception:
                continue
            for pattern, keys in COMMON_PATTERNS:
                if re.search(pattern, content):
                    for key in keys:
                        if key not in seen:
                            detected.append(key)
                            seen.add(key)

    env_example = os.path.join(folder, ".env.example")
    if os.path.exists(env_example):
        print(f"[Agent] 📄 Found .env.example — reading suggested vars...")
        with open(env_example) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k = line.split("=")[0].strip()
                if k not in seen:
                    detected.append(k)
                    seen.add(k)

    return detected

def node_create_branch_and_dockerfile(state: AgentState) -> AgentState:
    print("\n[Agent] ── Node: Create Branch & Dockerfile ─────────────")
    folder         = state["folder"]
    default_branch = state["default_branch"]
    openai_api_key = state["openai_api_key"]
    fork_url       = state["fork_url"]

    try:
        stash_name = f"agent-pre-branch-{int(time.time())}"
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=folder, capture_output=True, text=True, check=True
        )
        has_local_changes = bool(status_result.stdout.strip())

        subprocess.run(["git", "remote", "set-url", "origin", fork_url], cwd=folder, check=True)
        subprocess.run(["git", "remote", "set-url", "upstream", state["repo_url"]], cwd=folder, check=True)

        if has_local_changes:
            print("[Agent] 📦 Local changes detected — stashing them before branch sync")
            subprocess.run(
                ["git", "stash", "push", "-u", "-m", stash_name],
                cwd=folder, check=True
            )

        subprocess.run(["git", "fetch", "upstream", default_branch], cwd=folder, check=True)
        subprocess.run(["git", "checkout", "-B", default_branch, f"upstream/{default_branch}"],
                       cwd=folder, check=True)
        subprocess.run(["git", "checkout", "-B", "ai-docker-setup"], cwd=folder, check=True)

        if has_local_changes:
            stash_pop = subprocess.run(
                ["git", "stash", "pop"],
                cwd=folder, capture_output=True, text=True
            )
            print(stash_pop.stdout.strip())
            if stash_pop.returncode != 0:
                # ── Stash conflict — show user, ask LLM, get approval ──
                print(f"\n[Agent] ⚠️  STASH CONFLICT DETECTED!")
                print(f"[Agent] ℹ️  Your local changes conflict with the latest upstream {default_branch}")

                conflict_result = subprocess.run(
                    ["git", "diff", "--name-only", "--diff-filter=U"],
                    cwd=folder, capture_output=True, text=True
                )
                conflict_files = [f.strip() for f in conflict_result.stdout.splitlines() if f.strip()]

                print(f"\n[Agent] 📋 Conflicting files ({len(conflict_files)}):")
                for fname in conflict_files:
                    print(f"         • {fname}")
                print()

                # ── Read conflicting file contents ────────────────────
                file_contents = {}
                for fname in conflict_files:
                    fpath = os.path.join(folder, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                            file_contents[fname] = fh.read(3000)
                    except Exception:
                        file_contents[fname] = "(could not read file)"

                conflict_context = "\n\n".join(
                    f"--- {fname} (with conflict markers) ---\n{content}"
                    for fname, content in file_contents.items()
                )

                # ── Ask LLM to analyze ────────────────────────────────
                print(f"[Agent] 🤖 Asking GPT-4o to analyze the conflict...\n")
                client = OpenAI(api_key=openai_api_key)

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a Git expert helping resolve stash conflicts. Be concise and clear."},
                        {"role": "user", "content": f"""
A git stash pop caused conflicts when re-applying local changes on top of the latest upstream '{default_branch}' branch.

CONFLICTING FILES:
{conflict_context}

STASH OUTPUT:
{stash_pop.stdout + stash_pop.stderr}

Please:
1. Explain what caused this conflict in simple terms
2. Recommend which version to keep:
   - OURS = your local changes (what you edited during the pause step)
   - THEIRS = the latest upstream {default_branch} version
3. Explain WHY you recommend that choice

Be specific and actionable.
"""}
                    ],
                    temperature=0.1,
                )

                llm_analysis = response.choices[0].message.content.strip()

                print(f"{'='*55}")
                print(f"[Agent] 🤖 LLM CONFLICT ANALYSIS:")
                print(f"{'='*55}")
                print(llm_analysis)
                print(f"{'='*55}\n")

                # ── Ask LLM for resolution strategy ──────────────────
                response2 = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a Git expert. Return ONLY valid JSON, nothing else."},
                        {"role": "user", "content": f"""
Based on the conflict analysis, give a resolution strategy.

Conflict files: {conflict_files}
Stash output: {stash_pop.stdout + stash_pop.stderr}

Return ONLY this JSON:
{{
  "strategy": "ours" or "theirs",
  "reason": "one sentence explaining why",
  "files": [{{"file": "filename", "action": "ours" or "theirs", "reason": "why"}}]
}}
"""}
                    ],
                    temperature=0,
                )

                raw = response2.choices[0].message.content.strip()
                if raw.startswith("```"):
                    lines = [l for l in raw.splitlines() if not l.strip().startswith("```")]
                    raw   = "\n".join(lines).strip()

                try:
                    strategy = json.loads(raw)
                except Exception:
                    strategy = {
                        "strategy": "ours",
                        "reason":   "Could not parse LLM response — defaulting to our local changes",
                        "files":    []
                    }

                # ── Show strategy and ask user approval ───────────────
                print(f"\n[Agent] 🤖 LLM RECOMMENDED STRATEGY:")
                print(f"         Strategy: {strategy.get('strategy', 'ours').upper()}")
                print(f"         Reason:   {strategy.get('reason', '')}")
                if strategy.get("files"):
                    print(f"         Per file:")
                    for f in strategy["files"]:
                        print(f"           • {f.get('file')} → keep {f.get('action','ours').upper()} — {f.get('reason','')}")
                print()

                approval = input("[Agent] Do you approve this resolution? (y/n): ").strip().lower()

                if approval in ("y", "yes"):
                    print(f"[Agent] ✅ Applying LLM-suggested resolution...")
                    overall_strategy = strategy.get("strategy", "ours")
                    file_strategies  = {
                        f.get("file"): f.get("action", overall_strategy)
                        for f in strategy.get("files", [])
                    }

                    for fname in conflict_files:
                        action   = file_strategies.get(fname, overall_strategy)
                        git_side = "--ours" if action == "ours" else "--theirs"
                        subprocess.run(["git", "checkout", git_side, fname],
                                       cwd=folder, capture_output=True)
                        subprocess.run(["git", "add", fname],
                                       cwd=folder, capture_output=True)
                        print(f"[Agent] ✅ {fname} → kept {action.upper()}")

                    subprocess.run(["git", "stash", "drop"], cwd=folder, capture_output=True)
                    print(f"\n[Agent] ✅ Stash conflict resolved by LLM — continuing...")

                else:
                    # User rejected — open VS Code for manual resolution
                    print(f"[Agent] 👨‍💻 Opening VS Code for manual conflict resolution...")
                    subprocess.Popen(["code", os.path.abspath(folder)])
                    print(f"\n[Agent] ℹ️  In VS Code:")
                    print(f"           • Look for files with <<<<<<< markers")
                    print(f"           • Delete the version you don't want")
                    print(f"           • Save the files")
                    print(f"           • Run in terminal: git add . && git stash drop")
                    print()
                    input("[Agent] Press Enter when you have finished resolving conflicts manually...")
                    print(f"[Agent] ✅ Continuing with manually resolved conflicts")


        dockerfile_content, context = generate_dockerfile_with_openai(folder, openai_api_key)

        gitignore_path = os.path.join(folder, ".gitignore")
        if os.path.exists(gitignore_path):
            with open(gitignore_path, "r") as f:
                lines = f.readlines()
            filtered = [l for l in lines if l.strip().lower() not in
                        ("dockerfile", "/dockerfile", "dockerfile/")]
            if len(filtered) != len(lines):
                with open(gitignore_path, "w") as f:
                    f.writelines(filtered)

        # ── Stage ALL changes — user edits + agent files ──────────────
        subprocess.run(["git", "add", "-A"], cwd=folder, check=True)

        # ── Force add agent files in case .gitignore blocked them ──────
        subprocess.run(["git", "add", "--force", "Dockerfile"], cwd=folder, check=True)
        compose_path = os.path.join(folder, "docker-compose.yml")
        if os.path.exists(compose_path):
            subprocess.run(["git", "add", "--force", "docker-compose.yml"],
                           cwd=folder, check=True)

        # ── Show user what's going into the PR ────────────────────────
        staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=folder, capture_output=True, text=True
        )
        staged_files = [f for f in staged.stdout.strip().splitlines() if f]
        if staged_files:
            print(f"\n[Agent] 📦 Files going into PR ({len(staged_files)} files):")
            for f in staged_files:
                print(f"         • {f}")
            print()

        # ── Dynamic commit message ─────────────────────────────────────
        has_user_changes = any(
            f for f in staged_files
            if f not in ("Dockerfile", "docker-compose.yml", ".gitignore")
        )
        commit_msg = (
            "AI generated Dockerfile + user changes"
            if has_user_changes else
            "AI generated Dockerfile via OpenAI"
        )

        result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=folder, capture_output=True, text=True,
        )
        print(result.stdout.strip())
        if result.returncode != 0:
            combined = (result.stdout + result.stderr).lower()
            if "nothing to commit" in combined or "nothing added to commit" in combined:
                print("[Agent] ℹ️  Dockerfile already up to date — skipping commit")
            else:
                raise RuntimeError(f"Commit failed: {result.stderr.strip()}")
        else:
            print("[Agent] Committed Dockerfile")

        return {**state,
                "context":      context,
                "dockerfile":   dockerfile_content,
                "current_step": "create_branch_and_dockerfile",
                "error":        None}
    except Exception as e:
        return {**state, "error": str(e), "current_step": "create_branch_and_dockerfile"}

def node_test_docker(state: AgentState) -> AgentState:
    print("\n[Agent] ── Node: Test Docker Image ──────────────────────")
    folder         = state["folder"]
    context        = state["context"]
    openai_api_key = state["openai_api_key"]
    app_name       = os.path.basename(folder).lower().replace("_", "-")

    test_passed = test_docker_image(
        folder=folder,
        app_name=app_name,
        context=context,
        openai_api_key=openai_api_key,
        max_retries=3,
    )
    return {**state, "test_passed": test_passed, "current_step": "test_docker"}


def node_hitl_pr_approval(state: AgentState) -> AgentState:
    print("\n[Agent] ── Node: Create PR & Wait For Approval ──────────")
    return {**state, "pr_approved": True, "current_step": "hitl_pr_approval"}


def poll_pr_status(repo_url, token, fork_owner, pr_url=None, poll_interval=30, timeout_minutes=30):
    repo      = repo_url.replace("https://github.com/", "").rstrip("/")
    headers   = make_github_headers(token)
    check_url = f"https://api.github.com/repos/{repo}/pulls"
    deadline  = time.time() + timeout_minutes * 60
    last_reported_state = None
    pr_number = _extract_pr_number(pr_url)

    repo_owner = repo.split("/")[0]
    head_param = "ai-docker-setup" if fork_owner == repo_owner else f"{fork_owner}:ai-docker-setup"

    print(f"[Agent] 👀 Polling GitHub PR status every {poll_interval}s (timeout: {timeout_minutes}min)...")

    while time.time() < deadline:
        try:
            if pr_number:
                pr = get_pr_by_number(repo_url, token, pr_number)
                if pr:
                    if pr.get("merged_at"):
                        print(f"[Agent] ✅ PR MERGED: {pr['html_url']}")
                        return "merged"
                    if pr.get("state") == "closed":
                        print(f"[Agent] ❌ PR CLOSED/REJECTED: {pr['html_url']}")
                        return "closed"

                    mergeable = pr.get("mergeable")
                    mergeable_state = pr.get("mergeable_state", "unknown")

                    if mergeable is None:
                        current_state = "computing"
                        if last_reported_state != current_state:
                            print(f"[Agent] ⏳ GitHub is still computing mergeability for: {pr['html_url']}")
                        last_reported_state = current_state
                    elif mergeable is False or mergeable_state in {"dirty", "blocked", "behind", "unstable"}:
                        current_state = "conflict"
                        if last_reported_state != current_state:
                            print(f"\n[Agent] ⚠️  PR HAS MERGE ISSUES on GitHub!")
                            print(f"[Agent] 🔗 PR URL: {pr['html_url']}")
                            print(f"[Agent] ℹ️  mergeable={mergeable} | mergeable_state={mergeable_state}")
                            print(f"[Agent] ℹ️  GitHub cannot auto-merge yet.")
                            print(f"\n[Agent] Options:")
                            print(f"  1. Go to GitHub PR and resolve conflicts/update the branch")
                            print(f"  2. Or fix locally and force push to ai-docker-setup branch")
                            print()
                            print(f"[Agent] ⏳ Waiting for the PR to become mergeable...")
                        last_reported_state = current_state
                    else:
                        current_state = "open_clean"
                        if last_reported_state != current_state:
                            print(f"[Agent] ⏳ PR still open and mergeable: {pr['html_url']}")
                        last_reported_state = current_state

                    time.sleep(poll_interval)
                    continue

            r = requests.get(check_url, headers=headers,
                             params={"head": head_param, "state": "open"})
            open_prs = r.json() if r.status_code == 200 else []

            if open_prs:
                pr_summary = open_prs[0]
                pr = get_pr_details(repo_url, token, pr_summary["number"])
                mergeable = pr.get("mergeable")
                mergeable_state = pr.get("mergeable_state", "unknown")

                if mergeable is None:
                    current_state = "computing"
                    if last_reported_state != current_state:
                        print(f"[Agent] ⏳ GitHub is still computing mergeability for: {pr['html_url']}")
                    last_reported_state = current_state
                elif mergeable is False or mergeable_state in {"dirty", "blocked", "behind", "unstable"}:
                    current_state = "conflict"
                    if last_reported_state != current_state:
                        print(f"\n[Agent] ⚠️  PR HAS MERGE ISSUES on GitHub!")
                        print(f"[Agent] 🔗 PR URL: {pr['html_url']}")
                        print(f"[Agent] ℹ️  mergeable={mergeable} | mergeable_state={mergeable_state}")
                        print(f"[Agent] ℹ️  GitHub cannot auto-merge yet.")
                        print(f"\n[Agent] Options:")
                        print(f"  1. Go to GitHub PR and resolve conflicts/update the branch")
                        print(f"  2. Or fix locally and force push to ai-docker-setup branch")
                        print()
                        print(f"[Agent] ⏳ Waiting for the PR to become mergeable...")
                    last_reported_state = current_state
                else:
                    current_state = "open_clean"
                    if last_reported_state != current_state:
                        print(f"[Agent] ⏳ PR still open and mergeable: {pr['html_url']}")
                    last_reported_state = current_state

                time.sleep(poll_interval)
                continue


            r2 = requests.get(check_url, headers=headers,
                              params={"head": head_param, "state": "closed"})
            closed_prs = r2.json() if r2.status_code == 200 else []

            if closed_prs:
                pr = closed_prs[0]
                if pr.get("merged_at"):
                    print(f"[Agent] ✅ PR MERGED: {pr['html_url']}")
                    return "merged"
                else:
                    print(f"[Agent] ❌ PR CLOSED/REJECTED: {pr['html_url']}")
                    return "closed"

            print("[Agent] ⏳ PR not found yet — waiting...")
            time.sleep(poll_interval)

        except Exception as e:
            print(f"[Agent] ⚠️  Poll error: {e} — retrying...")
            time.sleep(poll_interval)

    return "timeout"




def _send_conflict_email(conflict_files, repo_url, folder):
    """Send email notification when merge conflicts are detected."""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    smtp_user  = os.getenv("NOTIFY_EMAIL", "")
    smtp_pass  = os.getenv("NOTIFY_EMAIL_PASSWORD", "")
    notify_to  = os.getenv("NOTIFY_EMAIL_TO", smtp_user)
    smtp_host  = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port  = int(os.getenv("SMTP_PORT", "587"))

    if not smtp_user or not smtp_pass:
        print(f"[Agent] ℹ️  Email not configured — skipping notification")
        print(f"[Agent] ℹ️  Add NOTIFY_EMAIL, NOTIFY_EMAIL_PASSWORD, NOTIFY_EMAIL_TO to .env to enable")
        return

    try:
        conflict_list = "\n".join(f"  • {f}" for f in conflict_files)
        repo_name     = repo_url.rstrip("/").split("/")[-1].replace(".git", "")

        body = f"""
Hi,

Your AI Docker Agent detected MERGE CONFLICTS in your repository.

Repository: {repo_url}
Branch:     ai-docker-setup → main

Conflicting files:
{conflict_list}

The agent has paused and is waiting for your input.
Please check your terminal — GPT-4o has analyzed the conflicts
and is waiting for your approval to resolve them automatically.

— AI Docker Agent
"""
        msg            = MIMEMultipart()
        msg["From"]    = smtp_user
        msg["To"]      = notify_to
        msg["Subject"] = f"⚠️ Merge Conflicts Detected — {repo_name}"
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, notify_to, msg.as_string())

        print(f"[Agent] 📧 Conflict notification sent to {notify_to}")

    except Exception as e:
        print(f"[Agent] ⚠️  Could not send email: {e}")
        print(f"[Agent] ℹ️  Check NOTIFY_EMAIL and NOTIFY_EMAIL_PASSWORD in .env")


def node_push_and_create_pr(state: AgentState) -> AgentState:
    print("\n[Agent] ── Node: Push Branch, Create PR & Wait ──────────")
    try:
        push_branch(state["folder"], state["fork_url"], state["token"])

        # ── Check for merge conflicts ─────────────────────────────────
        conflict_result = check_upstream_merge_conflicts(
            folder=state["folder"],
            repo_url=state["repo_url"],
            token=state["token"],
            default_branch=state["default_branch"],
        )

        if conflict_result.get("error"):
            print(f"[Agent] ⚠️  Could not run local upstream conflict check: {conflict_result['error']}")
            print(f"[Agent] ℹ️  Details: {conflict_result.get('conflict_text', '').strip()}")

        if conflict_result.get("has_conflicts"):
            conflict_files = conflict_result.get("conflict_files", [])
            conflict_text = conflict_result.get("conflict_text", "")

            print(f"\n[Agent] ⚠️  MERGE CONFLICTS DETECTED in {len(conflict_files)} file(s):")
            for cf in conflict_files:
                print(f"         • {cf}")
            print()

            # ── Send email notification ───────────────────────────────
            _send_conflict_email(
                conflict_files=conflict_files,
                repo_url=state["repo_url"],
                folder=state["folder"],
            )

            # ── Read the actual conflicting file contents ─────────────
            file_contents = {}
            for fname in conflict_files:
                fpath = os.path.join(state["folder"], fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        file_contents[fname] = f.read(3000)
                except Exception:
                    file_contents[fname] = "(could not read file)"

            # ── Ask LLM to analyze conflicts ──────────────────────────
            print(f"[Agent] 🤖 Asking GPT-4o to analyze conflicts...\n")

            conflict_context = "\n\n".join(
                f"--- {fname} ---\n{content}"
                for fname, content in file_contents.items()
            )

            client   = OpenAI(api_key=state["openai_api_key"])
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a Git expert helping resolve merge conflicts. Be concise and clear."},
                    {"role": "user",   "content": f"""
We have merge conflicts when trying to merge branch 'ai-docker-setup' into '{state["default_branch"]}' in repo: {state["repo_url"]}

CONFLICTING FILES:
{conflict_context}

CONFLICT OUTPUT:
{conflict_text[:2000]}

Please:
1. Explain what caused each conflict in simple terms
2. Recommend which version to keep (ours=ai-docker-setup or theirs=main) and WHY
3. If it needs manual merge, explain exactly what lines to keep

Be specific and actionable. Format your response clearly.
"""}
                ],
                temperature=0.1,
            )

            llm_analysis = response.choices[0].message.content.strip()

            print(f"{'='*55}")
            print(f"[Agent] 🤖 LLM CONFLICT ANALYSIS:")
            print(f"{'='*55}")
            print(llm_analysis)
            print(f"{'='*55}\n")

            # ── Ask LLM for specific resolution strategy ──────────────
            response2 = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a Git expert. Return ONLY valid JSON, nothing else."},
                    {"role": "user",   "content": f"""
Based on this conflict analysis, give me a resolution strategy.

Conflict output:
{conflict_text[:2000]}

Return ONLY this JSON:
{{
  "strategy": "ours" or "theirs" or "manual",
  "reason": "one sentence explaining why",
  "files": [{{"file": "filename", "action": "ours" or "theirs", "reason": "why"}}]
}}
"""}
                ],
                temperature=0,
            )

            raw = response2.choices[0].message.content.strip()
            if raw.startswith("```"):
                lines = [l for l in raw.splitlines() if not l.strip().startswith("```")]
                raw   = "\n".join(lines).strip()

            try:
                strategy = json.loads(raw)
            except Exception:
                strategy = {
                    "strategy": "ours",
                    "reason":   "Could not parse LLM response — defaulting to our changes",
                    "files":    []
                }

            # ── Show strategy to user and ask approval ────────────────
            print(f"\n[Agent] 🤖 LLM RECOMMENDED STRATEGY:")
            print(f"         Strategy: {strategy.get('strategy', 'ours').upper()}")
            print(f"         Reason:   {strategy.get('reason', '')}")
            if strategy.get("files"):
                print(f"         Per file:")
                for f in strategy["files"]:
                    print(f"           • {f.get('file')} → keep {f.get('action','ours').upper()} — {f.get('reason','')}")
            print()

            approval = input("[Agent] Do you approve this resolution? (y/n): ").strip().lower()

            if approval in ("y", "yes"):
                print(f"[Agent] ✅ Applying LLM-suggested resolution...")
                overall_strategy = strategy.get("strategy", "ours")
                file_strategies  = {
                    f.get("file"): f.get("action", overall_strategy)
                    for f in strategy.get("files", [])
                }

                for fname in conflict_files:
                    action   = file_strategies.get(fname, overall_strategy)
                    git_side = "--ours" if action == "ours" else "--theirs"
                    subprocess.run(["git", "checkout", git_side, fname],
                                   cwd=state["folder"], capture_output=True)
                    subprocess.run(["git", "add", fname],
                                   cwd=state["folder"], capture_output=True)
                    print(f"[Agent] ✅ {fname} → kept {action.upper()}")

                print(f"\n[Agent] ✅ All conflicts resolved by LLM suggestion!")

            else:
                # User rejected — open VS Code for manual resolution
                print(f"[Agent] 👨‍💻 Opening VS Code for manual conflict resolution...")
                subprocess.Popen(["code", os.path.abspath(state["folder"])])
                print(f"\n[Agent] ℹ️  In VS Code:")
                print(f"           • Look for files with <<<<<<< markers")
                print(f"           • Delete the version you don't want")
                print(f"           • Save the files")
                print(f"           • Run: git add .")
                print()
                input("[Agent] Press Enter when you have finished resolving conflicts manually...")
                print(f"[Agent] ✅ Continuing with manually resolved conflicts")

        # ── Check if there are actual commits between base and head ───
        result = subprocess.run(
            ["git", "log", f"upstream/{state['default_branch']}..ai-docker-setup", "--oneline"],
            cwd=state["folder"], capture_output=True, text=True
        )

        if not result.stdout.strip():
            print("[Agent] ℹ️  No commits between main and ai-docker-setup — Dockerfile already on main, skipping PR.")

            old_folder = state["folder"]
            if os.path.exists(old_folder):
                safe_rmtree(old_folder)

            auth_url     = state["repo_url"].replace("https://", f"https://{state['token']}@")
            repo_name    = state["repo_url"].rstrip("/").split("/")[-1].replace(".git", "")
            subprocess.run(
                ["git", "clone", "--branch", state["default_branch"], "--single-branch", auth_url, repo_name],
                check=True
            )
            fresh_folder = repo_name
            print(f"[Agent] ✅ Fresh clone ready: {fresh_folder}")

            test_folder = f"{repo_name}_test"
            if os.path.exists(test_folder):
                safe_rmtree(test_folder)
            shutil.copytree(fresh_folder, test_folder)
            print(f"[Agent] ✅ Test folder created: {test_folder}")

            print("\n[Agent] 🧪 Testing in fresh environment...")
            context  = deep_scan_repo(test_folder)
            local_ok = run_project_locally(
                folder=test_folder,
                context=context,
                openai_api_key=state["openai_api_key"],
            )
            safe_rmtree(test_folder)

            if local_ok:
                print("[Agent] ✅ Local test passed — ready for deployment!")
            else:
                print("[Agent] ⚠️  Local test failed — proceeding anyway")

            return {**state, "folder": fresh_folder, "pr_url": "",
                    "current_step": "push_and_create_pr", "error": None}

        pr_url = create_pull_request(
            state["repo_url"], state["token"],
            state["fork_owner"], state["default_branch"]
        )
        print(f"\n[Agent] 🔗 PR created. Waiting for approval on GitHub...")
        print(f"[Agent] 👉 Open this URL to review and merge the PR:")
        print(f"[Agent]    {pr_url or 'Check GitHub for PR URL'}")
        print(f"[Agent] ⏳ Agent will continue automatically once PR is merged.\n")

        status = poll_pr_status(
            state["repo_url"], state["token"],
            state["fork_owner"],
            pr_url=pr_url,
            poll_interval=30,
            timeout_minutes=30,
        )

        if status == "merged":
            print("[Agent] ✅ PR merged — fresh cloning from original repo...")

            old_folder = state["folder"]
            if os.path.exists(old_folder):
                safe_rmtree(old_folder)

            auth_url     = state["repo_url"].replace("https://", f"https://{state['token']}@")
            repo_name    = state["repo_url"].rstrip("/").split("/")[-1].replace(".git", "")
            subprocess.run(
                ["git", "clone", "--branch", state["default_branch"], "--single-branch", auth_url, repo_name],
                check=True
            )
            fresh_folder = repo_name
            print(f"[Agent] ✅ Fresh clone ready: {fresh_folder}")

            test_folder = f"{repo_name}_test"
            if os.path.exists(test_folder):
                safe_rmtree(test_folder)
            shutil.copytree(fresh_folder, test_folder)
            print(f"[Agent] ✅ Test folder created: {test_folder}")

            print("\n[Agent] 🧪 Testing merged code locally in fresh environment...")
            context  = deep_scan_repo(test_folder)
            local_ok = run_project_locally(
                folder=test_folder,
                context=context,
                openai_api_key=state["openai_api_key"],
            )

            safe_rmtree(test_folder)

            if local_ok:
                print("[Agent] ✅ Local test passed — ready for deployment!")
            else:
                print("[Agent] ⚠️  Local test failed — proceeding anyway")

            return {**state, "folder": fresh_folder, "pr_url": pr_url or "",
                    "current_step": "push_and_create_pr", "error": None}

        elif status == "closed":
            msg = "PR was closed/rejected on GitHub. Fix your changes and run --resume."
            print(f"[Agent] ❌ {msg}")
            save_state({
                "folder":         state["folder"],
                "fork_url":       state["fork_url"],
                "token":          state["token"],
                "fork_owner":     state["fork_owner"],
                "default_branch": state["default_branch"],
                "repo_url":       state["repo_url"],
                "openai_api_key": state["openai_api_key"],
                "paused":         True,
            })
            return {**state, "error": msg, "current_step": "push_and_create_pr"}

        else:
            msg = "PR approval timed out after 30 minutes. Merge the PR on GitHub then run --resume."
            print(f"[Agent] ⏰ {msg}")
            save_state({
                "folder":         state["folder"],
                "fork_url":       state["fork_url"],
                "token":          state["token"],
                "fork_owner":     state["fork_owner"],
                "default_branch": state["default_branch"],
                "repo_url":       state["repo_url"],
                "openai_api_key": state["openai_api_key"],
                "paused":         True,
            })
            return {**state, "error": msg, "current_step": "push_and_create_pr"}

    except Exception as e:
        return {**state, "error": str(e), "current_step": "push_and_create_pr"}
def node_hitl_deploy_approval(state: AgentState) -> AgentState:
    print("\n[Agent] ── Node: HITL Deploy Approval ───────────────────")
    print("\n[Agent] ── Deployment Decision ──────────────────────────")
    deploy_approval = input("[Agent] Do you want to deploy the application? (yes/no): ").strip().lower()
    approved        = deploy_approval in ("yes", "y")
    if not approved:
        print("[Agent] 🛑 Deployment skipped by user. All done!")
    return {**state, "deploy_approved": approved, "current_step": "hitl_deploy_approval"}


def node_collect_deploy_info(state: AgentState) -> AgentState:
    print("\n[Agent] ── Node: Collect Deploy Info ────────────────────")
    print("\n[Agent] ── Deployment ────────────────────────────────────")
    deploy_input = input("\nWhere to deploy? (e.g. 'deploy to railway'): ").strip()
    targets      = parse_deploy_targets(deploy_input, state["openai_api_key"])

    if not targets:
        print("[Agent] No valid targets. Skipping deployment.")
        return {**state, "deploy_targets": [], "current_step": "collect_deploy_info"}

    app_name = os.getenv("APP_NAME", "").strip() or input("\nApp name: ").strip()

    env_vars = {}
    folder   = state["folder"]
    env_file = os.path.join(folder, ".env")

    if os.path.exists(env_file):
        print(f"[Agent] 📦 Found local .env — loading env vars for deployment...")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                env_vars[k.strip()] = v.strip().strip('"').strip("'")
        print(f"[Agent] ✅ Loaded {len(env_vars)} vars: {list(env_vars.keys())}")
        add_more = input("[Agent] Add more env vars? (y/n): ").strip().lower()
        if add_more in ("y", "yes"):
            print("[Agent] Enter vars one by one. Press Enter with empty key to finish.")
            while True:
                key = input("  KEY: ").strip()
                if not key:
                    break
                val = input(f"  {key}=: ").strip()
                env_vars[key] = val
    else:
        print(f"[Agent] ℹ️  No local .env found.")
        needs_env = input("[Agent] Does your app need environment variables? (y/n): ").strip().lower()
        if needs_env in ("y", "yes"):
            print("[Agent] Enter your env vars one by one. Press Enter with empty key to finish.")
            while True:
                key = input("  KEY: ").strip()
                if not key:
                    break
                val = input(f"  {key}=: ").strip()
                env_vars[key] = val
            if env_vars:
                print(f"[Agent] ✅ Collected {len(env_vars)} env vars")

    return {**state,
            "deploy_targets": targets,
            "app_name":       app_name,
            "env_vars":       env_vars,
            "current_step":   "collect_deploy_info"}


def collect_credentials(targets, app_name):
    app_name = app_name.lower().replace(" ", "-")
    if len(app_name) < 4:
        app_name = f"{app_name}-app"
    print(f"[Agent] App name: {app_name}")
    creds = {}

    def get_value(env_key, label):
        val = os.getenv(env_key, "").strip()
        if val:
            print(f"  ✅ {env_key} loaded from .env")
            return val
        return input(f"  {label}: ").strip()

    if "aws" in targets:
        creds["aws"] = {
            "access_key": get_value("AWS_ACCESS_KEY_ID",     "AWS_ACCESS_KEY_ID"),
            "secret_key": get_value("AWS_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY"),
            "region":     get_value("AWS_REGION",            "AWS_REGION (e.g. ap-south-1)"),
            "app_name":   app_name,
        }

    if "azure" in targets:
        creds["azure"] = {
            "client_id":       get_value("AZURE_CLIENT_ID",       "AZURE_CLIENT_ID"),
            "client_secret":   get_value("AZURE_CLIENT_SECRET",   "AZURE_CLIENT_SECRET"),
            "tenant_id":       get_value("AZURE_TENANT_ID",       "AZURE_TENANT_ID"),
            "subscription_id": get_value("AZURE_SUBSCRIPTION_ID", "AZURE_SUBSCRIPTION_ID"),
            "resource_group":  get_value("AZURE_RESOURCE_GROUP",  "AZURE_RESOURCE_GROUP"),
            "dockerhub_user":  get_value("DOCKERHUB_USERNAME",    "Docker Hub Username"),
            "dockerhub_pass":  get_value("DOCKERHUB_PASSWORD",    "Docker Hub Password"),
            "app_name":        app_name,
            "fork_url":        "",
        }

    if "render" in targets:
        creds["render"] = {
            "api_key":  get_value("RENDER_API_KEY", "RENDER_API_KEY"),
            "app_name": app_name,
            "fork_url": "",
        }

    if "railway" in targets:
        creds["railway"] = {
            "token":          get_value("RAILWAY_TOKEN",      "RAILWAY_TOKEN"),
            "dockerhub_user": get_value("DOCKERHUB_USERNAME", "Docker Hub Username"),
            "dockerhub_pass": get_value("DOCKERHUB_PASSWORD", "Docker Hub Password"),
            "app_name":       app_name,
        }

    return creds


def node_deploy(state: AgentState) -> AgentState:
    print("\n[Agent] ── Node: Deploy ──────────────────────────────────")
    targets  = state.get("deploy_targets", [])
    app_name = state.get("app_name", "")
    folder   = state["folder"]
    fork_url = state["fork_url"]

    if not targets:
        print("[Agent] No deploy targets — skipping.")
        return {**state, "deploy_results": {}, "current_step": "deploy"}

    # ── Check for existing partial-deploy state ────────────────────
    existing      = load_deploy_state()
    prior_results = {}

    if existing:
        prior_results  = existing.get("results", {})
        already_done   = [p for p, r in prior_results.items()
                          if not str(r).startswith("FAILED") and r not in ("PENDING", "IN_PROGRESS", "")]
        still_todo     = [p for p in targets if p not in already_done]

        if already_done:
            print(f"\n[Deploy] ♻️  Resuming — already succeeded: {already_done}")
            print(f"[Deploy] 🔁 Still to deploy: {still_todo}")
            for p in already_done:
                print(f"  ✅ {p.upper():<10} -> {prior_results[p]}  (skipped)")

        targets = still_todo
        if not targets:
            print("[Deploy] ✅ All platforms already deployed!")
            clear_deploy_state()
            return {**state, "deploy_results": prior_results, "current_step": "deploy"}
    # ── Check Docker is running before deploying ──────────────────
    needs_docker = any(p in targets for p in ["aws", "azure", "railway"])
    if needs_docker and not _check_docker_running():
        print(f"\n[Deploy] ❌ Docker is NOT running!")
        print(f"[Deploy] 👉 Open Docker Desktop first — AWS, Azure and Railway all need Docker to build and push images")
        print(f"[Deploy] ⏳ Waiting up to 2 minutes for Docker to start...")
        if _wait_for_docker(max_wait_seconds=120):
            print(f"[Deploy] ✅ Docker started — continuing deployment")
        else:
            print(f"\n[Deploy] ❌ Docker did not start in time")
            print(f"[Deploy] 💡 Steps to fix:")
            print(f"[Deploy]    1. Open Docker Desktop")
            print(f"[Deploy]    2. Wait for the whale icon to stop animating")
            print(f"[Deploy]    3. Run --resume-deploy to retry")
            return {**state,
                    "error": "Docker is not running. Open Docker Desktop and run --resume-deploy",
                    "current_step": "deploy"}

    creds = collect_credentials(targets, app_name)
    for platform in creds:
        creds[platform]["fork_url"] = fork_url
        creds[platform]["folder"]   = folder
        creds[platform]["env_vars"] = state.get("env_vars", {})
    creds = collect_credentials(targets, app_name)
    for platform in creds:
        creds[platform]["fork_url"] = fork_url
        creds[platform]["folder"]   = folder
        creds[platform]["env_vars"] = state.get("env_vars", {})

    all_results = dict(prior_results)
    for p in targets:
        all_results[p] = "PENDING"

    save_deploy_state({
        "targets":  targets,
        "app_name": app_name,
        "folder":   folder,
        "fork_url": fork_url,
        "results":  all_results,
        "creds":    creds,
        "env_vars": state.get("env_vars", {}),
    })

    for platform in targets:
        print(f"\n{'='*50}\n[Deploy] Starting: {platform.upper()}\n{'='*50}")
        all_results[platform] = "IN_PROGRESS"
        save_deploy_state({
            "targets": targets, "app_name": app_name, "folder": folder,
            "fork_url": fork_url, "results": all_results,
            "creds": creds, "env_vars": state.get("env_vars", {}),
        })

        retry_count = 0
        max_retries = 2

        while retry_count <= max_retries:
            try:
                if platform == "aws":
                    url = deploy_to_aws(folder, creds["aws"])
                elif platform == "azure":
                    url = deploy_to_azure(folder, creds["azure"])
                elif platform == "render":
                    url = deploy_to_render(fork_url, creds["render"], folder=folder)
                elif platform == "railway":
                    url = deploy_to_railway(folder, creds["railway"])
                else:
                    raise ValueError(f"Unknown platform: {platform}")

                all_results[platform] = url
                print(f"[Deploy] ✅ {platform.upper()} -> {url}")
                break

            except Exception as e:
                error_msg = str(e)
                print(f"\n[Deploy] ❌ {platform.upper()} failed (attempt {retry_count+1}): {error_msg}")

                if _is_credential_error(error_msg) and retry_count < max_retries:
                    print(f"\n[Deploy] 🔑 Credential error detected")
                    _print_credential_hint(platform, error_msg)
                    new_creds = _prompt_recredential(platform, app_name, state.get("env_vars", {}))
                    if new_creds:
                        creds[platform] = new_creds
                        creds[platform]["fork_url"] = fork_url
                        creds[platform]["folder"]   = folder
                        creds[platform]["env_vars"] = state.get("env_vars", {})
                        retry_count += 1
                        print(f"[Deploy] 🔄 Retrying {platform.upper()} with new credentials...")
                        time.sleep(2)
                        continue
                    else:
                        all_results[platform] = f"FAILED: {error_msg}"
                        break

                elif _is_quota_error(error_msg):
                    print(f"\n[Deploy] 💳 Quota/free tier error on {platform.upper()}")
                    _print_quota_hint(platform)
                    choice = input(f"\n[Deploy] Retry {platform.upper()} with different credentials? (y/n): ").strip().lower()
                    if choice in ("y", "yes") and retry_count < max_retries:
                        new_creds = _prompt_recredential(platform, app_name, state.get("env_vars", {}))
                        if new_creds:
                            creds[platform] = new_creds
                            creds[platform]["fork_url"] = fork_url
                            creds[platform]["folder"]   = folder
                            creds[platform]["env_vars"] = state.get("env_vars", {})
                            retry_count += 1
                            continue
                    all_results[platform] = f"FAILED: {error_msg}"
                    break

                elif _is_network_error(error_msg) and retry_count < max_retries:
                    wait = 10 * (retry_count + 1)
                    print(f"[Deploy] 🌐 Network error — waiting {wait}s before retry...")
                    time.sleep(wait)
                    retry_count += 1
                    continue

                else:
                    all_results[platform] = f"FAILED: {error_msg}"
                    save_deploy_state({
                        "targets": targets, "app_name": app_name, "folder": folder,
                        "fork_url": fork_url, "results": all_results,
                        "creds": creds, "env_vars": state.get("env_vars", {}),
                    })
                    print(f"\n[Deploy] 💾 {platform.upper()} failed — continuing to next platform...")
                    break

        save_deploy_state({
            "targets": targets, "app_name": app_name, "folder": folder,
            "fork_url": fork_url, "results": all_results,
            "creds": creds, "env_vars": state.get("env_vars", {}),
        })

    print(f"\n{'='*50}\n[Deploy] 🚀 SUMMARY\n{'='*50}")
    all_succeeded = True
    for p, url in all_results.items():
        icon = "✅" if not str(url).startswith("FAILED") and url not in ("PENDING", "IN_PROGRESS") else "❌"
        if str(url).startswith("FAILED") or url in ("PENDING", "IN_PROGRESS"):
            all_succeeded = False
        print(f"  {icon} {p.upper():<10} -> {url}")
    print(f"{'='*50}\n")

    if all_succeeded:
        clear_deploy_state()
        print("[Deploy] ✅ All deployed — state cleared")
    else:
        failed = [p for p, r in all_results.items() if str(r).startswith("FAILED")]
        print(f"[Deploy] ⚠️  {len(failed)} platform(s) failed: {failed}")
        print(f"[Deploy] 💡 Fix and run:  python langgraph_agent.py --resume-deploy")

    return {**state, "deploy_results": all_results, "current_step": "deploy"}


def node_done(state: AgentState) -> AgentState:
    print("\n[Agent] ══════════════════════════════════════════════════")
    print("[Agent] ✅ Pipeline complete!")
    results = state.get("deploy_results", {})
    if results:
        for p, url in results.items():
            icon = "✅" if not str(url).startswith("FAILED") else "❌"
            print(f"  {icon} {p.upper():<10} -> {url}")
    print("[Agent] ══════════════════════════════════════════════════\n")
    return {**state, "current_step": "done"}


def node_error(state: AgentState) -> AgentState:
    step  = state.get("current_step", "unknown")
    error = state.get("error", "Unknown error")

    print(f"\n{'='*55}")
    print(f"[Agent] ❌ FAILED at step: {step}")
    print(f"{'='*55}")
    print(f"Error: {error}")
    print(f"{'='*55}")

    fix_hints = {
        "authenticate": [
            "Check GITHUB_TOKEN is set in .env",
            "Token needs repo and workflow scopes",
            "Go to github.com/settings/tokens to verify",
        ],
        "get_branch": [
            "Check repo URL is correct",
            "Make sure repo exists and is accessible",
            "Private repos need repo scope on token",
        ],
        "fork_repo": [
            "Check token has permission to fork",
            "Try manually forking on GitHub first",
        ],
        "clone_repo": [
            "Check internet connection",
            "Make sure git is installed: git --version",
            "Check disk space",
        ],
        "create_branch_and_dockerfile": [
            "Run --check to see what was detected",
            "Run --resume to retry from this point",
        ],
        "test_docker": [
            "Make sure Docker is running: docker ps",
            "Run --resume to retry",
        ],
        "push_and_create_pr": [
            "Check internet connection",
            "Run --resume after fixing the issue",
        ],
        "deploy": [
            "Check deploy credentials in .env",
            "Make sure Docker is running",
            "Run --resume to retry deployment",
        ],
    }

    hints = fix_hints.get(step, ["Run --resume after fixing the issue"])
    print(f"\nHow to fix:")
    for i, hint in enumerate(hints, 1):
        print(f"  {i}. {hint}")

    print(f"\nTo retry:  python langgraph_agent.py --resume")
    print(f"{'='*55}\n")
    return state

# ══════════════════════════════════════════════════════════════════════════════
# CONDITIONAL EDGE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def route_after_auth(state: AgentState) -> str:
    return "error" if state.get("error") else "get_default_branch"

def route_after_branch(state: AgentState) -> str:
    return "error" if state.get("error") else "fork_repo"

def route_after_fork(state: AgentState) -> str:
    return "error" if state.get("error") else "clone_repo"

def route_after_clone(state: AgentState) -> str:
    return "error" if state.get("error") else "pause_for_user"

def route_after_dockerfile(state: AgentState) -> str:
    return "error" if state.get("error") else "test_docker"

def route_after_test(state: AgentState) -> str:
    if state.get("test_passed"):
        return "hitl_pr_approval"
    else:
        print("\n[Agent] ❌ Docker test FAILED — skipping PR and deployment")
        print("[Agent] Fix the issues and run 'python langgraph_agent.py --resume' again")
        return "done"

def route_after_pr_approval(state: AgentState) -> str:
    return "push_and_create_pr"

def route_after_push_pr(state: AgentState) -> str:
    return "error" if state.get("error") else "hitl_deploy_approval"

def route_after_deploy_approval(state: AgentState) -> str:
    return "collect_deploy_info" if state.get("deploy_approved") else "done"

def route_after_collect_deploy(state: AgentState) -> str:
    return "deploy" if state.get("deploy_targets") else "done"


# ══════════════════════════════════════════════════════════════════════════════
# BUILD THE LANGGRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("authenticate",                 node_authenticate)
    builder.add_node("get_default_branch",           node_get_default_branch)
    builder.add_node("fork_repo",                    node_fork_repo)
    builder.add_node("clone_repo",                   node_clone_repo)
    builder.add_node("pause_for_user",               node_pause_for_user)
    builder.add_node("create_branch_and_dockerfile", node_create_branch_and_dockerfile)
    builder.add_node("test_docker",                  node_test_docker)
    builder.add_node("hitl_pr_approval",             node_hitl_pr_approval)
    builder.add_node("push_and_create_pr",           node_push_and_create_pr)
    builder.add_node("hitl_deploy_approval",         node_hitl_deploy_approval)
    builder.add_node("collect_deploy_info",          node_collect_deploy_info)
    builder.add_node("deploy",                       node_deploy)
    builder.add_node("done",                         node_done)
    builder.add_node("error",                        node_error)

    builder.set_entry_point("authenticate")

    builder.add_conditional_edges("authenticate",       route_after_auth,            {"get_default_branch": "get_default_branch", "error": "error"})
    builder.add_conditional_edges("get_default_branch", route_after_branch,          {"fork_repo": "fork_repo",                   "error": "error"})
    builder.add_conditional_edges("fork_repo",          route_after_fork,            {"clone_repo": "clone_repo",                 "error": "error"})
    builder.add_conditional_edges("clone_repo",         route_after_clone,           {"pause_for_user": "pause_for_user",         "error": "error"})
    builder.add_edge("pause_for_user", "create_branch_and_dockerfile")
    builder.add_conditional_edges("create_branch_and_dockerfile", route_after_dockerfile, {"test_docker": "test_docker", "error": "error"})
    builder.add_conditional_edges("test_docker",        route_after_test,            {"hitl_pr_approval": "hitl_pr_approval",     "done": "done"})
    builder.add_conditional_edges("hitl_pr_approval",   route_after_pr_approval,     {"push_and_create_pr": "push_and_create_pr"})
    builder.add_conditional_edges("push_and_create_pr", route_after_push_pr,         {"hitl_deploy_approval": "hitl_deploy_approval", "error": "error"})
    builder.add_conditional_edges("hitl_deploy_approval", route_after_deploy_approval, {"collect_deploy_info": "collect_deploy_info", "done": "done"})
    builder.add_conditional_edges("collect_deploy_info", route_after_collect_deploy,  {"deploy": "deploy", "done": "done"})

    builder.add_edge("deploy", "done")
    builder.add_edge("done",   END)
    builder.add_edge("error",  END)

    return builder.compile()


# ══════════════════════════════════════════════════════════════════════════════
# RESUME GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_resume_graph():
    builder = StateGraph(AgentState)

    builder.add_node("create_branch_and_dockerfile", node_create_branch_and_dockerfile)
    builder.add_node("test_docker",                  node_test_docker)
    builder.add_node("hitl_pr_approval",             node_hitl_pr_approval)
    builder.add_node("push_and_create_pr",           node_push_and_create_pr)
    builder.add_node("hitl_deploy_approval",         node_hitl_deploy_approval)
    builder.add_node("collect_deploy_info",          node_collect_deploy_info)
    builder.add_node("deploy",                       node_deploy)
    builder.add_node("done",                         node_done)
    builder.add_node("error",                        node_error)

    builder.set_entry_point("create_branch_and_dockerfile")

    builder.add_conditional_edges("create_branch_and_dockerfile", route_after_dockerfile,      {"test_docker": "test_docker",             "error": "error"})
    builder.add_conditional_edges("test_docker",                  route_after_test,            {"hitl_pr_approval": "hitl_pr_approval",   "done": "done"})
    builder.add_conditional_edges("hitl_pr_approval",             route_after_pr_approval,     {"push_and_create_pr": "push_and_create_pr"})
    builder.add_conditional_edges("push_and_create_pr",           route_after_push_pr,         {"hitl_deploy_approval": "hitl_deploy_approval", "error": "error"})
    builder.add_conditional_edges("hitl_deploy_approval",         route_after_deploy_approval, {"collect_deploy_info": "collect_deploy_info", "done": "done"})
    builder.add_conditional_edges("collect_deploy_info",          route_after_collect_deploy,  {"deploy": "deploy", "done": "done"})

    builder.add_edge("deploy", "done")
    builder.add_edge("done",   END)
    builder.add_edge("error",  END)

    return builder.compile()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    def get_env_or_input(env_key, label):
        val = os.getenv(env_key, "").strip()
        if val:
            print(f"  ✅ {env_key} loaded from .env")
            return val
        return input(f"{label}: ").strip()

    if "--resume-deploy" in sys.argv:
        saved_deploy = load_deploy_state()
        if not saved_deploy:
            print("[Agent] ❌ No saved deploy state found. Run deployment first.")
            sys.exit(1)

        print(f"\n[Agent] ▶️  Resuming deploy from saved state...")
        print(f"[Agent] Folder:  {saved_deploy['folder']}")
        print(f"[Agent] Targets: {saved_deploy['targets']}")
        failed = [p for p, r in saved_deploy.get("results", {}).items()
                if str(r).startswith("FAILED") or r in ("PENDING", "IN_PROGRESS")]
        print(f"[Agent] Failed:  {failed}\n")

        main_state = load_state()

        resume_state: AgentState = {
            "repo_url":        main_state.get("repo_url", "")        if main_state else "",
            "token":           main_state.get("token", "")           if main_state else "",
            "openai_api_key":  main_state.get("openai_api_key", get_env_or_input("OPENAI_API_KEY", "OpenAI API Key")) if main_state else get_env_or_input("OPENAI_API_KEY", "OpenAI API Key"),
            "fork_owner":      main_state.get("fork_owner", "")      if main_state else "",
            "default_branch":  main_state.get("default_branch", "")  if main_state else "",
            "fork_url":        saved_deploy.get("fork_url", main_state.get("fork_url", "") if main_state else ""),
            "folder":          saved_deploy["folder"],
            "context":         {},
            "dockerfile":      "",
            "test_passed":     True,
            "deploy_targets":  saved_deploy["targets"],
            "app_name":        saved_deploy.get("app_name", ""),
            "deploy_results":  saved_deploy.get("results", {}),
            "pr_approved":     True,
            "pr_url":          "",
            "deploy_approved": True,
            "env_vars":        saved_deploy.get("env_vars", {}),
            "paused":          False,
            "error":           None,
            "current_step":    "resume_deploy",
        }

        final_state = node_deploy(resume_state)
        node_done(final_state)
        sys.exit(0)
    if "--check" in sys.argv:
        state = load_state()
        if not state:
            print("[Agent] ❌ No paused session found. Run normally first.")
            sys.exit(1)
        check_mode(state["folder"])
        sys.exit(0)

    if "--resume" in sys.argv:
        saved = load_state()
        if not saved:
            print("[Agent] ❌ No paused session found. Run normally first.")
            sys.exit(1)

        print(f"\n[Agent] ▶️  Resuming from paused state...")
        print(f"[Agent] Folder: {saved['folder']}")
        os.remove(STATE_FILE)

        initial_state: AgentState = {
            "repo_url":        saved["repo_url"],
            "token":           saved["token"],
            "openai_api_key":  saved["openai_api_key"],
            "fork_owner":      saved["fork_owner"],
            "default_branch":  saved["default_branch"],
            "fork_url":        saved["fork_url"],
            "folder":          saved["folder"],
            "context":         {},
            "dockerfile":      "",
            "test_passed":     False,
            "deploy_targets":  [],
            "app_name":        "",
            "deploy_results":  {},
            "pr_approved":     False,
            "pr_url":          "",
            "deploy_approved": False,
            "env_vars":        {},
            "paused":          False,
            "error":           None,
            "current_step":    "resume",
        }

        graph = build_resume_graph()
        graph.invoke(initial_state)
        sys.exit(0)

    print("\n[Agent] ── Configuration ─────────────────────────────────")
    repo_url       = input("Enter GitHub repo URL: ").strip()
    token          = get_env_or_input("GITHUB_TOKEN",   "GitHub Token")
    openai_api_key = get_env_or_input("OPENAI_API_KEY", "OpenAI API Key")

    initial_state: AgentState = {
        "repo_url":        repo_url,
        "token":           token,
        "openai_api_key":  openai_api_key,
        "fork_owner":      "",
        "default_branch":  "",
        "fork_url":        "",
        "folder":          "",
        "context":         {},
        "dockerfile":      "",
        "test_passed":     False,
        "deploy_targets":  [],
        "app_name":        "",
        "deploy_results":  {},
        "pr_approved":     False,
        "pr_url":          "",
        "deploy_approved": False,
        "env_vars":        {},
        "paused":          False,
        "error":           None,
        "current_step":    "start",
    }

    graph = build_graph()
    graph.invoke(initial_state)
