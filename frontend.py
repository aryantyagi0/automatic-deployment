import streamlit as st
import os
import subprocess
import json
import shutil
import builtins
import time
import threading
import sys
import requests

import langgraph_4 as agent
import shutil as _shutil

def _check_tool(name):
    return _shutil.which(name) is not None

TOOLS = {
    "git":    _check_tool("git"),
    "docker": _check_tool("docker"),
    "code":   _check_tool("code"),
}

WORKSPACE = os.path.join(os.path.expanduser("~"), "ai-devops-workspace")
os.makedirs(WORKSPACE, exist_ok=True)
DEPLOY_STATE_FILE = "_deploy_state.json"

def _save_deploy_state_ui(data: dict):
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
    with open(DEPLOY_STATE_FILE, "w") as f:
        json.dump(safe_data, f, indent=2)

def _load_deploy_state_ui() -> dict:
    if not os.path.exists(DEPLOY_STATE_FILE):
        return None
    try:
        with open(DEPLOY_STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return None

def _clear_deploy_state_ui():
    if os.path.exists(DEPLOY_STATE_FILE):
        os.remove(DEPLOY_STATE_FILE)

def _is_credential_error_ui(msg: str) -> bool:
    kws = ["unauthorized", "authentication", "invalid token", "access denied",
           "forbidden", "401", "403", "invalid api key", "incorrect credentials",
           "invalid credentials", "not authorized", "docker login failed",
           "denied: requested access", "invalidsignature", "no such access key"]
    return any(k in msg.lower() for k in kws)

def _is_quota_error_ui(msg: str) -> bool:
    kws = ["free tier", "quota exceeded", "limit exceeded", "instancelimitexceeded",
           "billing", "payment", "upgrade", "plan limit", "rate limit",
           "insufficient capacity", "vcpu"]
    return any(k in msg.lower() for k in kws)

CREDENTIAL_HINTS = {
    "aws":     "Go to AWS Console → IAM → Users → Security credentials and generate a new Access Key + Secret. Ensure your IAM user has EC2, ECR, and STS permissions.",
    "azure":   "Go to Azure Portal → Azure Active Directory → App registrations. Verify Client ID, Tenant ID, Client Secret. Secrets expire — create a new one if needed.",
    "render":  "Go to Render Dashboard → Account Settings → API Keys and generate a new key.",
    "railway": "Go to Railway Dashboard → Account Settings → Tokens and create a new token. Also verify your Docker Hub username/password.",
}

QUOTA_HINTS = {
    "aws":     "Your AWS free tier or EC2 instance limit may be exhausted. Check EC2 → Limits in your region, or terminate unused instances.",
    "azure":   "Your Azure free tier CPU quota may be exhausted. Upgrade to Pay-As-You-Go in the Azure Portal.",
    "render":  "Render free tier allows 1 web service. Consider upgrading to the Starter plan ($7/month).",
    "railway": "Railway Hobby plan has a $5/month credit. Add a payment method at railway.app/dashboard.",
}

st.set_page_config(layout="wide")

# ── Monkeypatch builtins.input so agent functions never block ──
_orig_input = builtins.input

def _patch_input():
    builtins.input = lambda *_: "y"

def _restore_input():
    builtins.input = _orig_input

# ── Session state defaults ──
for k, v in {
    "stage": "idle",
    "saved_repo_url": "",
    "folder": None,
    "fork_url": None,
    "default_branch": None,
    "user": None,
    "context": None,
    "pr_url": None,
    "deploy_results": {},
    "logs": [],
    "stash_conflict": None,
    "merge_conflict": None,
    "test_proc": None,
    "test_folder": None,
    "test_port": None,
    "test_venv": None,
    "test_ctx": None,
    "user_dockerfile": None,
    "user_has_dockerfile_changes": False,
    "deploy_failed_info": {},
    "resume_deploy_data": {},
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ──
st.sidebar.title("Configuration")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
token      = st.sidebar.text_input("GitHub Token",   type="password")

st.sidebar.markdown("---")
st.sidebar.markdown("**System Tools**")
for _tool, _ok in TOOLS.items():
    if _ok:
        st.sidebar.success(f"✅ {_tool} found")
    else:
        st.sidebar.error(f"❌ {_tool} not in PATH — install it or add to PATH")

if st.session_state.logs:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Activity Log**")
    st.sidebar.code("\n".join(st.session_state.logs[-30:]))

_existing_deploy = _load_deploy_state_ui()
if _existing_deploy:
    _failed = [p for p, r in _existing_deploy.get("results", {}).items()
               if str(r).startswith("FAILED") or r in ("PENDING", "IN_PROGRESS")]
    _succeeded = [p for p, r in _existing_deploy.get("results", {}).items()
                  if not str(r).startswith("FAILED") and r not in ("PENDING", "IN_PROGRESS", "")]
    if _failed and st.session_state.stage not in ("deployed",):
        st.sidebar.divider()
        st.sidebar.warning(f"⚠️ Unfinished deploy detected")
        st.sidebar.caption(f"Failed: {_failed}")
        if _succeeded:
            st.sidebar.caption(f"Already done: {_succeeded}")
        if st.sidebar.button("Resume Failed Deployments"):
            st.session_state.stage = "resume_deploy"
            st.session_state.resume_deploy_data = _existing_deploy
            st.rerun()

def log(msg):
    st.session_state.logs.append(msg)

def _open_vscode(path):
    """Open VS Code if available, otherwise show path to user."""
    if TOOLS["code"]:
        try:
            subprocess.Popen(["code", os.path.abspath(path)])
        except Exception as e:
            st.info(f"Could not open VS Code: {e}\nOpen manually: `{os.path.abspath(path)}`")
    else:
        st.info(f"VS Code not found. Open this folder manually:\n`{os.path.abspath(path)}`")

def _git_clone(auth_url, branch, clone_dest):
    """Clone a repo with clear error messages if git is missing."""
    if not TOOLS["git"]:
        st.error("git not found on this machine. Install Git and ensure it is in your PATH.")
        st.stop()
    try:
        subprocess.run(
            ["git", "clone", "--branch", branch, "--single-branch", auth_url, clone_dest],
            check=True
        )
    except subprocess.CalledProcessError as e:
        st.error(f"git clone failed: {e}")
        st.stop()

# ── Stage progress indicator ──
STAGE_LABELS = {
    "idle":             "Step 1 / 8 — Clone Repo",
    "cloned":           "Step 2 / 8 — Edit Files & Setup",
    "docker":           "Step 3 / 8 — Generate Dockerfile",
    "stash_conflict":   "Step 3 / 8 — Resolve Stash Conflict",
    "stash_manual":     "Step 3 / 8 — Manual Stash Resolution",
    "docker_done":      "Step 4 / 8 — Docker Test",
    "push":             "Step 5 / 8 — Push & Create PR",
    "merge_conflict":   "Step 5 / 8 — Resolve Merge Conflict",
    "merge_manual":     "Step 5 / 8 — Manual Merge Resolution",
    "pr_created":       "Step 6 / 8 — Wait for PR Merge",
    "fresh_cloned":     "Step 6b / 8 — Review & .env Setup",
    "local_testing":    "Step 6c / 8 — Local Test",
    "server_running":   "Step 6c / 8 — Server Running (Review & Approve)",
    "deploy_approval":  "Step 7 / 8 — Deploy?",
    "local_done":       "Step 8 / 8 — Deploy",
    "deployed":         "Done — Deployment Complete",
    "done_no_deploy":   "Done — Deployment Skipped",
}

st.title("AI DevOps Agent")
st.caption(f"Stage: **{STAGE_LABELS.get(st.session_state.stage, st.session_state.stage)}**")
st.divider()


# ══════════════════════════════════════════════════════════════════
# STEP 1 — CLONE
# ══════════════════════════════════════════════════════════════════
if st.session_state.stage == "idle":
    repo_url = st.text_input("GitHub Repo URL")

    if st.button("Clone Repo"):
        if not repo_url.strip():
            st.error("Enter a GitHub repo URL first.")
            st.stop()
        if not TOOLS["git"]:
            st.error("git not found in PATH. Install Git before continuing.")
            st.stop()
        if not TOOLS["docker"]:
            st.error("❌ Docker not found in PATH. Install Docker Desktop before continuing.")
            st.stop()
        if not agent._check_docker_running():
            st.error("❌ Docker is not running!")
            st.warning("👉 Open Docker Desktop and wait for the whale icon to stop animating, then click Clone Repo again.")
            st.stop()
        try:
            with st.spinner("Authenticating..."):
                user = agent.get_authenticated_user(token)
                st.session_state.user = user
                log(f"Authenticated as: {user}")

            with st.spinner("Getting default branch..."):
                branch = agent.get_default_branch(repo_url, token)
                st.session_state.default_branch = branch
                log(f"Default branch: {branch}")

            with st.spinner("Forking repo..."):
                fork_url = agent.fork_repo(repo_url, token)
                st.session_state.fork_url = fork_url
                log(f"Fork ready: {fork_url}")

            with st.spinner("Cloning..."):
                folder = agent.download_repo(repo_url, fork_url, branch)
                st.session_state.folder = folder
                st.session_state.saved_repo_url = repo_url
                log(f"Cloned to: {folder}")

            st.session_state.stage = "cloned"
            try:
                subprocess.Popen(["code", os.path.abspath(folder)])
            except Exception:
                pass
            st.rerun()
        except Exception as e:
            st.error(str(e))


# ══════════════════════════════════════════════════════════════════
# STEP 2 — PAUSE: EDIT + ENV VARS + NOTES
# ══════════════════════════════════════════════════════════════════
if st.session_state.stage == "cloned":
    folder   = st.session_state.folder
    repo_url = st.session_state.saved_repo_url

    st.subheader("Edit Files")
    all_files = sorted([
        f for f in os.listdir(folder)
        if not f.startswith(".") and f not in ("__pycache__", "_test_venv")
    ])
    sel_file = st.selectbox("Select file to edit", all_files)
    sel_path = os.path.join(folder, sel_file)
    if os.path.isfile(sel_path):
        with open(sel_path, "r", errors="ignore") as fh:
            current_content = fh.read()
        edited = st.text_area("Content", current_content, height=300, key="file_editor")
        if st.button("Save File"):
            with open(sel_path, "w") as fh:
                fh.write(edited)
            st.success("Saved")

    st.divider()

    st.markdown("**Agent Notes** *(hints for Dockerfile generation — e.g. 'entry is src/main.py', 'use port 8080')*")
    notes_path     = os.path.join(folder, "_agent_notes.txt")
    existing_notes = open(notes_path).read() if os.path.exists(notes_path) else ""
    notes_input    = st.text_area("Notes", existing_notes, height=80, key="notes_area")

    st.divider()

    st.markdown("**Environment Variables**")
    env_file = os.path.join(folder, ".env")
    if os.path.exists(env_file):
        st.info("Found existing .env — edit below or leave as-is.")
        existing_env = open(env_file).read()
    else:
        existing_env = ""

    needed_vars = agent._detect_env_var_needs(folder)
    if needed_vars:
        st.caption(f"Likely needed: {', '.join(needed_vars)}")

    env_text = st.text_area("KEY=VALUE (one per line)", existing_env, key="env_area")

    st.divider()

    req_missing = not any(
        os.path.exists(os.path.join(folder, f))
        for f in ["requirements.txt", "package.json", "go.mod", "Gemfile",
                  "Cargo.toml", "composer.json", "pom.xml", "Pipfile"]
    )
    auto_gen_reqs = False
    if req_missing:
        auto_gen_reqs = st.checkbox("Auto-generate missing requirements file", value=True)

    if st.button("Done Editing — Generate Dockerfile"):
        if notes_input.strip():
            with open(notes_path, "w") as fh:
                fh.write(notes_input.strip())
            log("Saved _agent_notes.txt")

        env_vars = {}
        for line in env_text.strip().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                env_vars[k.strip()] = v.strip()
        if env_vars:
            with open(env_file, "w") as fh:
                for k, v in env_vars.items():
                    fh.write(f"{k}={v}\n")
            gitignore = os.path.join(folder, ".gitignore")
            if os.path.exists(gitignore):
                gi = open(gitignore).read()
                if ".env" not in gi:
                    open(gitignore, "a").write("\n.env\n")
            else:
                open(gitignore, "w").write(".env\n")
            log(f"Saved {len(env_vars)} env vars to .env")

        st.session_state.auto_gen_reqs = auto_gen_reqs
        st.session_state.stage = "docker"
        st.rerun()


# ══════════════════════════════════════════════════════════════════
# STEP 3 — CREATE BRANCH + GENERATE DOCKERFILE
# ══════════════════════════════════════════════════════════════════
if st.session_state.stage == "docker":
    folder         = st.session_state.folder
    fork_url       = st.session_state.fork_url
    repo_url       = st.session_state.saved_repo_url
    default_branch = st.session_state.default_branch

    st.subheader("Generating Dockerfile")
    st.info("Setting up branch and generating Dockerfile — this may take a minute.")
    if st.button("← Back to Edit Files (wrong framework detected?)"):
        st.session_state.stage = "cloned"
        st.rerun()
    st.divider()

    if st.button("Generate Dockerfile"):
        conflict_hit = False

        with st.spinner("Creating ai-docker-setup branch..."):
            try:
                status = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=folder, capture_output=True, text=True, check=True
                )
                has_changes = bool(status.stdout.strip())

                subprocess.run(["git", "remote", "set-url", "origin",   fork_url], cwd=folder, check=True)
                subprocess.run(["git", "remote", "set-url", "upstream", repo_url], cwd=folder, check=True)

                if has_changes:
                    log("Stashing local changes...")
                    subprocess.run(
                        ["git", "stash", "push", "-u", "-m", f"agent-stash-{int(time.time())}"],
                        cwd=folder, check=True
                    )

                subprocess.run(["git", "fetch", "upstream", default_branch], cwd=folder, check=True)
                subprocess.run(
                    ["git", "checkout", "-B", default_branch, f"upstream/{default_branch}"],
                    cwd=folder, check=True
                )
                subprocess.run(["git", "checkout", "-B", "ai-docker-setup"], cwd=folder, check=True)
                log("Checked out ai-docker-setup branch")

                if has_changes:
                    pop = subprocess.run(
                        ["git", "stash", "pop"], cwd=folder, capture_output=True, text=True
                    )
                    if pop.returncode == 0:
                        _df_path = os.path.join(folder, "Dockerfile")
                        _upstream_df = subprocess.run(
                            ["git", "show", f"upstream/{default_branch}:Dockerfile"],
                            cwd=folder, capture_output=True, text=True
                        )
                        _user_df = open(_df_path, "r", errors="ignore").read() if os.path.exists(_df_path) else None
                        _upstream_df_content = _upstream_df.stdout if _upstream_df.returncode == 0 else None
                        st.session_state["user_dockerfile"] = _user_df
                        st.session_state["user_has_dockerfile_changes"] = (
                            _user_df is not None
                            and _upstream_df_content is not None
                            and _user_df.strip() != _upstream_df_content.strip()
                        )
                        if st.session_state["user_has_dockerfile_changes"]:
                            log("Local Dockerfile changes detected — will preserve after AI generation")
                    if pop.returncode != 0:
                        log("Stash conflict — asking GPT-4o...")

                        conflict_files_raw = subprocess.run(
                            ["git", "diff", "--name-only", "--diff-filter=U"],
                            cwd=folder, capture_output=True, text=True
                        )
                        conflict_files = [
                            f.strip() for f in conflict_files_raw.stdout.splitlines() if f.strip()
                        ]

                        file_contents = {}
                        for fn in conflict_files:
                            fp = os.path.join(folder, fn)
                            try:
                                file_contents[fn] = open(fp, "r", errors="ignore").read(3000)
                            except Exception:
                                file_contents[fn] = "(unreadable)"

                        ctx_text  = "\n\n".join(f"--- {fn} ---\n{c}" for fn, c in file_contents.items())
                        stash_out = pop.stdout + pop.stderr

                        from openai import OpenAI
                        client = OpenAI(api_key=openai_key)

                        r1 = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "You are a Git expert. Be concise."},
                                {"role": "user", "content": f"""
Stash pop conflict when re-applying local changes on top of upstream '{default_branch}'.

CONFLICTING FILES:
{ctx_text}

STASH OUTPUT:
{stash_out}

1. Explain the conflict in simple terms
2. Recommend: OURS (local changes) or THEIRS (upstream) and WHY
"""}
                            ], temperature=0.1
                        )
                        analysis = r1.choices[0].message.content.strip()

                        r2 = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "Return ONLY valid JSON, nothing else."},
                                {"role": "user", "content": f"""
Conflict:\n{stash_out}\n\nFiles:\n{ctx_text[:1500]}

Return ONLY:
{{"strategy":"ours or theirs","reason":"one sentence","files":[{{"file":"name","action":"ours or theirs","reason":"why"}}]}}
"""}
                            ], temperature=0
                        )
                        raw = r2.choices[0].message.content.strip()
                        if raw.startswith("```"):
                            raw = "\n".join(
                                l for l in raw.splitlines() if not l.strip().startswith("```")
                            ).strip()
                        try:
                            strategy = json.loads(raw)
                        except Exception:
                            strategy = {"strategy": "ours", "reason": "Parse error — defaulting to ours", "files": []}

                        st.session_state.stash_conflict = {
                            "files":    conflict_files,
                            "analysis": analysis,
                            "strategy": strategy,
                        }
                        st.session_state.stage = "stash_conflict"
                        conflict_hit = True
                        st.rerun()

            except Exception as e:
                st.error(f"Branch setup failed: {e}")
                st.stop()

        if not conflict_hit:
            with st.spinner("Ensuring requirements file..."):
                _patch_input()
                try:
                    scan_ctx = agent.deep_scan_repo(folder)
                    if st.session_state.get("auto_gen_reqs", False):
                        agent._ensure_requirements(folder, scan_ctx, openai_key)
                        log("Requirements ensured")
                finally:
                    _restore_input()

            with st.spinner("Generating Dockerfile (GPT-4o)..."):
                _patch_input()
                try:
                    dockerfile, context = agent.generate_dockerfile_with_openai(folder, openai_key)
                    st.session_state.context = context
                    log("Dockerfile generated")
                    if st.session_state.get("user_has_dockerfile_changes") and st.session_state.get("user_dockerfile"):
                        _df_path = os.path.join(folder, "Dockerfile")
                        with open(_df_path, "w", encoding="utf-8") as _f:
                            _f.write(st.session_state["user_dockerfile"])
                        log("Restored user's local Dockerfile changes on top of AI generation")
                except Exception as e:
                    st.error(f"Dockerfile generation failed: {e}")
                    st.stop()
                finally:
                    _restore_input()

            st.session_state.stage = "docker_done"
            st.rerun()


# ══════════════════════════════════════════════════════════════════
# STEP 3b — STASH CONFLICT
# ══════════════════════════════════════════════════════════════════
if st.session_state.stage == "stash_conflict":
    folder = st.session_state.folder
    data   = st.session_state.stash_conflict

    st.subheader("Stash Conflict Detected")
    st.warning(f"Conflicting files: `{'`, `'.join(data['files'])}`")

    with st.expander("GPT-4o Analysis", expanded=True):
        st.markdown(data["analysis"])

    strategy = data["strategy"]
    st.markdown(f"**Recommended strategy:** `{strategy.get('strategy','ours').upper()}` — {strategy.get('reason','')}")
    for f_item in strategy.get("files", []):
        st.write(f"  • `{f_item.get('file')}` → keep **{f_item.get('action','ours').upper()}** — {f_item.get('reason','')}")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Accept LLM Resolution"):
            overall  = strategy.get("strategy", "ours")
            file_map = {fi.get("file"): fi.get("action", overall) for fi in strategy.get("files", [])}
            for fn in data["files"]:
                side = "--ours" if file_map.get(fn, overall) == "ours" else "--theirs"
                subprocess.run(["git", "checkout", side, fn], cwd=folder, capture_output=True)
                subprocess.run(["git", "add", fn],            cwd=folder, capture_output=True)
            log(f"Stash conflict resolved ({overall})")

            with st.spinner("Generating Dockerfile..."):
                _patch_input()
                try:
                    scan_ctx = agent.deep_scan_repo(folder)
                    agent._ensure_requirements(folder, scan_ctx, openai_key)
                    dockerfile, context = agent.generate_dockerfile_with_openai(folder, openai_key)
                    st.session_state.context = context
                    log("Dockerfile generated")
                finally:
                    _restore_input()

            st.session_state.stash_conflict = None
            st.session_state.stage = "docker_done"
            st.rerun()

    with col2:
        if st.button("Resolve Manually (VS Code)"):
            _open_vscode(folder)
            st.session_state.stage = "stash_manual"
            st.rerun()


# ══════════════════════════════════════════════════════════════════
# STEP 3c — MANUAL STASH RESOLUTION
# ══════════════════════════════════════════════════════════════════
if st.session_state.stage == "stash_manual":
    folder = st.session_state.folder
    st.subheader("Manual Stash Conflict Resolution")
    st.info(f"Open this folder in your editor: `{os.path.abspath(folder)}`")
    st.markdown("""
1. Find files with `<<<<<<<` markers in VS Code
2. Edit and keep the version you want, remove all conflict markers
3. Save the files
4. In your terminal: `git add .`
5. Click **Done** below
""")
    if st.button("Done — Generate Dockerfile"):
        with st.spinner("Generating Dockerfile..."):
            _patch_input()
            try:
                scan_ctx = agent.deep_scan_repo(folder)
                agent._ensure_requirements(folder, scan_ctx, openai_key)
                dockerfile, context = agent.generate_dockerfile_with_openai(folder, openai_key)
                st.session_state.context = context
                log("Dockerfile generated after manual resolution")
            finally:
                _restore_input()

        st.session_state.stash_conflict = None
        st.session_state.stage = "docker_done"
        st.rerun()


# ══════════════════════════════════════════════════════════════════
# STEP 4 — DOCKER TEST
# ══════════════════════════════════════════════════════════════════
if st.session_state.stage == "docker_done":
    folder  = st.session_state.folder
    context = st.session_state.context

    st.subheader("Dockerfile")
    dockerfile_path = os.path.join(folder, "Dockerfile")
    if os.path.exists(dockerfile_path):
        with open(dockerfile_path) as fh:
            st.code(fh.read(), language="dockerfile")

    lang      = context.get("detected_language", "?")
    framework = context.get("detected_framework", "?")
    ml_type   = context.get("ml_type", "?")
    entries   = context.get("entry_points_found", [])
    st.caption(f"Detected: {lang} / {framework} / ML: {ml_type} | Entry: {entries}")

    if not TOOLS["docker"]:
        st.error("Docker not found in PATH. Install Docker Desktop and ensure it is running before running this step.")
    elif st.button("Run Docker Test"):
        app_name = os.path.basename(folder).lower().replace("_", "-")

        if not agent._check_docker_running():
            st.error("❌ Docker is not running!")
            st.warning("👉 Open Docker Desktop, wait for it to fully start (whale icon stops animating), then click Run Docker Test again.")
            st.stop()

        with st.spinner("Building & testing Docker image..."):
            try:
                _patch_input()
                try:
                    passed = agent.test_docker_image(folder, app_name, context, openai_key, max_retries=3)
                finally:
                    _restore_input()

                if passed:
                    log("Docker test passed")
                    st.success("Docker test passed")
                    st.session_state.stage = "push"
                    st.rerun()
                else:
                    st.error("Docker test failed after 3 retries + nuclear regeneration")
                    log("Docker test failed (including nuclear round)")
                    st.warning("The app built but failed at runtime — likely a code issue.")
                    col_bt1, col_bt2 = st.columns(2)
                    with col_bt1:
                        if st.button("✏️ Go Back to Edit Files (Step 2)"):
                            st.session_state.stage = "cloned"
                            st.rerun()
                    with col_bt2:
                        if st.button("🔧 Go Back to Edit Dockerfile (Step 3)"):
                            st.session_state.stage = "docker"
                            st.rerun()
            except Exception as e:
                st.error(str(e))


# ══════════════════════════════════════════════════════════════════
# STEP 5 — PUSH + CONFLICT CHECK + CREATE PR
# ══════════════════════════════════════════════════════════════════
if st.session_state.stage == "push":
    folder         = st.session_state.folder
    fork_url       = st.session_state.fork_url
    repo_url       = st.session_state.saved_repo_url
    default_branch = st.session_state.default_branch

    st.subheader("Push Branch & Create PR")
    if st.button("← Back to Docker Test (re-test before pushing?)"):
        st.session_state.stage = "docker_done"
        st.rerun()
    if st.button("Push Branch & Create PR"):
        conflict_hit = False

        with st.spinner("Committing and pushing..."):
            try:
                marker = os.path.join(folder, "ai_changes.txt")
                with open(marker, "a") as fh:
                    fh.write("AI change\n")
                subprocess.run(["git", "add", "."], cwd=folder, check=True)
                result = subprocess.run(
                    ["git", "commit", "-m", "AI Docker setup"],
                    cwd=folder, capture_output=True, text=True
                )
                if "nothing to commit" not in result.stdout.lower():
                    log("Changes committed")

                agent.push_branch(folder, fork_url, token)
                log("Branch pushed to fork")

            except Exception as e:
                st.error(f"Push failed: {e}")
                st.stop()

        with st.spinner("Checking for upstream merge conflicts..."):
            try:
                conflict_result = agent.check_upstream_merge_conflicts(
                    folder, repo_url, token, default_branch
                )

                if conflict_result.get("error"):
                    log(f"Conflict check warning: {conflict_result['error']}")

                if conflict_result.get("has_conflicts"):
                    conflict_files = conflict_result["conflict_files"]
                    conflict_text  = conflict_result["conflict_text"]
                    log(f"Merge conflicts in: {conflict_files}")

                    file_contents = {}
                    for fn in conflict_files:
                        fp = os.path.join(folder, fn)
                        try:
                            file_contents[fn] = open(fp, "r", errors="ignore").read(3000)
                        except Exception:
                            file_contents[fn] = "(unreadable)"

                    ctx_text = "\n\n".join(f"--- {fn} ---\n{c}" for fn, c in file_contents.items())

                    from openai import OpenAI
                    client = OpenAI(api_key=openai_key)

                    r1 = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a Git expert. Be concise and actionable."},
                            {"role": "user", "content": f"""
Merge conflicts when merging 'ai-docker-setup' into '{default_branch}' in {repo_url}.

CONFLICTING FILES:
{ctx_text}

CONFLICT OUTPUT:
{conflict_text[:2000]}

1. Explain each conflict
2. Recommend: ours (ai-docker-setup) or theirs (main) and WHY
"""}
                        ], temperature=0.1
                    )
                    analysis = r1.choices[0].message.content.strip()

                    r2 = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "Return ONLY valid JSON, nothing else."},
                            {"role": "user", "content": f"""
{conflict_text[:1500]}
{ctx_text[:1000]}

Return ONLY:
{{"strategy":"ours or theirs","reason":"one sentence","files":[{{"file":"name","action":"ours or theirs","reason":"why"}}]}}
"""}
                        ], temperature=0
                    )
                    raw = r2.choices[0].message.content.strip()
                    if raw.startswith("```"):
                        raw = "\n".join(
                            l for l in raw.splitlines() if not l.strip().startswith("```")
                        ).strip()
                    try:
                        strategy = json.loads(raw)
                    except Exception:
                        strategy = {"strategy": "ours", "reason": "Parse error", "files": []}

                    st.session_state.merge_conflict = {
                        "files":    conflict_files,
                        "analysis": analysis,
                        "strategy": strategy,
                    }
                    st.session_state.stage = "merge_conflict"
                    conflict_hit = True
                    st.rerun()

            except Exception as e:
                log(f"Conflict check error: {e}")

        if not conflict_hit:
            with st.spinner("Creating pull request..."):
                try:
                    commit_check = subprocess.run(
                        ["git", "log", f"upstream/{default_branch}..ai-docker-setup", "--oneline"],
                        cwd=folder, capture_output=True, text=True
                    )

                    df_diff = subprocess.run(
                        ["git", "diff", f"upstream/{default_branch}", "--", "Dockerfile"],
                        cwd=folder, capture_output=True, text=True
                    )
                    dockerfile_differs = bool(df_diff.stdout.strip())

                    if not commit_check.stdout.strip() and not dockerfile_differs:
                        log("No new commits — Dockerfile already on upstream. Cloning fresh...")
                        auth_url   = repo_url.replace("https://", f"https://{token}@")
                        repo_name  = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
                        clone_dest = os.path.join(WORKSPACE, repo_name)
                        if os.path.exists(folder):
                            agent.safe_rmtree(folder)
                        _git_clone(auth_url, default_branch, clone_dest)
                        st.session_state.folder = clone_dest
                        log(f"Fresh clone: {clone_dest}")
                        st.session_state.stage = "pre_deploy"
                    else:
                        pr_url = agent.create_pull_request(
                            repo_url, token, st.session_state.user, default_branch
                        )
                        st.session_state.pr_url = pr_url
                        log(f"PR created: {pr_url}")
                        st.session_state.stage = "pr_created"

                    st.rerun()

                except Exception as e:
                    st.error(f"PR creation failed: {e}")


# ══════════════════════════════════════════════════════════════════
# STEP 5b — MERGE CONFLICT
# ══════════════════════════════════════════════════════════════════
if st.session_state.stage == "merge_conflict":
    folder = st.session_state.folder
    data   = st.session_state.merge_conflict

    st.subheader("Merge Conflict Detected")
    st.warning(f"Conflicting files: `{'`, `'.join(data['files'])}`")

    with st.expander("GPT-4o Analysis", expanded=True):
        st.markdown(data["analysis"])

    strategy = data["strategy"]
    st.markdown(f"**Recommended:** `{strategy.get('strategy','ours').upper()}` — {strategy.get('reason','')}")
    for f_item in strategy.get("files", []):
        st.write(f"  • `{f_item.get('file')}` → keep **{f_item.get('action','ours').upper()}** — {f_item.get('reason','')}")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Accept LLM Resolution & Create PR"):
            folder         = st.session_state.folder
            fork_url       = st.session_state.fork_url
            repo_url       = st.session_state.saved_repo_url
            default_branch = st.session_state.default_branch

            overall  = strategy.get("strategy", "ours")
            file_map = {fi.get("file"): fi.get("action", overall) for fi in strategy.get("files", [])}
            for fn in data["files"]:
                side = "--ours" if file_map.get(fn, overall) == "ours" else "--theirs"
                subprocess.run(["git", "checkout", side, fn], cwd=folder, capture_output=True)
                subprocess.run(["git", "add", fn],            cwd=folder, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "Resolve merge conflicts (AI suggestion)"],
                cwd=folder, capture_output=True
            )
            with st.spinner("Re-pushing and creating PR..."):
                agent.push_branch(folder, fork_url, token)
                log(f"Conflicts resolved and re-pushed ({overall})")
                pr_url = agent.create_pull_request(
                    repo_url, token, st.session_state.user, default_branch
                )
                st.session_state.pr_url = pr_url
                log(f"PR created: {pr_url}")

            st.session_state.merge_conflict = None
            st.session_state.stage = "pr_created"
            st.rerun()

    with col2:
        if st.button("Resolve Manually (VS Code)"):
            _open_vscode(folder)
            st.session_state.stage = "merge_manual"
            st.rerun()


# ══════════════════════════════════════════════════════════════════
# STEP 5c — MANUAL MERGE RESOLUTION
# ══════════════════════════════════════════════════════════════════
if st.session_state.stage == "merge_manual":
    folder         = st.session_state.folder
    fork_url       = st.session_state.fork_url
    repo_url       = st.session_state.saved_repo_url
    default_branch = st.session_state.default_branch

    st.subheader("Manual Merge Conflict Resolution")
    st.info(f"Open this folder in your editor: `{os.path.abspath(folder)}`")
    st.markdown("""
1. Find files with `<<<<<<<` markers in VS Code
2. Keep the version you want and remove all conflict markers
3. Save the files
4. In your terminal: `git add .`
5. Click **Done** below
""")
    if st.button("Done — Re-push & Create PR"):
        with st.spinner("Re-pushing and creating PR..."):
            subprocess.run(
                ["git", "commit", "-m", "Resolve merge conflicts (manual)"],
                cwd=folder, capture_output=True
            )
            agent.push_branch(folder, fork_url, token)
            log("Re-pushed after manual merge resolution")
            pr_url = agent.create_pull_request(
                repo_url, token, st.session_state.user, default_branch
            )
            st.session_state.pr_url = pr_url
            log(f"PR created: {pr_url}")
        st.session_state.merge_conflict = None
        st.session_state.stage = "pr_created"
        st.rerun()


# ══════════════════════════════════════════════════════════════════
# STEP 6 — POLL PR STATUS
# ══════════════════════════════════════════════════════════════════
if st.session_state.stage == "pr_created":
    pr_url         = st.session_state.pr_url
    folder         = st.session_state.folder
    repo_url       = st.session_state.saved_repo_url
    default_branch = st.session_state.default_branch

    st.subheader("Pull Request")
    st.markdown(f"**[Open PR on GitHub]({pr_url})**")
    st.info("Review the PR on GitHub, merge it, then click Check Status.")

    if st.button("Check PR Status"):
        with st.spinner("Checking PR status..."):
            try:
                repo       = repo_url.replace("https://github.com/", "").rstrip("/")
                headers    = agent.make_github_headers(token)
                fork_owner = st.session_state.user
                repo_owner = repo.split("/")[0]
                head_param = "ai-docker-setup" if fork_owner == repo_owner else f"{fork_owner}:ai-docker-setup"

                r_open = requests.get(
                    f"https://api.github.com/repos/{repo}/pulls",
                    headers=headers, params={"head": head_param, "state": "open"}
                )
                open_prs = r_open.json() if r_open.status_code == 200 else []

                r_closed = requests.get(
                    f"https://api.github.com/repos/{repo}/pulls",
                    headers=headers, params={"head": head_param, "state": "closed"}
                )
                closed_prs = r_closed.json() if r_closed.status_code == 200 else []

                if isinstance(closed_prs, list) and closed_prs and closed_prs[0].get("merged_at"):
                    log("PR merged!")
                    st.success("PR merged — fresh cloning...")

                    auth_url   = repo_url.replace("https://", f"https://{token}@")
                    repo_name  = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
                    clone_dest = os.path.join(WORKSPACE, repo_name)
                    if os.path.exists(folder):
                        agent.safe_rmtree(folder)
                    _git_clone(auth_url, default_branch, clone_dest)
                    st.session_state.folder = clone_dest
                    log(f"Fresh clone: {clone_dest}")

                    _open_vscode(clone_dest)

                    st.session_state.stage = "fresh_cloned"
                    st.rerun()

                elif isinstance(closed_prs, list) and closed_prs and not closed_prs[0].get("merged_at"):
                    st.error("PR was closed/rejected. Fix your changes and push again.")
                    log("PR closed/rejected")

                elif isinstance(open_prs, list) and open_prs:
                    pr      = open_prs[0]
                    m_state = pr.get("mergeable_state", "unknown")
                    if m_state in ("dirty", "blocked", "behind"):
                        st.warning(f"PR has issues: `{m_state}` — resolve conflicts on GitHub then check again")
                    else:
                        st.info(f"PR is open (`{m_state}`) — merge it on GitHub, then check again")
                else:
                    st.warning("PR not found yet — try again in a moment")

            except Exception as e:
                st.error(str(e))


# ══════════════════════════════════════════════════════════════════
# PRE-DEPLOY (Dockerfile already existed — PR skipped)
# ══════════════════════════════════════════════════════════════════
if st.session_state.stage == "pre_deploy":
    folder = st.session_state.folder
    _open_vscode(folder)
    st.session_state.stage = "fresh_cloned"
    st.rerun()


# ══════════════════════════════════════════════════════════════════
# STEP 6b — FRESH CLONE REVIEW: .env setup + VS Code
# ══════════════════════════════════════════════════════════════════
if st.session_state.stage == "fresh_cloned":
    folder = st.session_state.folder

    st.subheader("Fresh Clone Ready — Review & Setup")
    st.info(f"Open this folder in your editor: `{os.path.abspath(folder)}`")
    st.markdown("Review the code in VS Code. Since `.env` is gitignored, set your environment variables below before testing.")

    env_file = os.path.join(folder, ".env")
    if os.path.exists(env_file):
        st.success("Found existing `.env` — edit below or leave as-is.")
        existing_env = open(env_file).read()
    else:
        st.warning("No `.env` found (gitignored). Add your environment variables below:")
        existing_env = ""

    env_text = st.text_area("Environment Variables (KEY=VALUE, one per line)", existing_env, key="fresh_env_area")

    st.divider()
    if st.button("Save .env & Run Local Test"):
        env_vars = {}
        for line in env_text.strip().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                env_vars[k.strip()] = v.strip()
        if env_vars:
            with open(env_file, "w") as fh:
                for k, v in env_vars.items():
                    fh.write(f"{k}={v}\n")
            gi_path = os.path.join(folder, ".gitignore")
            if os.path.exists(gi_path):
                gi_content = open(gi_path).read()
                if ".env" not in gi_content:
                    open(gi_path, "a").write("\n.env\n")
            log(f"Saved {len(env_vars)} env vars to .env")
        st.session_state.stage = "local_testing"
        st.rerun()

    if st.button("Skip .env & Go to Deploy"):
        log("Skipped local test — going to deploy")
        _patch_input()
        try:
            st.session_state.context = agent.deep_scan_repo(folder)
        finally:
            _restore_input()
        st.session_state.stage = "deploy_approval"
        st.rerun()


# ══════════════════════════════════════════════════════════════════
# STEP 6c — LOCAL TESTING
# ══════════════════════════════════════════════════════════════════
def _find_free_port(preferred):
    import socket
    preferred = int(preferred)
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return str(port)
            except OSError:
                continue
    return str(preferred)


def _start_local_server(folder, ctx):
    folder    = os.path.abspath(folder)
    framework = ctx.get("detected_framework", "unknown")
    ml_type   = ctx.get("ml_type", "unknown")
    entries   = ctx.get("entry_points_found", [])
    entry     = entries[0] if entries else ""

    port_map = {
        "streamlit": "8502", "gradio": "7860",
        "fastapi": "8000", "fastapi_ml": "8000",
        "flask": "5000",   "flask_ml": "5000",
        "django": "8000",  "express": "3000", "fastify": "3000",
    }
    key       = ml_type if ml_type not in ("unknown", "none", "") else framework
    preferred = agent.detect_port_from_dockerfile(folder) or port_map.get(key, "8000")
    test_port = _find_free_port(preferred)

    venv_dir    = os.path.join(folder, "_test_venv")
    subprocess.run([sys.executable, "-m", "venv", venv_dir], capture_output=True, text=True)
    venv_python = (
        os.path.join(venv_dir, "Scripts", "python.exe") if os.name == "nt"
        else os.path.join(venv_dir, "bin", "python")
    )

    req_path = os.path.join(folder, "requirements.txt")
    if os.path.exists(req_path):
        subprocess.run(
            [venv_python, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"],
            cwd=folder, capture_output=True
        )

    cmd             = None
    success_signals = []

    if framework == "fastapi" or ml_type == "fastapi_ml":
        subprocess.run([venv_python, "-m", "pip", "install", "uvicorn", "--quiet"], capture_output=True)
        e       = ctx.get("fastapi_entry_file", entry)
        mod     = e.replace("\\", "/").replace("/", ".").replace(".py", "")
        app_var = ctx.get("app_variable_name", "app")
        cmd     = [venv_python, "-m", "uvicorn", f"{mod}:{app_var}", "--host", "0.0.0.0", "--port", str(test_port)]
        success_signals = ["application startup complete", "uvicorn running"]

    elif ml_type == "streamlit" or framework == "streamlit":
        subprocess.run([venv_python, "-m", "pip", "install", "streamlit", "--quiet"], capture_output=True)
        e   = ctx.get("streamlit_entry_file", entry)
        cmd = [venv_python, "-m", "streamlit", "run", e,
               "--server.headless=true", f"--server.port={test_port}", "--server.address=0.0.0.0"]
        success_signals = ["you can now view", "network url", "local url", "http://"]

    elif ml_type == "gradio" or framework == "gradio":
        subprocess.run([venv_python, "-m", "pip", "install", "gradio", "--quiet"], capture_output=True)
        e   = ctx.get("gradio_entry_file", entry)
        cmd = [venv_python, e]
        success_signals = ["running on", "local url", "gradio"]

    elif framework == "flask" or ml_type == "flask_ml":
        subprocess.run([venv_python, "-m", "pip", "install", "flask", "--quiet"], capture_output=True)
        e   = ctx.get("flask_entry_file", entry)
        mod = e.replace("\\", "/").replace(".py", "").replace("/", ".")
        cmd = [venv_python, "-m", "flask", "--app", mod, "run", "--host=0.0.0.0", f"--port={test_port}"]
        success_signals = ["running on", "debugger is active"]

    elif framework == "django":
        subprocess.run([venv_python, "-m", "pip", "install", "django", "--quiet"], capture_output=True)
        cmd = [venv_python, "manage.py", "runserver", f"0.0.0.0:{test_port}"]
        success_signals = ["starting development server", "quit the server"]

    if not cmd:
        return None, venv_dir, test_port, []

    proc_env = os.environ.copy()
    env_file = os.path.join(folder, ".env")
    if os.path.exists(env_file):
        for _line in open(env_file):
            _line = _line.strip()
            if "=" in _line and not _line.startswith("#"):
                _k, _, _v = _line.partition("=")
                proc_env[_k.strip()] = _v.strip()

    collected_lines = []
    lock = threading.Lock()

    proc = subprocess.Popen(
        cmd, cwd=folder,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="replace",
        env=proc_env,
    )

    def reader(stream):
        try:
            for raw_line in iter(stream.readline, ""):
                line = raw_line.rstrip()
                if line:
                    with lock:
                        collected_lines.append(line)
        except Exception:
            pass

    threading.Thread(target=reader, args=(proc.stdout,), daemon=True).start()
    threading.Thread(target=reader, args=(proc.stderr,), daemon=True).start()

    start_time = time.time()
    timeout    = 45

    FATAL_KEYWORDS = [
        "modulenotfounderror", "importerror", "no module named",
        "syntaxerror", "traceback (most recent", "typeerror", "error:",
        "address already in use",
    ]

    while time.time() - start_time < timeout:
        with lock:
            lines_lower = [l.lower() for l in collected_lines]
        if any(sig in l for l in lines_lower for sig in success_signals):
            break

        if any(kw in l for l in lines_lower for kw in FATAL_KEYWORDS):
            with lock:
                fatal_log = "\n".join(collected_lines)
            import re
            
# Find ALL file references, prefer user files over venv/framework files
            all_matches = re.findall(r'File "([^"]+\.py)", line (\d+)', fatal_log)
            # Filter out venv/framework files, prefer user project files
            user_matches = [
                (f, l) for f, l in all_matches
                if "_test_venv" not in f
                and "site-packages" not in f
                and "streamlit" not in f.lower()
            ]
            file_match_tuple = user_matches[-1] if user_matches else (all_matches[-1] if all_matches else None)
            if file_match_tuple and openai_key:
                failing_file, failing_line = file_match_tuple
                if os.path.isfile(failing_file):
                    try:
                        from openai import OpenAI
                        file_src = open(failing_file, "r", errors="ignore").read()
                        client   = OpenAI(api_key=openai_key)
                        resp     = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "You are a Python expert. Fix the bug in the file. Return ONLY the complete fixed file content, no markdown, no backticks."},
                                {"role": "user",   "content": f"ERROR:\n{fatal_log[-2000:]}\n\nFILE ({failing_file}):\n{file_src}"}
                            ],
                            temperature=0.1,
                        )
                        fixed_src = resp.choices[0].message.content.strip()
                        if fixed_src.startswith("```"):
                            fixed_src = "\n".join(
                                l for l in fixed_src.splitlines()
                                if not l.strip().startswith("```")
                            ).strip()
                        with open(failing_file, "w", encoding="utf-8") as _f:
                            _f.write(fixed_src)
                        try:
                            proc.kill()
                            proc.wait(timeout=5)
                        except Exception:
                            pass
                        with lock:
                            collected_lines.clear()
                        proc = subprocess.Popen(
                            cmd, cwd=folder,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace",
                            env=proc_env,
                        )
                        threading.Thread(target=reader, args=(proc.stdout,), daemon=True).start()
                        threading.Thread(target=reader, args=(proc.stderr,), daemon=True).start()
                        start_time = time.time()
                        time.sleep(0.3)
                        continue
                    except Exception:
                        pass
            break

        if proc.poll() is not None:
            break
        time.sleep(0.3)

    time.sleep(0.5)
    with lock:
        final_lines = list(collected_lines)

    return proc, venv_dir, test_port, final_lines


if st.session_state.stage == "local_testing":
    folder = st.session_state.folder

    st.subheader("Local Test")
    st.info("This will install dependencies in an isolated venv, start your app, and let you verify it in the browser before deploying.")
    if st.button("← Back to .env Setup"):
        st.session_state.stage = "fresh_cloned"
        st.rerun()

    if st.button("Start Local Test Server"):
        test_folder = f"{folder}_test"
        if os.path.exists(test_folder):
            agent.safe_rmtree(test_folder)

        with st.spinner("Copying repo to isolated test folder..."):
            shutil.copytree(folder, test_folder, ignore=shutil.ignore_patterns("_test_venv", ".git"))
            env_src = os.path.join(folder, ".env")
            if os.path.exists(env_src):
                shutil.copy2(env_src, os.path.join(test_folder, ".env"))

        with st.spinner("Scanning repo..."):
            _patch_input()
            try:
                ctx = agent.deep_scan_repo(test_folder)
            finally:
                _restore_input()
            st.session_state.test_ctx = ctx

        with st.spinner("Installing dependencies & starting server (up to 45s)..."):
            proc, venv_dir, test_port, server_logs = _start_local_server(test_folder, ctx)

        if proc is not None and proc.poll() is None:
            st.session_state.test_proc   = proc
            st.session_state.test_folder = test_folder
            st.session_state.test_port   = test_port
            st.session_state.test_venv   = venv_dir
            log(f"Local test server started on port {test_port}")
            st.session_state.stage = "server_running"
            st.rerun()
        else:
            st.error("Server process exited before becoming ready.")
            if server_logs:
                with st.expander("Server output (crash log)", expanded=True):
                    st.code("\n".join(server_logs[-100:]))
            else:
                st.warning("No output captured. The process may have failed silently.")
            st.info("Common causes: missing env var, import error, port already in use, or DB unreachable at startup.")
            if os.path.exists(test_folder):
                agent.safe_rmtree(test_folder)
            if st.button("← Fix Environment Variables (go back to .env setup)"):
                st.session_state.stage = "fresh_cloned"
                st.rerun()
            if os.path.exists(test_folder):
                agent.safe_rmtree(test_folder)

    if st.button("Skip Local Test — Go to Deploy"):
        log("Local test skipped by user")
        _patch_input()
        try:
            st.session_state.context = agent.deep_scan_repo(folder)
        finally:
            _restore_input()
        st.session_state.stage = "deploy_approval"
        st.rerun()


# ══════════════════════════════════════════════════════════════════
# STEP 6c (continued) — SERVER RUNNING
# ══════════════════════════════════════════════════════════════════
if st.session_state.stage == "server_running":
    port        = st.session_state.get("test_port", "8000")
    folder      = st.session_state.folder
    test_folder = st.session_state.get("test_folder")

    st.subheader("Local Server Running — Test Your App")
    st.success("Server started successfully! Open the links below to verify your app.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**[http://localhost:{port}](http://localhost:{port})**")
    with col_b:
        st.markdown(f"**[http://127.0.0.1:{port}](http://127.0.0.1:{port})**")

    st.info("Test your app in the browser, then click **Stop Server & Continue** to proceed to deployment.")

    if st.button("Stop Server & Continue to Deploy"):
        proc = st.session_state.get("test_proc")
        if proc:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
            import time; time.sleep(1.5)
        if test_folder and os.path.exists(test_folder):
            agent.safe_rmtree(test_folder)
        st.session_state.test_proc   = None
        st.session_state.test_folder = None
        log("Local test server stopped — proceeding to deploy")
        _patch_input()
        try:
            st.session_state.context = agent.deep_scan_repo(folder)
        finally:
            _restore_input()
        st.session_state.stage = "deploy_approval"
        st.rerun()

    if st.button("Stop Server & Skip Deploy"):
        proc = st.session_state.get("test_proc")
        if proc:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
        if test_folder and os.path.exists(test_folder):
            agent.safe_rmtree(test_folder)
        st.session_state.test_proc   = None
        st.session_state.test_folder = None
        log("Local test server stopped — skipping deploy")
        st.session_state.stage = "done_no_deploy"
        st.rerun()


# ══════════════════════════════════════════════════════════════════
# STEP 7 — DEPLOY APPROVAL
# ══════════════════════════════════════════════════════════════════
if st.session_state.stage == "deploy_approval":
    st.subheader("Deploy?")
    st.success("Code is pushed, PR merged, and local test complete. Deploy now?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes — Deploy"):
            log("Deploy approved")
            st.session_state.stage = "local_done"
            st.rerun()
    with col2:
        if st.button("No — Skip Deployment"):
            log("Deploy skipped by user")
            st.session_state.stage = "done_no_deploy"
            st.rerun()


# ══════════════════════════════════════════════════════════════════
# STEP 8 — COLLECT DEPLOY INFO + DEPLOY
# ══════════════════════════════════════════════════════════════════
if st.session_state.stage == "local_done":
    folder   = st.session_state.folder
    fork_url = st.session_state.fork_url

    st.subheader("Deployment")

    deploy_target = st.selectbox("Where to deploy?", ["render", "railway", "aws", "azure"])
    app_name      = st.text_input("App Name")

    st.markdown("**Environment Variables**")
    env_file = os.path.join(folder, ".env")
    pre_env  = open(env_file).read() if os.path.exists(env_file) else ""
    env_text = st.text_area("KEY=VALUE (one per line)", pre_env, key="deploy_env")
    env_vars = {}
    for line in env_text.strip().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            env_vars[k.strip()] = v.strip()

    st.markdown("**Credentials**")
    creds = {}

    if deploy_target == "aws":
        creds["aws"] = {
            "access_key": st.text_input("AWS_ACCESS_KEY_ID"),
            "secret_key": st.text_input("AWS_SECRET_ACCESS_KEY", type="password"),
            "region":     st.text_input("AWS_REGION", value="us-east-1"),
            "app_name":   app_name,
        }
    elif deploy_target == "azure":
        creds["azure"] = {
            "client_id":       st.text_input("AZURE_CLIENT_ID"),
            "client_secret":   st.text_input("AZURE_CLIENT_SECRET",   type="password"),
            "tenant_id":       st.text_input("AZURE_TENANT_ID"),
            "subscription_id": st.text_input("AZURE_SUBSCRIPTION_ID"),
            "resource_group":  st.text_input("AZURE_RESOURCE_GROUP"),
            "dockerhub_user":  st.text_input("Docker Hub Username"),
            "dockerhub_pass":  st.text_input("Docker Hub Password",    type="password"),
            "app_name":        app_name,
        }
    elif deploy_target == "render":
        _render_env = {}
        if os.path.exists(os.path.join(folder, ".env")):
            for _line in open(os.path.join(folder, ".env")).readlines():
                _line = _line.strip()
                if "=" in _line and not _line.startswith("#"):
                    _k, _, _v = _line.partition("=")
                    _render_env[_k.strip()] = _v.strip()
        _render_key_default = _render_env.get("RENDER_API_KEY", "")
        render_api_key = st.text_input("RENDER_API_KEY", value=_render_key_default, type="password")
        creds["render"] = {
            "api_key":  render_api_key,
            "app_name": app_name,
        }
    elif deploy_target == "railway":
        creds["railway"] = {
            "token":          st.text_input("RAILWAY_TOKEN"),
            "dockerhub_user": st.text_input("Docker Hub Username"),
            "dockerhub_pass": st.text_input("Docker Hub Password", type="password"),
            "app_name":       app_name,
        }

    if st.button("Deploy Now"):
        if not app_name.strip():
            st.error("App name is required.")
            st.stop()

        for platform in creds:
            creds[platform]["fork_url"] = fork_url
            creds[platform]["folder"]   = folder
            creds[platform]["env_vars"] = env_vars

        _save_deploy_state_ui({
            "targets":  [deploy_target],
            "app_name": app_name,
            "folder":   folder,
            "fork_url": fork_url,
            "results":  {deploy_target: "PENDING"},
            "creds":    creds,
            "env_vars": env_vars,
        })

        with st.spinner(f"Deploying to {deploy_target}..."):
            try:
                results = agent.deploy_to_platforms([deploy_target], folder, fork_url, creds)
                log(f"Deploy complete: {results}")
                failed = [p for p, r in results.items() if str(r).startswith("FAILED")]
                if failed:
                    _save_deploy_state_ui({
                        "targets":  [deploy_target],
                        "app_name": app_name,
                        "folder":   folder,
                        "fork_url": fork_url,
                        "results":  results,
                        "creds":    creds,
                        "env_vars": env_vars,
                    })
                    st.session_state.deploy_results     = results
                    st.session_state.deploy_failed_info = {
                        "folder": folder, "fork_url": fork_url,
                        "app_name": app_name, "env_vars": env_vars, "creds": creds,
                    }
                    st.session_state.stage = "deploy_failed"
                    st.rerun()
                else:
                    _clear_deploy_state_ui()
                    st.session_state.deploy_results = results
                    st.session_state.stage = "deployed"
                    st.rerun()
            except Exception as e:
                err_msg = str(e)
                _save_deploy_state_ui({
                    "targets":  [deploy_target],
                    "app_name": app_name,
                    "folder":   folder,
                    "fork_url": fork_url,
                    "results":  {deploy_target: f"FAILED: {err_msg}"},
                    "creds":    creds,
                    "env_vars": env_vars,
                })
                st.session_state.deploy_results     = {deploy_target: f"FAILED: {err_msg}"}
                st.session_state.deploy_failed_info = {
                    "folder": folder, "fork_url": fork_url,
                    "app_name": app_name, "env_vars": env_vars, "creds": creds,
                }
                st.session_state.stage = "deploy_failed"
                st.rerun()


# ── Deploy Failed — re-enter credentials and retry ────────────────
if st.session_state.stage == "deploy_failed":
    results  = st.session_state.get("deploy_results", {})
    info     = st.session_state.get("deploy_failed_info", {})
    folder   = info.get("folder",   st.session_state.get("folder", ""))
    fork_url = info.get("fork_url", st.session_state.get("fork_url", ""))
    app_name = info.get("app_name", "")
    env_vars = info.get("env_vars", {})

    st.subheader("Deployment Failed")

    for platform, result in results.items():
        if str(result).startswith("FAILED"):
            err_msg = result.replace("FAILED: ", "")
            st.error(f"❌ **{platform.upper()}** failed")

            if _is_credential_error_ui(err_msg):
                st.warning(f"🔑 Credential error — {CREDENTIAL_HINTS.get(platform, 'Check your keys.')}")
            elif _is_quota_error_ui(err_msg):
                st.warning(f"💳 Quota/billing issue — {QUOTA_HINTS.get(platform, 'Check your plan.')}")
            else:
                st.warning(f"Error: `{err_msg[:300]}`")

            with st.expander("Full error details", expanded=False):
                st.code(err_msg)

            st.markdown(f"**Re-enter credentials for {platform.upper()}:**")
            new_creds = {}

            if platform == "aws":
                new_creds = {
                    "access_key": st.text_input("AWS_ACCESS_KEY_ID",     key="retry_aws_ak"),
                    "secret_key": st.text_input("AWS_SECRET_ACCESS_KEY", key="retry_aws_sk", type="password"),
                    "region":     st.text_input("AWS_REGION",            key="retry_aws_rg", value="us-east-1"),
                    "app_name": app_name, "folder": folder, "fork_url": fork_url, "env_vars": env_vars,
                }
            elif platform == "azure":
                new_creds = {
                    "client_id":       st.text_input("AZURE_CLIENT_ID",       key="retry_az_cid"),
                    "client_secret":   st.text_input("AZURE_CLIENT_SECRET",   key="retry_az_cs",  type="password"),
                    "tenant_id":       st.text_input("AZURE_TENANT_ID",       key="retry_az_tid"),
                    "subscription_id": st.text_input("AZURE_SUBSCRIPTION_ID", key="retry_az_sub"),
                    "resource_group":  st.text_input("AZURE_RESOURCE_GROUP",  key="retry_az_rg"),
                    "dockerhub_user":  st.text_input("Docker Hub Username",   key="retry_az_dhu"),
                    "dockerhub_pass":  st.text_input("Docker Hub Password",   key="retry_az_dhp", type="password"),
                    "app_name": app_name, "folder": folder, "fork_url": fork_url, "env_vars": env_vars,
                }
            elif platform == "render":
                new_creds = {
                    "api_key":  st.text_input("RENDER_API_KEY", key="retry_render_key", type="password"),
                    "app_name": app_name, "folder": folder, "fork_url": fork_url, "env_vars": env_vars,
                }
            elif platform == "railway":
                new_creds = {
                    "token":          st.text_input("RAILWAY_TOKEN",       key="retry_rw_tok", type="password"),
                    "dockerhub_user": st.text_input("Docker Hub Username", key="retry_rw_dhu"),
                    "dockerhub_pass": st.text_input("Docker Hub Password", key="retry_rw_dhp", type="password"),
                    "app_name": app_name, "folder": folder, "fork_url": fork_url, "env_vars": env_vars,
                }

            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Retry {platform.upper()}", key=f"retry_{platform}"):
                    with st.spinner(f"Retrying {platform.upper()}..."):
                        try:
                            retry_results = agent.deploy_to_platforms(
                                [platform], folder, fork_url, {platform: new_creds}
                            )
                            log(f"Retry result: {retry_results}")
                            url = retry_results.get(platform, "")
                            if str(url).startswith("FAILED"):
                                st.session_state.deploy_results[platform] = url
                                _save_deploy_state_ui({
                                    "targets": [platform], "app_name": app_name,
                                    "folder": folder, "fork_url": fork_url,
                                    "results": retry_results, "creds": {platform: new_creds},
                                    "env_vars": env_vars,
                                })
                                st.error("Still failed — state saved, close browser and resume later.")
                                st.rerun()
                            else:
                                st.session_state.deploy_results[platform] = url
                                all_ok = all(
                                    not str(r).startswith("FAILED")
                                    for r in st.session_state.deploy_results.values()
                                )
                                _clear_deploy_state_ui()
                                st.session_state.stage = "deployed" if all_ok else "deploy_failed"
                                st.rerun()
                        except Exception as e:
                            st.error(f"Retry error: {e}")
            with col2:
                if st.button(f"Skip {platform.upper()}", key=f"skip_{platform}"):
                    st.session_state.deploy_results[platform] = "SKIPPED"
                    remaining_failed = [
                        p for p, r in st.session_state.deploy_results.items()
                        if str(r).startswith("FAILED")
                    ]
                    if remaining_failed:
                        st.session_state.stage = "deploy_failed"
                    else:
                        _clear_deploy_state_ui()
                        st.session_state.stage = "deployed"
                    st.rerun()
        else:
            st.success(f"✅ **{platform.upper()}** → {result}")

    st.divider()
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        if st.button("Skip All & View Summary"):
            _clear_deploy_state_ui()
            st.session_state.stage = "deployed"
            st.rerun()
    with col_f2:
        if st.button("Try a Different Platform"):
            _clear_deploy_state_ui()
            st.session_state.stage = "local_done"
            st.rerun()

# ── Resume Deploy — triggered from sidebar ────────────────────────
if st.session_state.stage == "resume_deploy":
    data     = st.session_state.get("resume_deploy_data") or _load_deploy_state_ui() or {}
    folder   = data.get("folder",   st.session_state.get("folder", ""))
    fork_url = data.get("fork_url", st.session_state.get("fork_url", ""))
    app_name = data.get("app_name", "")
    env_vars = data.get("env_vars", {})
    targets  = data.get("targets",  [])
    results  = data.get("results",  {})

    failed    = [p for p, r in results.items()
                 if str(r).startswith("FAILED") or r in ("PENDING", "IN_PROGRESS")]
    succeeded = [p for p, r in results.items()
                 if not str(r).startswith("FAILED") and r not in ("PENDING", "IN_PROGRESS", "")]

    st.subheader("Resume Deployment")

    if succeeded:
        st.success(f"Already deployed: {', '.join(p.upper() for p in succeeded)}")
        for p in succeeded:
            st.write(f"✅ **{p.upper()}** → {results[p]}")
        st.divider()

    st.warning(f"Retrying: {', '.join(p.upper() for p in failed)}")

    for platform in failed:
        err = results.get(platform, "")
        if str(err).startswith("FAILED"):
            err_msg = err.replace("FAILED: ", "")
            st.error(f"❌ {platform.upper()} — previous error: `{err_msg[:200]}`")
            if _is_credential_error_ui(err_msg):
                st.warning(f"🔑 {CREDENTIAL_HINTS.get(platform, '')}")
            elif _is_quota_error_ui(err_msg):
                st.warning(f"💳 {QUOTA_HINTS.get(platform, '')}")

        st.markdown(f"**Fresh credentials for {platform.upper()}:**")
        new_creds = {}

        if platform == "aws":
            new_creds = {
                "access_key": st.text_input("AWS_ACCESS_KEY_ID",     key="res_aws_ak"),
                "secret_key": st.text_input("AWS_SECRET_ACCESS_KEY", key="res_aws_sk", type="password"),
                "region":     st.text_input("AWS_REGION",            key="res_aws_rg", value="us-east-1"),
                "app_name": app_name, "folder": folder, "fork_url": fork_url, "env_vars": env_vars,
            }
        elif platform == "azure":
            new_creds = {
                "client_id":       st.text_input("AZURE_CLIENT_ID",       key="res_az_cid"),
                "client_secret":   st.text_input("AZURE_CLIENT_SECRET",   key="res_az_cs",  type="password"),
                "tenant_id":       st.text_input("AZURE_TENANT_ID",       key="res_az_tid"),
                "subscription_id": st.text_input("AZURE_SUBSCRIPTION_ID", key="res_az_sub"),
                "resource_group":  st.text_input("AZURE_RESOURCE_GROUP",  key="res_az_rg"),
                "dockerhub_user":  st.text_input("Docker Hub Username",   key="res_az_dhu"),
                "dockerhub_pass":  st.text_input("Docker Hub Password",   key="res_az_dhp", type="password"),
                "app_name": app_name, "folder": folder, "fork_url": fork_url, "env_vars": env_vars,
            }
        elif platform == "render":
            new_creds = {
                "api_key":  st.text_input("RENDER_API_KEY", key="res_render_key", type="password"),
                "app_name": app_name, "folder": folder, "fork_url": fork_url, "env_vars": env_vars,
            }
        elif platform == "railway":
            new_creds = {
                "token":          st.text_input("RAILWAY_TOKEN",       key="res_rw_tok", type="password"),
                "dockerhub_user": st.text_input("Docker Hub Username", key="res_rw_dhu"),
                "dockerhub_pass": st.text_input("Docker Hub Password", key="res_rw_dhp", type="password"),
                "app_name": app_name, "folder": folder, "fork_url": fork_url, "env_vars": env_vars,
            }

        if st.button(f"Retry {platform.upper()}", key=f"res_retry_{platform}"):
            with st.spinner(f"Deploying {platform.upper()}..."):
                try:
                    retry_res = agent.deploy_to_platforms(
                        [platform], folder, fork_url, {platform: new_creds}
                    )
                    url = retry_res.get(platform, "")
                    results[platform] = url
                    if str(url).startswith("FAILED"):
                        st.error(f"Still failed: {url}")
                        _save_deploy_state_ui({
                            "targets": targets, "app_name": app_name, "folder": folder,
                            "fork_url": fork_url, "results": results,
                            "creds": {platform: new_creds}, "env_vars": env_vars,
                        })
                        st.session_state.resume_deploy_data = {**data, "results": results}
                        st.rerun()
                    else:
                        log(f"Resumed {platform}: {url}")
                        st.session_state.deploy_results = results
                        all_ok = all(
                            not str(r).startswith("FAILED") and r not in ("PENDING", "IN_PROGRESS")
                            for r in results.values()
                        )
                        if all_ok:
                            _clear_deploy_state_ui()
                            st.session_state.stage = "deployed"
                        else:
                            _save_deploy_state_ui({
                                "targets": targets, "app_name": app_name, "folder": folder,
                                "fork_url": fork_url, "results": results,
                                "creds": {platform: new_creds}, "env_vars": env_vars,
                            })
                            st.session_state.resume_deploy_data = {**data, "results": results}
                            st.session_state.stage = "resume_deploy"
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()

    st.markdown("**Want to also deploy to another platform?**")
    extra_targets = st.multiselect(
        "Add more platforms",
        [p for p in ["render", "railway", "aws", "azure"] if p not in targets],
        key="resume_extra_targets"
    )
    if extra_targets:
        st.info(f"After finishing above, you can deploy to: {extra_targets}. Go to Summary first, then use Start Over → Deploy to add them.")

    col_done1, col_done2 = st.columns(2)
    with col_done1:
        if st.button("Go to Summary"):
            _clear_deploy_state_ui()
            st.session_state.deploy_results = results
            st.session_state.stage = "deployed"
            st.rerun()
    with col_done2:
        if st.button("Deploy to a Different Platform Instead"):
            _clear_deploy_state_ui()
            st.session_state.deploy_results = results
            st.session_state.stage = "local_done"
            st.rerun()


# ══════════════════════════════════════════════════════════════════
# DONE — NO DEPLOY
# ══════════════════════════════════════════════════════════════════
if st.session_state.stage == "done_no_deploy":
    st.success("Pipeline complete! Deployment skipped.")
    if st.button("Start Over"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ══════════════════════════════════════════════════════════════════
# DONE — DEPLOYED
# ══════════════════════════════════════════════════════════════════
if st.session_state.stage == "deployed":
    st.success("Deployment complete!")

    results = st.session_state.get("deploy_results", {})
    if not results:
        st.info("No deployment results to show.")
    else:
        for platform, url in results.items():
            if str(url).startswith("FAILED"):
                st.error(f"❌ **{platform.upper()}**: {url}")
            elif url == "SKIPPED":
                st.warning(f"⏭️ **{platform.upper()}**: Skipped")
            elif str(url).startswith("http"):
                st.success(f"✅ **{platform.upper()}**: [{url}]({url})")
            else:
                st.write(f"✅ **{platform.upper()}**: {url}")

    st.divider()
    st.markdown("**Want to deploy to another platform?**")
    if st.button("Deploy to Another Platform"):
        st.session_state.stage = "local_done"
        st.rerun()

    if st.button("Start Over"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
