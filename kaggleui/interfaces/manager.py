"""Optional front-ends per deployment: Gradio (in-process proxy), Open WebUI, ComfyUI and OvertChat."""

from __future__ import annotations

import asyncio
import re
import secrets
import sys
from typing import TYPE_CHECKING

import httpx
import uvicorn

from ..engines.context import DeployError, EngineContext
from ..engines.install import ensure_venv, pip_install
from ..proc import ManagedProcess, run_cmd
from ..util import free_port

if TYPE_CHECKING:
    from ..deploy.manager import Deployment
    from ..server.state import AppState

SHARE_RE = re.compile(r"https://[a-z0-9-]+\.gradio\.live")

# OvertChat keeps its model list in SQLite (no env-var provider config), so the deployment is
# registered by inserting a row after migrations have run. Columns are introspected so a schema
# change degrades to the manual Settings → Models path instead of writing a bad row.
OVERTCHAT_SEED_JS = """\
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const Database = require("better-sqlite3");
const [dbPath, baseUrl, apiKey, model] = process.argv.slice(2);
const db = new Database(dbPath);
const columns = new Set(db.prepare("PRAGMA table_info(model_configs)").all().map((c) => c.name));
if (!columns.has("id")) {
  console.error("model_configs table missing");
  process.exit(2);
}
const now = Date.now();
const row = {
  id: "kaggleui", label: model, model_type: "chat", provider_id: "custom", api_format: "openai-chat",
  base_url: baseUrl, api_key: apiKey, model, tool_calling_enabled: 1, enabled: 1, task_model: 0,
  credential_scope: "shared", sort_order: 0, created_at: now, updated_at: now,
};
const use = Object.keys(row).filter((k) => columns.has(k));
const updates = use.filter((k) => k !== "id").map((k) => `${k}=excluded.${k}`).join(", ");
db.prepare(
  `INSERT INTO model_configs (${use.join(",")}) VALUES (${use.map(() => "?").join(",")})
   ON CONFLICT(id) DO UPDATE SET ${updates}`,
).run(...use.map((k) => row[k]));
db.close();
"""


class InterfaceManager:
    def __init__(self, state: AppState):
        self.state = state
        self._servers: dict[str, tuple[uvicorn.Server, asyncio.Task]] = {}

    async def start(self, dep: Deployment, ctx: EngineContext) -> None:
        kind = dep.spec.interface
        dep.interface = {"kind": kind, "status": "starting", "url": None, "error": None}
        self.state.bus.publish("deployment", dep.public())
        try:
            if kind == "gradio":
                await self._gradio(dep, ctx)
            elif kind == "openwebui":
                await self._openwebui(dep, ctx)
            elif kind == "comfyui":
                await self._comfyui(dep, ctx)
            elif kind == "overtchat":
                await self._overtchat(dep, ctx)
            dep.interface["status"] = "up"
        except DeployError as e:
            dep.interface.update(status="error", error=str(e))
            if kind == "comfyui":
                raise
        except Exception as e:  # noqa: BLE001
            dep.interface.update(status="error", error=f"{type(e).__name__}: {e}")
            if kind == "comfyui":
                raise
        self.state.bus.publish("deployment", dep.public())

    async def stop(self, dep: Deployment) -> None:
        srv = self._servers.pop(dep.id, None)
        if srv:
            srv[0].should_exit = True
            try:
                await asyncio.wait_for(srv[1], timeout=5)
            except Exception:
                srv[1].cancel()
        await self.state.tunnels.close(f"ui-{dep.id}")
        if dep.interface:
            dep.interface["status"] = "stopped"

    async def _wait_http(self, url: str, proc: ManagedProcess, label: str, timeout: int = 900) -> None:
        async with httpx.AsyncClient(timeout=5) as c:
            for _ in range(timeout):
                if not proc.running:
                    raise DeployError(f"{label} exited: " + " | ".join(list(proc.tail)[-4:]))
                try:
                    r = await c.get(url)
                    if r.status_code < 500:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1)
        raise DeployError(f"{label} didn't start in time")

    async def _serve_proxy(self, dep: Deployment, target_port: int, title: str) -> str:
        """Run an auth proxy for ``target_port`` on a new port and open a tunnel to it."""
        from ..server.proxy import make_root_proxy

        port = free_port()
        app = make_root_proxy(self.state, target_port, title)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
        task = asyncio.create_task(server.serve())
        self._servers[dep.id] = (server, task)
        provider = self.state.settings.exposure.provider
        tunnel = await self.state.tunnels.open(f"ui-{dep.id}", port, provider if provider != "cloudflare_named" else "cloudflare")
        url = await self.state.tunnels.wait_url(tunnel.name, timeout=90)
        return url or f"http://127.0.0.1:{port}"

    # -- Gradio -------------------------------------------------------------------------------
    async def _gradio(self, dep: Deployment, ctx: EngineContext) -> None:
        await pip_install(ctx, ["gradio>=5"], label="Gradio")
        port = free_port()
        root = f"/gradio/{dep.id}"
        cmd = [sys.executable, "-m", "kaggleui.interfaces.gradio_app", "--modality", dep.spec.modality,
               "--model", dep.served_name, "--api", f"http://127.0.0.1:{self.state.port}", "--port", str(port),
               "--root-path", root]
        share = bool(dep.spec.options.get("gradio_share"))
        if share:
            cmd.append("--share")

        def on_line(line: str) -> None:
            m = SHARE_RE.search(line)
            if m:
                dep.interface["share_url"] = m.group(0)
                self.state.bus.publish("deployment", dep.public())
            ctx.log(f"[gradio] {line}")

        env = {"KAGGLEUI_INTERNAL_KEY": self.state.auth.internal_key, "GRADIO_ANALYTICS_ENABLED": "False"}
        if share and self.state.settings.access.password_enabled:
            env["KAGGLEUI_GRADIO_PASSWORD"] = self.state.auth.password
        proc = ManagedProcess(f"gradio-{dep.id}", cmd, env=env, log_path=ctx.paths.logs / f"{dep.id}-gradio.log",
                              on_line=on_line)
        dep.processes.append(proc)
        await proc.start()
        await self._wait_http(f"http://127.0.0.1:{port}/", proc, "Gradio", timeout=180)
        dep.interface.update(local_port=port, path=f"{root}/")
        main = self.state.tunnels.get("main")
        dep.interface["url"] = (main.url.rstrip("/") if main and main.url else f"http://127.0.0.1:{self.state.port}") + f"{root}/"

    # -- Open WebUI ---------------------------------------------------------------------------
    async def _openwebui(self, dep: Deployment, ctx: EngineContext) -> None:
        if dep.spec.modality != "text":
            raise DeployError("Open WebUI is for text models")
        py = await ensure_venv(ctx, "openwebui", ["open-webui"], python="3.11", label="Open WebUI")
        exe = str(ctx.paths.envs / "openwebui" / "bin" / "open-webui")
        port = free_port()
        data_dir = ctx.paths.home / "openwebui"
        data_dir.mkdir(parents=True, exist_ok=True)
        env = {
            "OPENAI_API_BASE_URL": f"http://127.0.0.1:{self.state.port}/v1",
            "OPENAI_API_KEY": self.state.auth.internal_key,
            "ENABLE_OLLAMA_API": "False",
            "WEBUI_AUTH": "False",
            "DATA_DIR": str(data_dir),
            "WEBUI_NAME": "KaggleUI · Open WebUI",
            "ENABLE_EVALUATION_ARENA_MODELS": "False",
            "HF_HOME": str(ctx.paths.hf_home),
        }
        ctx.log(f"Starting Open WebUI with {py}")
        proc = ManagedProcess(f"openwebui-{dep.id}", [exe, "serve", "--host", "127.0.0.1", "--port", str(port)],
                              env=env, log_path=ctx.paths.logs / f"{dep.id}-openwebui.log",
                              on_line=lambda line: ctx.log(f"[open-webui] {line}"))
        dep.processes.append(proc)
        dep.interface["message"] = "Starting Open WebUI (first start takes a few minutes)"
        await proc.start()
        await self._wait_http(f"http://127.0.0.1:{port}/health", proc, "Open WebUI")
        dep.interface["url"] = await self._serve_proxy(dep, port, "Open WebUI")
        dep.interface["local_port"] = port

    # -- OvertChat ----------------------------------------------------------------------------
    @staticmethod
    async def _node_version() -> tuple[int, int] | None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "node", "--version", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
            out, _ = await proc.communicate()
        except OSError:
            return None
        match = re.match(r"v(\d+)\.(\d+)", out.decode("utf-8", "replace").strip())
        return (int(match.group(1)), int(match.group(2))) if match else None

    async def _overtchat(self, dep: Deployment, ctx: EngineContext) -> None:
        if dep.spec.modality != "text":
            raise DeployError("OvertChat is for text models")
        version = await self._node_version()
        if version is None:
            raise DeployError("OvertChat needs Node.js 20.9+; node was not found on PATH")
        if version < (20, 9):
            raise DeployError(f"OvertChat needs Node.js 20.9+ (this host has {version[0]}.{version[1]})")
        src = ctx.paths.envs / "src" / "overtchat"
        build_env = {**ctx.env(), "CI": "1", "NEXT_TELEMETRY_DISABLED": "1", "DISABLE_UPDATE_CHECK": "1"}
        if not (src / "package.json").exists():
            src.parent.mkdir(parents=True, exist_ok=True)
            ctx.progress("downloading", None, "Cloning OvertChat")
            code = await run_cmd(["git", "clone", "--depth", "1", "https://github.com/yoloyash/overtchat",
                                  str(src)], env=build_env, on_line=ctx.log)
            if code != 0:
                raise DeployError(f"Couldn't clone OvertChat (git exited {code}); see the deployment log")
        if not (src / ".kui-build").exists():
            ctx.progress("downloading", None, "Installing OvertChat dependencies (npm ci)")
            code = await run_cmd(["npm", "ci", "--no-audit", "--no-fund"], env=build_env, cwd=src,
                                 on_line=ctx.log)
            if code != 0:
                raise DeployError("OvertChat dependency install failed; see the deployment log")
            ctx.progress("downloading", None, "Building OvertChat (next build)")
            code = await run_cmd(["npm", "run", "build", "-w", "@overtchat/web"], env=build_env, cwd=src,
                                 on_line=ctx.log)
            if code != 0:
                raise DeployError("OvertChat build failed; see the deployment log")
            (src / ".kui-build").write_text("built\n", encoding="utf-8")
        data = ctx.paths.home / "overtchat"
        data.mkdir(parents=True, exist_ok=True)
        secret_file = data / "auth_secret"
        if not secret_file.exists():
            secret_file.write_text(secrets.token_hex(32), encoding="utf-8")
            secret_file.chmod(0o600)
        port = free_port()
        base_url = f"http://127.0.0.1:{self.state.port}/v1"
        env = {
            **ctx.env(),
            "DATABASE_URL": str(data / "chat.db"),
            "BETTER_AUTH_URL": f"http://127.0.0.1:{port}",
            "BETTER_AUTH_SECRET": secret_file.read_text(encoding="utf-8").strip(),
            "DISABLE_UPDATE_CHECK": "1",
            "NEXT_TELEMETRY_DISABLED": "1",
        }
        next_bin = src / "node_modules" / "next" / "dist" / "bin" / "next"
        if not next_bin.exists():
            next_bin = src / "apps" / "web" / "node_modules" / "next" / "dist" / "bin" / "next"
        if not next_bin.exists():
            raise DeployError("OvertChat installed without a Next.js binary; delete "
                              f"{src} and deploy again")
        proc = ManagedProcess(f"overtchat-{dep.id}", ["node", str(next_bin), "start", "-p", str(port)],
                              env=env, cwd=src / "apps" / "web",
                              log_path=ctx.paths.logs / f"{dep.id}-overtchat.log",
                              on_line=lambda line: ctx.log(f"[overtchat] {line}"))
        dep.processes.append(proc)
        dep.interface["message"] = "Starting OvertChat"
        await proc.start()
        await self._wait_http(f"http://127.0.0.1:{port}/", proc, "OvertChat", timeout=600)
        await self._seed_overtchat(ctx, src, data / "chat.db", base_url, dep.served_name)
        dep.interface["url"] = await self._serve_proxy(dep, port, "OvertChat")
        dep.interface["local_port"] = port

    async def _seed_overtchat(self, ctx: EngineContext, src, db, base_url: str, model: str) -> None:
        """Register the deployment as a Chat Completions model in OvertChat's SQLite database."""
        script = src / "kui-seed.mjs"
        script.write_text(OVERTCHAT_SEED_JS, encoding="utf-8")
        code = await run_cmd(["node", str(script), str(db), base_url, self.state.auth.internal_key, model],
                             env=ctx.env(), cwd=src, on_line=ctx.log)
        if code != 0:
            ctx.log("OvertChat is running, but the model wasn't registered automatically. "
                    f"Add it under Settings → Models: base URL {base_url}, "
                    f"API key {self.state.auth.internal_key}, model {model}")

    # -- ComfyUI ------------------------------------------------------------------------------
    async def _comfyui(self, dep: Deployment, ctx: EngineContext) -> None:
        src = ctx.paths.envs / "src" / "ComfyUI"
        if not (src / "main.py").exists():
            ctx.progress("installing", None, "Cloning ComfyUI")
            rc = await run_cmd(["git", "clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI", str(src)],
                               on_line=ctx.log)
            if rc != 0:
                raise DeployError("Couldn't clone ComfyUI")
            mgr = src / "custom_nodes" / "ComfyUI-Manager"
            await run_cmd(["git", "clone", "--depth", "1", "https://github.com/Comfy-Org/ComfyUI-Manager", str(mgr)],
                          on_line=ctx.log)
        reqs = [r.strip() for r in (src / "requirements.txt").read_text().splitlines()
                if r.strip() and not r.startswith("#") and not re.match(r"^(torch|torchvision|torchaudio)\b", r.strip())]
        await pip_install(ctx, reqs, label="ComfyUI")

        # If the repo ships a single-file checkpoint, put it where ComfyUI looks for it.
        if dep.spec.model.source == "hf":
            from ..engines.download import hf_download, repo_files

            sib = await repo_files(dep.spec.model.id)
            singles = [s for s in sib if "/" not in s["rfilename"] and s["rfilename"].endswith(".safetensors")]
            if singles:
                pick = min(singles, key=lambda s: (".fp16." not in s["rfilename"] and "fp16" not in s["rfilename"],
                                                   s.get("size") or 0))
                path = await hf_download(ctx, dep.spec.model.id, [pick["rfilename"]], pick.get("size"))
                dest = src / "models" / "checkpoints" / pick["rfilename"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    dest.symlink_to(f"{path}/{pick['rfilename']}")
            else:
                ctx.log("This repo has no single-file checkpoint; download models from ComfyUI Manager.")

        port = free_port()
        args = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(port), "--preview-method", "auto"]
        if (ctx.hw.min_sm or 0) < 80:
            args.append("--force-fp16")
        proc = ManagedProcess(f"comfyui-{dep.id}", args, env=ctx.env(), log_path=ctx.paths.logs / f"{dep.id}-comfyui.log",
                              on_line=lambda line: ctx.log(f"[comfyui] {line}"))
        proc.cwd = str(src)
        dep.processes.append(proc)
        ctx.progress("loading", None, "Starting ComfyUI")
        await proc.start()
        await self._wait_http(f"http://127.0.0.1:{port}/", proc, "ComfyUI", timeout=600)
        dep.interface["url"] = await self._serve_proxy(dep, port, "ComfyUI")
        dep.interface["local_port"] = port
