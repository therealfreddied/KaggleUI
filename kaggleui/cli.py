"""Command line: ``kaggleui [serve|launch|selftest|bench|stop]``."""

from __future__ import annotations

import argparse
import json
import sys

from .config import DEFAULT_PORT, load_preset


def _common(p: argparse.ArgumentParser, *, host: bool = True) -> None:
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--tunnel", choices=["cloudflare", "cloudflare_named", "ngrok", "none"], default=None,
                   help="How to expose the panel (default: cloudflare on Kaggle/Colab, none elsewhere)")
    p.add_argument("--preset", default=None, help="Preset JSON string or file to auto-deploy")
    p.add_argument("--mock-engines", action="store_true", help="Simulate engines (UI development without a GPU)")
    p.add_argument("--fake-hw", choices=["t4x2", "p100", "tpu", "cpu"], default=None,
                   help="Pretend to run on this Kaggle hardware")
    p.add_argument("--home", default=None, help="Working directory for models, envs and logs (default /tmp/kaggleui)")
    if host:
        p.add_argument("--host", default="127.0.0.1")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kaggleui", description="Host AI models on Kaggle's free GPUs.")
    sub = parser.add_subparsers(dest="cmd")
    _common(sub.add_parser("serve", help="Run the server in the foreground"))
    lp = sub.add_parser("launch", help="Start in the background and print the link (notebook-friendly)")
    _common(lp)
    lp.add_argument("--no-block", action="store_true")
    st = sub.add_parser("selftest", help="Deploy the smallest model of each type and exercise every API")
    _common(st, host=False)
    st.add_argument("--modalities", default="text,image,video,tts,stt,world")
    st.add_argument("--out", default="selftest.json")
    st.add_argument("--host", action="store_true", help="Print the detected host and exit")
    bp = sub.add_parser("bench", help="Measure TTFT and tokens/s of a serving deployment")
    _common(bp)
    bp.add_argument("--model", required=True, help="Served name, deployment id, or model id")
    bp.add_argument("--out", default="bench.json")
    bp.add_argument("--ctx", default="2048,4096", help="Context sizes to compare")
    bp.add_argument("--prompt-tokens", default="128,512", help="Prompt lengths to compare")
    bp.add_argument("--runs", type=int, default=3)
    sp = sub.add_parser("stop", help="Stop a background KaggleUI")
    sp.add_argument("--home", default=None)

    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0].startswith("-"):
        argv = ["serve", *argv]
    args = parser.parse_args(argv)

    if args.cmd == "stop":
        from .launcher import stop

        stop(args.home)
        return
    if args.cmd == "launch":
        from .launcher import launch

        launch(args.preset, block=not args.no_block, tunnel=args.tunnel, port=args.port, mock=args.mock_engines,
               fake_hw=args.fake_hw, home=args.home)
        return
    if args.cmd == "selftest" and args.host:
        from dataclasses import asdict

        from .hosts import detect

        host = detect()
        print(json.dumps({"host": host.name, **asdict(host.capabilities()), **asdict(host.session_info()),
                          "guidance": host.startup_guidance()}))
        return 0
    if args.cmd == "selftest":
        from .selftest import run

        sys.exit(run(args))
    if args.cmd == "bench":
        import asyncio

        from .server.run import build_state
        from .workbench.bench import bench, format_table, save

        state = build_state(port=args.port, tunnel=args.tunnel, mock=args.mock_engines, fake_hw=args.fake_hw,
                            home=args.home)
        ctxs = tuple(int(x) for x in args.ctx.split(",") if x.strip())
        sizes = tuple(int(x) for x in args.prompt_tokens.split(",") if x.strip())
        report = asyncio.run(bench(state, args.model, ctxs=ctxs, prompt_tokens=sizes, runs=args.runs))
        written = save(state, report, out=args.out)
        print(format_table(report), flush=True)
        for label, path in written.items():
            print(f"{label}: {path}", flush=True)
        return

    from .server.run import build_state, serve

    state = build_state(port=args.port, tunnel=args.tunnel, preset=load_preset(args.preset), mock=args.mock_engines,
                        fake_hw=args.fake_hw, home=args.home)
    serve(state, host=args.host)


if __name__ == "__main__":
    main()
