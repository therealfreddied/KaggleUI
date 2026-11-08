import { useEffect, useRef, useState } from "react";
import { FitAddon } from "@xterm/addon-fit";
import { Terminal } from "@xterm/xterm";
import { RotateCw } from "lucide-react";
import "@xterm/xterm/css/xterm.css";

import { Button } from "../components/ui";

type State = "connecting" | "open" | "closed";

export default function Console() {
  const host = useRef<HTMLDivElement | null>(null);
  const [state, setState] = useState<State>("connecting");
  const [detail, setDetail] = useState<string | null>(null);
  const [generation, setGeneration] = useState(0);

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const style = getComputedStyle(document.documentElement);
    const token = (name: string, fallback: string) => style.getPropertyValue(name).trim() || fallback;
    const colors = {
      bg: token("--bg-2", "#0b0b0a"),
      fg: token("--fg", "#f2f1ed"),
      accent: token("--good", "#4ea36a"),
    };
    const term = new Terminal({
      convertEol: false,
      cursorBlink: true,
      fontFamily: '"Geist Mono Variable", ui-monospace, monospace',
      fontSize: 13,
      scrollback: 5000,
      theme: { background: colors.bg, foreground: colors.fg, cursor: colors.fg, selectionBackground: colors.accent },
    });
    const fit = new FitAddon();
    term.loadAddon(fit);
    term.open(el);
    fit.fit();
    setState("connecting");
    setDetail(null);

    const proto = location.protocol === "https:" ? "wss" : "ws";
    const sock = new WebSocket(`${proto}://${location.host}/_kui/ws/console`);
    sock.binaryType = "arraybuffer";
    const decoder = new TextDecoder();

    sock.onopen = () => {
      setState("open");
      sock.send(JSON.stringify({ op: "resize", cols: term.cols, rows: term.rows }));
      term.focus();
    };
    sock.onmessage = (event) => {
      if (typeof event.data === "string") {
        try {
          const msg = JSON.parse(event.data) as { op?: string; code?: number | null };
          if (msg.op === "exit") {
            setState("closed");
            setDetail(`Shell exited${msg.code === null || msg.code === undefined ? "" : ` with code ${msg.code}`}.`);
          }
        } catch {
          setDetail(event.data);
        }
        return;
      }
      term.write(decoder.decode(new Uint8Array(event.data as ArrayBuffer), { stream: true }));
    };
    sock.onclose = (event) => {
      setState("closed");
      if (event.reason) setDetail(event.reason);
      else setDetail((current) => current ?? "The terminal disconnected.");
    };

    const input = term.onData((data) => {
      if (sock.readyState === WebSocket.OPEN) sock.send(JSON.stringify({ op: "input", data }));
    });
    const resize = term.onResize(({ cols, rows }) => {
      if (sock.readyState === WebSocket.OPEN) sock.send(JSON.stringify({ op: "resize", cols, rows }));
    });
    const observer = new ResizeObserver(() => fit.fit());
    observer.observe(el);

    return () => {
      observer.disconnect();
      input.dispose();
      resize.dispose();
      sock.close();
      term.dispose();
    };
  }, [generation]);

  return (
    <div className="flex min-h-0 flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-[22px] font-semibold tracking-tight">Console</h1>
          <p className="mt-0.5 text-[13px] text-fg-3">
            A shell in the server process, running in <code className="font-mono">{`KAGGLEUI_HOME`}</code>.
            It ends when you close this page.
          </p>
        </div>
        <Button onClick={() => setGeneration((n) => n + 1)}>
          <RotateCw className="size-4" /> New session
        </Button>
      </div>
      <div className="panel min-h-0 flex-1 overflow-hidden p-2">
        <div ref={host} className="h-[62vh] w-full" />
      </div>
      {state !== "open" && (
        <p className="text-[13px] text-fg-2">
          {state === "connecting" ? "Connecting…" : detail ?? "Disconnected."}
        </p>
      )}
    </div>
  );
}