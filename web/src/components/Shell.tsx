import { GitBranch, Boxes, CornerDownLeft, Download, Gauge, HardDrive, KeyRound, LayoutGrid, ListChecks, LogOut, Menu, MessageSquare, Moon, Plus, Search, Server, Sun, Wand2, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { Link, NavLink, useLocation, useNavigate } from "react-router";

import { api } from "../lib/api";
import { bytes, cx, duration, memoryPool } from "../lib/format";
import { useLive } from "../lib/live";
import { useTheme } from "../lib/theme";
import { CommunityLinks } from "./CommunityLinks";
import { UpdateNotice } from "./Updates";
import { Button } from "./ui";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutGrid, end: true },
  { to: "/models", label: "Models", icon: Boxes },
  { to: "/imports", label: "Imports", icon: Download },
  { to: "/storage", label: "Storage", icon: HardDrive },
  { to: "/workspace", label: "Workspace", icon: GitBranch },
  { to: "/deployments", label: "Deployments", icon: Server },
  { to: "/studio", label: "Studio", icon: MessageSquare },
  { to: "/access", label: "Access & API", icon: KeyRound },
  { to: "/create", label: "Model Studio", icon: Wand2 },
  { to: "/jobs", label: "Jobs", icon: ListChecks },
  { to: "/system", label: "System", icon: Gauge },
];

/** Wordmark: three level bars, the shortest lit, like a meter at rest. */
export function Brand({ small }: { small?: boolean }) {
  return (
    <div className="flex items-center gap-2.5">
      <svg viewBox="0 0 28 28" className={small ? "size-7" : "size-8"} aria-hidden>
        <rect width="28" height="28" rx="8" fill="var(--fg)" />
        <rect x="7" y="8" width="3" height="12" rx="1.5" fill="var(--bg)" />
        <rect x="12.5" y="12" width="3" height="8" rx="1.5" fill="var(--bg)" opacity=".7" />
        <rect x="18" y="16" width="3" height="4" rx="1.5" fill="var(--good)" />
      </svg>
      <span className="text-[16px] font-semibold tracking-[-0.03em]">KaggleUI</span>
    </div>
  );
}

function QuickJump({ open, close }: { open: boolean; close: () => void }) {
  const { overview } = useLive();
  const [query, setQuery] = useState("");
  const [index, setIndex] = useState(0);
  const ref = useRef<HTMLDialogElement>(null);
  const nav = useNavigate();
  useEffect(() => {
    if (open) {
      setQuery("");
      setIndex(0);
      ref.current?.showModal();
    } else ref.current?.close();
  }, [open]);
  const options = [
    ...NAV.map((n) => ({ ...n, group: "Go to" })),
    { to: "/deploy/new", label: "New deployment", icon: Plus, group: "Do" },
    ...(overview?.deployments
      .filter((d) => d.status === "ready")
      .map((d) => ({ to: `/studio/${d.id}`, label: `Open ${d.name}`, icon: MessageSquare, group: "Running models" })) ?? []),
  ].filter((option) => option.label.toLowerCase().includes(query.toLowerCase()));
  const go = (to: string) => {
    nav(to);
    close();
  };
  return (
    <dialog
      aria-label="Quick navigation"
      ref={ref}
      onClose={close}
      onClick={(e) => {
        if (e.target === e.currentTarget) close();
      }}
      className="command-dialog"
    >
      <form
        method="dialog"
        onSubmit={(e) => {
          e.preventDefault();
          if (options[index]) go(options[index].to);
        }}
      >
        <div className="flex items-center gap-3 border-b border-line px-4">
          <Search className="size-4 shrink-0 text-fg-3" />
          <input
            autoFocus
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setIndex(0);
            }}
            onKeyDown={(e) => {
              if (e.key === "ArrowDown") {
                e.preventDefault();
                setIndex((i) => Math.min(i + 1, options.length - 1));
              } else if (e.key === "ArrowUp") {
                e.preventDefault();
                setIndex((i) => Math.max(i - 1, 0));
              }
            }}
            placeholder="Jump to a page or a running model"
            aria-label="Quick navigation"
            className="h-14 min-w-0 flex-1 bg-transparent text-[16px] text-fg placeholder:text-fg-3 focus:outline-none focus-visible:outline-none sm:text-[15px]"
          />
          <span className="kbd">esc</span>
        </div>
        <nav className="max-h-80 overflow-y-auto p-1.5 scroll-thin">
          {options.map(({ to, label, icon: Icon }, i) => (
            <Link
              key={to + label}
              to={to}
              onClick={close}
              onMouseEnter={() => setIndex(i)}
              className={cx(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-[13.5px] transition-colors",
                i === index ? "bg-panel-3 text-fg" : "text-fg-2",
              )}
            >
              <Icon className="size-4 text-fg-3" />
              <span className="flex-1">{label}</span>
              {i === index && <CornerDownLeft className="size-3.5 text-fg-3" />}
            </Link>
          ))}
          {!options.length && <p className="px-3 py-6 text-center text-[13px] text-fg-3">Nothing matches “{query}”.</p>}
        </nav>
      </form>
    </dialog>
  );
}

function MachineChip() {
  const { overview, connected, error } = useLive();
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(t);
  }, []);
  const system = overview?.system;
  if (!system) return null;
  const pool = memoryPool(system);
  const remaining = Math.max(0, system.session_started + system.session_limit - now / 1000);
  const low = remaining < system.session_limit * 0.1;
  const label = connected ? "Live" : error ? "Connection interrupted" : "Reconnecting";
  return (
    <Link to="/system" className="machine-chip" title={`${label} · ${system.accelerator_label} · session time left`}>
      <span className={cx("dot", connected ? "dot-live" : "dot-warn")} aria-label={label} />
      <b className="hide-xs">{system.accelerator_label.replace("GPU ", "")}</b>
      <span className="sep hide-sm" />
      <span className="hide-sm">{bytes(pool.free)} free</span>
      <span className="sep hide-xs" />
      <span className={cx(low && "text-worst")}>{duration(remaining)}</span>
    </Link>
  );
}

function MobileNav({ open, close, ready, busy }: { open: boolean; close: () => void; ready: number; busy: number }) {
  const panelRef = useRef<HTMLElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    closeRef.current?.focus();
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") close();
      if (event.key !== "Tab" || !panelRef.current) return;
      const items = [...panelRef.current.querySelectorAll<HTMLElement>('a[href], button:not([disabled])')];
      if (!items.length) return;
      const first = items[0];
      const last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = previous;
      requestAnimationFrame(() => document.querySelector<HTMLButtonElement>(".mobile-menu-button")?.focus());
    };
  }, [open, close]);
  if (!open) return null;
  return createPortal(
    <div className="mobile-nav">
      <button className="mobile-nav-scrim" aria-label="Close navigation" onClick={close} />
      <aside ref={panelRef} id="mobile-nav" className="mobile-nav-panel" role="dialog" aria-modal="true" aria-label="Workspace">
        <div className="flex h-16 shrink-0 items-center justify-between gap-3 border-b border-line px-4">
          <Brand small />
          <button ref={closeRef} className="toolbar-icon" aria-label="Close navigation" onClick={close}>
            <X className="size-4" />
          </button>
        </div>
        <nav className="min-h-0 flex-1 overflow-y-auto scroll-thin p-2" aria-label="Workspace">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end} onClick={close} className={({ isActive }) => cx("mobile-nav-link", isActive && "is-active")}>
              <Icon className="size-[17px] shrink-0" />
              <span className="min-w-0 flex-1 truncate">{label}</span>
              {to === "/deployments" && (ready > 0 || busy > 0) && (
                <span className="nav-count" title={`${ready} ready${busy ? `, ${busy} starting` : ""}`}>
                  {busy ? `${ready}+${busy}` : ready}
                </span>
              )}
            </NavLink>
          ))}
        </nav>
        <div className="shrink-0 border-t border-line p-3 pb-[max(12px,env(safe-area-inset-bottom))]">
          <Link to="/deploy/new" onClick={close} className="block">
            <Button variant="primary" className="w-full">
              <Plus className="size-4" /> Deploy a model
            </Button>
          </Link>
        </div>
      </aside>
    </div>,
    document.body,
  );
}

export function Shell({ children }: { children: ReactNode }) {
  const [theme, toggle] = useTheme();
  const [jump, setJump] = useState(false);
  const [navOpen, setNavOpen] = useState(false);
  const closeNav = useCallback(() => setNavOpen(false), []);
  const { overview } = useLive();
  const location = useLocation();
  useEffect(() => {
    setNavOpen(false);
  }, [location.pathname]);
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setJump((value) => !value);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);
  const ready = overview?.deployments.filter((d) => d.status === "ready").length ?? 0;
  const busy = overview?.deployments.filter((d) => ["queued", "installing", "downloading", "loading"].includes(d.status)).length ?? 0;
  return (
    <div className="flex min-h-dvh flex-col">
      <a href="#main-content" className="skip-link">
        Skip to content
      </a>
      <header className="app-header">
        <div className="app-header-inner">
          <Link to="/" aria-label="KaggleUI overview">
            <Brand small />
          </Link>
          <nav className="app-nav" aria-label="Workspace">
            {NAV.map(({ to, label, end }) => (
              <NavLink key={to} to={to} end={end} className={({ isActive }) => cx("app-link", isActive && "is-active")}>
                {label}
                {to === "/deployments" && (ready > 0 || busy > 0) && (
                  <span className="nav-count" title={`${ready} ready${busy ? `, ${busy} starting` : ""}`}>
                    {busy ? `${ready}+${busy}` : ready}
                  </span>
                )}
              </NavLink>
            ))}
          </nav>
          <div className="app-tools">
            <MachineChip />
            <button onClick={() => setJump(true)} className="quick-jump hidden md:inline-flex" aria-label="Quick navigation (Control or Command K)">
              <Search className="size-3.5" />
              <span className="kbd">⌘K</span>
            </button>
            <button onClick={toggle} className="toolbar-icon" aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}>
              {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
            </button>
            <button onClick={() => api.logout().then(() => window.location.assign("/login"))} className="toolbar-icon hidden sm:inline-grid" aria-label="Sign out">
              <LogOut className="size-4" />
            </button>
            <Link to="/deploy/new" className="ml-1 hidden sm:block">
              <Button size="sm" variant="primary">
                <Plus className="size-3.5" />
                Deploy
              </Button>
            </Link>
            <button
              onClick={() => setNavOpen(true)}
              className="mobile-menu-button toolbar-icon"
              aria-label="Open navigation"
              aria-expanded={navOpen}
              aria-controls="mobile-nav"
            >
              <Menu className="size-5" />
              {(ready > 0 || busy > 0) && <span className="mobile-menu-dot" aria-hidden />}
            </button>
          </div>
        </div>
      </header>
      <main id="main-content" tabIndex={-1} className="app-main flex-1 focus:outline-none">
        <UpdateNotice />
        <div key={location.pathname} className="page-enter">
          {children}
        </div>
      </main>
      <footer className="app-footer">
        <span className="font-mono">KaggleUI {__APP_VERSION__}</span>
        {overview?.mock && <span className="text-tight">Demo engines: nothing runs on a GPU</span>}
        <a href="https://github.com/therealfreddied/KaggleUI" target="_blank" rel="noreferrer">
          Source & help ↗
        </a>
        <button type="button" className="sm:hidden" onClick={() => api.logout().then(() => window.location.assign("/login"))}>
          Sign out
        </button>
        <CommunityLinks />
      </footer>
      <QuickJump open={jump} close={() => setJump(false)} />
      <MobileNav open={navOpen} close={closeNav} ready={ready} busy={busy} />
    </div>
  );
}
