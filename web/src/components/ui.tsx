import { Check, Copy, Eye, EyeOff, LoaderCircle, X } from "lucide-react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ButtonHTMLAttributes,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
  type TextareaHTMLAttributes,
} from "react";
import { createPortal } from "react-dom";

import { cx, VERDICT_LABEL } from "../lib/format";
import type { Verdict } from "../lib/types";

type Variant = "primary" | "ghost" | "outline" | "danger";

export function Button({
  variant = "outline",
  size = "md",
  busy,
  className,
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: "sm" | "md" | "lg"; busy?: boolean }) {
  return (
    <button
      {...rest}
      disabled={rest.disabled || busy}
      className={cx(
        "omp-button inline-flex select-none items-center justify-center gap-2 whitespace-nowrap rounded-lg font-medium",
        "transition-[background,border-color,color,box-shadow,transform,opacity] duration-150 ease-out",
        "disabled:pointer-events-none disabled:opacity-40 active:scale-[0.98]",
        size === "sm" && "h-8 px-3 text-[13px]",
        size === "md" && "h-9 px-3.5 text-[13.5px]",
        size === "lg" && "h-11 px-5 text-[14.5px]",
        variant === "primary" &&
          "bg-signal text-signal-ink shadow-[inset_0_1px_0_rgb(255_255_255/0.25),0_1px_2px_rgb(0_0_0/0.2)] hover:opacity-90",
        variant === "outline" && "border border-line-2 bg-panel text-fg shadow-[var(--edge)] hover:border-fg-3 hover:bg-panel-2",
        variant === "ghost" && "text-fg-2 hover:bg-panel-2 hover:text-fg",
        variant === "danger" && "border border-line-2 text-worst hover:border-worst/50 hover:bg-worst-soft",
        className,
      )}
    >
      {busy ? <LoaderCircle className="size-4 animate-spin" /> : null}
      {children}
    </button>
  );
}

export function Toggle({
  checked,
  onChange,
  label,
  hint,
  disabled,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label?: ReactNode;
  hint?: ReactNode;
  disabled?: boolean;
}) {
  return (
    <label className={cx("flex items-start gap-3 select-none", disabled ? "opacity-50" : "cursor-pointer")}>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={cx(
          "relative mt-px h-[22px] w-[38px] shrink-0 rounded-full transition-colors duration-200",
          checked ? "bg-fg" : "bg-panel-3 shadow-[inset_0_0_0_1px_var(--line-2)]",
        )}
      >
        <span
          className={cx(
            "absolute top-[3px] size-4 rounded-full shadow-sm transition-[left,background] duration-200 ease-out",
            checked ? "left-[19px] bg-bg" : "left-[3px] bg-fg-3",
          )}
        />
      </button>
      {(label || hint) && (
        <span className="min-w-0">
          {label && <span className="block text-[13.5px] font-medium leading-snug">{label}</span>}
          {hint && <span className="mt-0.5 block text-[12.5px] leading-snug text-fg-3">{hint}</span>}
        </span>
      )}
    </label>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  size = "md",
}: {
  value: T;
  options: { value: T; label: ReactNode }[];
  onChange: (v: T) => void;
  size?: "sm" | "md";
}) {
  return (
    <div className="segmented inline-flex max-w-full gap-0.5 overflow-x-auto rounded-[10px] border border-line bg-bg-2 p-[3px] scroll-thin">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          onClick={() => onChange(o.value)}
          aria-pressed={value === o.value}
          className={cx(
            "whitespace-nowrap rounded-[7px] font-medium transition-[background,color,box-shadow] duration-150",
            size === "sm" ? "h-7 px-2.5 text-[12.5px]" : "h-8 px-3.5 text-[13px]",
            value === o.value
              ? "bg-panel-3 text-fg shadow-[var(--edge),0_1px_2px_rgb(0_0_0/0.15)]"
              : "text-fg-3 hover:text-fg",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function useCopy(): [boolean, (text: string) => void] {
  const [copied, setCopied] = useState(false);
  const copy = useCallback((text: string) => {
    const done = () => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    };
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(text).then(done, () => fallbackCopy(text) && done());
    } else if (fallbackCopy(text)) {
      done();
    }
  }, []);
  return [copied, copy];
}

function fallbackCopy(text: string): boolean {
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  const ok = document.execCommand("copy");
  ta.remove();
  return ok;
}

export function CopyButton({ text, className, label }: { text: string; className?: string; label?: string }) {
  const [copied, copy] = useCopy();
  return (
    <button
      type="button"
      onClick={() => copy(text)}
      title={copied ? "Copied" : "Copy"}
      aria-label={label ?? "Copy"}
      className={cx(
        "copy-button inline-flex h-7 shrink-0 items-center gap-1.5 rounded-md px-2 text-[12px] text-fg-3 transition-colors hover:bg-panel-3 hover:text-fg",
        className,
      )}
    >
      {copied ? <Check className="size-3.5 text-good" /> : <Copy className="size-3.5" />}
      {label && <span>{copied ? "Copied" : label}</span>}
    </button>
  );
}

export function CopyField({ value, secret, label }: { value: string; secret?: boolean; label?: string }) {
  const [shown, setShown] = useState(!secret);
  return (
    <div>
      {label && <div className="readout mb-1.5">{label}</div>}
      <div className="flex h-10 items-center gap-1 rounded-lg border border-line bg-bg-2 pl-3 pr-1">
        <code className="min-w-0 flex-1 truncate font-mono text-[12.5px] text-fg">
          {shown ? value : "•".repeat(Math.min(value.length, 28))}
        </code>
        {secret && (
          <button
            type="button"
            onClick={() => setShown(!shown)}
            className="copy-reveal grid size-7 place-items-center rounded-md text-fg-3 hover:bg-panel-3 hover:text-fg"
            title={shown ? "Hide" : "Show"}
            aria-label={shown ? "Hide value" : "Show value"}
          >
            {shown ? <EyeOff className="size-3.5" /> : <Eye className="size-3.5" />}
          </button>
        )}
        <CopyButton text={value} />
      </div>
    </div>
  );
}

const SHORT_VERDICT: Record<Verdict, string> = { best: "Best fit", good: "Fits", tight: "Tight", worst: "Won't fit" };
const VERDICT_TONE: Record<Verdict, string> = {
  best: "text-good",
  good: "text-good",
  tight: "text-tight",
  worst: "text-worst",
};

/** Fit verdict: a colored dot and a word. "Best fit" gets a filled ink badge so it stands out in lists. */
export function VerdictTag({ verdict, size = "md" }: { verdict: Verdict; size?: "sm" | "md" }) {
  const label = size === "sm" ? SHORT_VERDICT[verdict] : VERDICT_LABEL[verdict];
  if (verdict === "best") {
    return (
      <span
        className={cx(
          "inline-flex items-center gap-1.5 whitespace-nowrap rounded-md bg-fg font-medium text-bg",
          size === "sm" ? "px-1.5 py-px text-[11.5px]" : "px-2 py-0.5 text-[12px]",
        )}
      >
        <span className="size-1.5 rounded-full bg-good" />
        {label}
      </span>
    );
  }
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1.5 whitespace-nowrap font-medium",
        size === "sm" ? "text-[12px]" : "text-[12.5px]",
        VERDICT_TONE[verdict],
      )}
    >
      <span className="size-1.5 rounded-full bg-current" />
      {label}
    </span>
  );
}

export function Chip({ children, tone = "plain", className }: { children: ReactNode; tone?: "plain" | "info" | "warn" | "signal"; className?: string }) {
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1 whitespace-nowrap rounded-[5px] px-1.5 py-px text-[11.5px] font-medium",
        tone === "plain" && "bg-panel-3 text-fg-2",
        tone === "info" && "bg-panel-3 text-info",
        tone === "warn" && "bg-tight-soft text-tight",
        tone === "signal" && "bg-good-soft text-good",
        className,
      )}
    >
      {children}
    </span>
  );
}

const FIELD =
  "w-full rounded-lg border border-line bg-bg-2 text-[16px] text-fg sm:text-[13.5px] placeholder:text-fg-3 transition-[border-color,box-shadow] duration-150 " +
  "hover:border-line-2 focus:border-fg-3 focus:shadow-[0_0_0_3px_var(--signal-soft)] focus:outline-none disabled:opacity-50";

export function Input({ className, ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return <input {...rest} className={cx(FIELD, "h-11 px-3 sm:h-9", className)} />;
}

export function Textarea({ className, ...rest }: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea {...rest} className={cx(FIELD, "min-h-11 resize-y px-3 py-2.5 leading-relaxed sm:min-h-0", className)} />;
}

export function Select({ className, children, ...rest }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select {...rest} className={cx(FIELD, "h-11 px-3 sm:h-9", className)}>
      {children}
    </select>
  );
}

export function Field({ label, hint, children }: { label: ReactNode; hint?: ReactNode; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-[12.5px] font-medium text-fg-2">{label}</span>
      {children}
      {hint && <span className="mt-1.5 block text-[12px] leading-snug text-fg-3">{hint}</span>}
    </label>
  );
}

export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  format,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}) {
  return (
    <label className="block">
      <span className="mb-1.5 flex items-baseline justify-between">
        <span className="text-[12.5px] font-medium text-fg-2">{label}</span>
        <span className="font-mono text-[12px] text-fg">{format ? format(value) : value}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-[var(--fg)]"
      />
    </label>
  );
}

/** Page title in the display serif, an optional one-line description, and actions on the right. */
export function SectionTitle({
  title,
  description,
  children,
}: {
  kicker?: string;
  title: ReactNode;
  description?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <header className="page-head">
      <div className="min-w-0">
        <h1 className="display">{title}</h1>
        {description && <p>{description}</p>}
      </div>
      {children && <div className="flex flex-wrap items-center gap-2">{children}</div>}
    </header>
  );
}

export function Empty({ icon, title, children }: { icon?: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="panel-quiet flex flex-col items-center px-6 py-14 text-center">
      {icon && <div className="mb-4 grid size-11 place-items-center rounded-xl border border-line bg-panel text-fg-2">{icon}</div>}
      <div className="display text-[28px]">{title}</div>
      {children && <div className="mt-2 max-w-md text-[13.5px] text-fg-2">{children}</div>}
    </div>
  );
}

export function Drawer({
  open,
  onClose,
  children,
  title,
  width = 560,
}: {
  open: boolean;
  onClose: () => void;
  children: ReactNode;
  title?: ReactNode;
  width?: number;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [open, onClose]);
  if (!open) return null;
  return createPortal(
    <div className="fixed inset-0 z-50 flex justify-end p-0 sm:p-2">
      <div className="absolute inset-0 bg-[rgb(8_8_7/0.5)] backdrop-blur-[3px] animate-[rise_.18s_ease_both]" onClick={onClose} />
      <aside
        role="dialog"
        aria-modal="true"
        className="relative flex h-full w-full flex-col overflow-hidden border-line bg-panel shadow-[0_30px_80px_-20px_rgb(0_0_0/0.6)] sm:rounded-2xl sm:border animate-[rise_.24s_cubic-bezier(.2,.7,.2,1)_both]"
        style={{ maxWidth: width }}
      >
        <div className="flex h-14 shrink-0 items-center justify-between gap-3 border-b border-line px-5">
          <div className="min-w-0 truncate text-[13.5px] font-medium text-fg-2">{title}</div>
          <button onClick={onClose} className="toolbar-icon" title="Close" aria-label="Close">
            <X className="size-4" />
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto scroll-thin">{children}</div>
      </aside>
    </div>,
    document.body,
  );
}

/* ---------------------------------------------------------------------------------------- */
/* Toasts                                                                                     */
/* ---------------------------------------------------------------------------------------- */

interface Toast {
  id: number;
  text: string;
  tone: "ok" | "error";
}
const ToastCtx = createContext<(text: string, tone?: Toast["tone"]) => void>(() => {});

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const push = useCallback((text: string, tone: Toast["tone"] = "ok") => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t, { id, text, tone }]);
    window.setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), tone === "error" ? 7000 : 3200);
  }, []);
  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div className="pointer-events-none fixed bottom-5 right-5 z-[60] flex w-[min(400px,calc(100vw-40px))] flex-col gap-2" aria-live="polite">
        {toasts.map((t) => (
          <div
            key={t.id}
            className="pointer-events-auto flex items-start gap-3 rounded-xl border border-line-2 bg-panel px-4 py-3 text-[13.5px] shadow-[var(--edge),0_20px_50px_-20px_rgb(0_0_0/0.5)] animate-[rise_.22s_ease_both]"
          >
            <span className={cx("mt-[7px] size-1.5 shrink-0 rounded-full", t.tone === "ok" ? "bg-good" : "bg-worst")} />
            <span className={t.tone === "error" ? "text-fg" : "text-fg"}>{t.text}</span>
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
}

export const useToast = () => useContext(ToastCtx);
