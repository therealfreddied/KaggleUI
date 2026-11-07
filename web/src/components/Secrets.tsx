import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, CircleAlert, ExternalLink, RefreshCw, Trash2 } from "lucide-react";
import { useState } from "react";

import { api } from "../lib/api";
import { cx } from "../lib/format";
import type { SecretDetails, SecretItem } from "../lib/types";
import { Button, Input, useToast } from "./ui";

const INFO: Record<string, { label: string; help: string; link?: string; placeholder?: string }> = {
  HF_TOKEN: {
    label: "Hugging Face token",
    help: "Needed for gated models (FLUX, Llama, SD 3.5…). A read token is enough; accept each model's license on its HF page.",
    link: "https://huggingface.co/settings/tokens",
    placeholder: "hf_…",
  },
  NGROK_AUTHTOKEN: { label: "ngrok authtoken", help: "Unlocks the ngrok tunnel, with real-time API streaming.", link: "https://dashboard.ngrok.com/get-started/your-authtoken" },
  NGROK_DOMAIN: { label: "ngrok static domain", help: "Optional free fixed URL, e.g. you.ngrok-free.app.", placeholder: "you.ngrok-free.app" },
  CF_TUNNEL_TOKEN: { label: "Cloudflare tunnel token", help: "For a named tunnel on your own domain." },
  CF_TUNNEL_HOSTNAME: { label: "Cloudflare tunnel hostname", help: "The public hostname routed to http://localhost:8890.", placeholder: "ai.example.com" },
  KAGGLEUI_NOTIFY_URL: { label: "Notification URL", help: "An ntfy.sh topic URL or Discord webhook. Gets the link when KaggleUI starts.", placeholder: "https://ntfy.sh/your-topic" },
};

const SOURCE_LABEL: Record<string, string> = { kaggle: "From Kaggle secret", colab: "From Colab secret", attached: "From host secret", kaggleui: "Set in KaggleUI", env: "From environment" };

function Row({ item, hf, onSaved }: { item: SecretItem; hf?: SecretDetails["hf"]; onSaved: (d: SecretDetails) => void }) {
  const info = INFO[item.name] ?? { label: item.name, help: "" };
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState<"save" | "clear" | null>(null);
  const toast = useToast();

  async function save() {
    setBusy("save");
    try {
      const d = await api.setSecret(item.name, value);
      setValue("");
      onSaved(d);
      if (item.name === "HF_TOKEN") {
        if (d.hf?.ok) toast(`Hugging Face token saved: signed in as ${d.hf.user}`);
        else toast(d.hf?.error ?? "Saved, but the token couldn't be checked", "error");
      } else {
        toast(`${info.label} saved`);
      }
    } catch (e) {
      toast(e instanceof Error ? e.message : "Couldn't save", "error");
    } finally {
      setBusy(null);
    }
  }

  async function clear() {
    setBusy("clear");
    try {
      onSaved(await api.clearSecret(item.name));
      toast(`${info.label} removed`);
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="border-b border-dashed border-line py-4 first:pt-0 last:border-0 last:pb-0">
      <div className="mb-1 flex flex-wrap items-center gap-2">
        <span className="font-medium">{info.label}</span>
        <span
          className={cx(
            "rounded-[5px] px-1.5 py-px text-[11px] font-medium",
            item.set ? "bg-signal-soft text-signal" : "bg-panel-3 text-fg-3",
          )}
        >
          {item.set ? (SOURCE_LABEL[item.source ?? ""] ?? "Set") : "Not set"}
        </span>
        {info.link && (
          <a href={info.link} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-[12px] text-fg-3 hover:text-fg">
            get one <ExternalLink className="size-3" />
          </a>
        )}
      </div>
      <p className="mb-2.5 text-[12.5px] text-fg-3">{info.help}</p>
      {item.name === "HF_TOKEN" && item.set && hf && (hf.ok || hf.error) && (
        <div className={cx("mb-2.5 flex items-center gap-1.5 text-[12.5px]", hf.ok ? "text-signal" : "text-worst")}>
          {hf.ok ? <Check className="size-3.5" /> : <CircleAlert className="size-3.5" />}
          {hf.ok ? `Signed in to Hugging Face as ${hf.user}` : hf.error}
        </div>
      )}
      <form
        className="flex flex-col gap-2 sm:flex-row"
        onSubmit={(e) => {
          e.preventDefault();
          if (value.trim()) save();
        }}
      >
        <Input
          type={item.name.endsWith("DOMAIN") || item.name.endsWith("HOSTNAME") || item.name.endsWith("URL") ? "text" : "password"}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder={item.set ? "Paste a new value to replace it" : (info.placeholder ?? "Paste value")}
          autoComplete="off"
          spellCheck={false}
          className="font-mono text-[12.5px]"
        />
        <div className="flex gap-2">
          <Button type="submit" variant={value.trim() ? "primary" : "outline"} busy={busy === "save"} disabled={!value.trim()}>
            Save
          </Button>
          {item.set && (
            <Button type="button" variant="ghost" busy={busy === "clear"} onClick={clear} title="Remove">
              <Trash2 className="size-4" />
            </Button>
          )}
        </div>
      </form>
    </div>
  );
}

export function SecretsCard() {
  const qc = useQueryClient();
  const toast = useToast();
  const q = useQuery({ queryKey: ["secrets"], queryFn: api.secrets });
  const [reloading, setReloading] = useState(false);

  function apply(d: SecretDetails) {
    qc.setQueryData<SecretDetails>(["secrets"], (prev) => ({ ...d, hf: d.hf ?? prev?.hf }));
    // Gated-model warnings and fit tags depend on the HF token.
    qc.invalidateQueries({ queryKey: ["search"] });
    qc.invalidateQueries({ queryKey: ["detail"] });
  }

  async function reload() {
    setReloading(true);
    try {
      const d = await api.reloadSecrets();
      apply(d);
      if (d.added?.length) toast(`Loaded from Kaggle: ${d.added.join(", ")}`);
      else if (d.kaggle.state !== "available") toast(d.kaggle.error ?? "Kaggle secrets aren't available here", "error");
      else toast("No new Kaggle secrets found. Is the secret switched on for this notebook (Add-ons → Secrets)?", "error");
    } catch (e) {
      toast(e instanceof Error ? e.message : "Reload failed", "error");
    } finally {
      setReloading(false);
    }
  }

  const items = (q.data?.items ?? []).filter((i) => i.editable);
  const kaggle = q.data?.kaggle;
  return (
    <section id="secrets" className="panel min-w-0 scroll-mt-24 p-5 sm:p-6 xl:col-span-2">
      <div className="mb-1.5 flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="readout mb-1.5">Hugging Face, tunnels, notifications</div>
          <h2 className="display text-[30px]">Keys & tokens</h2>
        </div>
        <Button size="sm" onClick={reload} busy={reloading}>
          <RefreshCw className="size-3.5" /> Reload Kaggle secrets
        </Button>
      </div>
      <p className="mb-5 max-w-3xl text-[13px] text-fg-2">
        Paste keys here, or add them as Kaggle secrets (<b>Add-ons → Secrets</b>, then switch each one <b>on</b> for this notebook) and press
        Reload. Values entered here stay on this session's temporary disk only. They're never written to <code className="font-mono text-[12px]">/kaggle/working</code>{" "}
        or the notebook output, and never sent back to the browser.
        {kaggle && kaggle.state !== "available" && <span className="mt-1 block text-fg-3">Kaggle secrets: {kaggle.error ?? kaggle.state}</span>}
      </p>
      {q.isLoading ? (
        <div className="h-40 animate-pulse rounded-lg bg-panel-2" />
      ) : (
        <div className="grid gap-x-8 lg:grid-cols-2">
          <div>
            {items.slice(0, 3).map((i) => (
              <Row key={i.name} item={i} hf={q.data?.hf} onSaved={apply} />
            ))}
          </div>
          <div className="border-t border-dashed border-line pt-4 lg:border-0 lg:pt-0">
            {items.slice(3).map((i) => (
              <Row key={i.name} item={i} hf={q.data?.hf} onSaved={apply} />
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
