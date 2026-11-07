import {
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  Download,
  ImagePlus,
  Mic,
  Square,
  Upload,
  Wand2,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { ProgressBar } from "../../components/meters";
import { Button, CopyButton, Field, Input, Select, Slider, Textarea, useToast } from "../../components/ui";
import { gateway } from "../../lib/api";
import { cx } from "../../lib/format";
import type { StudioModel } from "../../lib/types";

export function fileToB64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).split(",", 2)[1] ?? "");
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

export function ImagePick({ file, onFile, label }: { file: File | null; onFile: (f: File | null) => void; label: string }) {
  const url = file ? URL.createObjectURL(file) : null;
  useEffect(() => () => (url ? URL.revokeObjectURL(url) : undefined), [url]);
  return (
    <div>
      <div className="readout mb-1.5">{label}</div>
      {file && url ? (
        <div className="relative inline-block">
          <img src={url} alt="" className="max-h-36 rounded-lg border border-line" />
          <button onClick={() => onFile(null)} className="absolute -right-2 -top-2 rounded-full border border-line bg-panel-3 p-1">
            <X className="size-3" />
          </button>
        </div>
      ) : (
        <label className="flex cursor-pointer items-center gap-2 rounded-lg border border-dashed border-line-2 px-3 py-3 text-[13px] text-fg-2 hover:border-fg-3 hover:text-fg">
          <ImagePlus className="size-4" /> Choose an image (optional)
          <input type="file" accept="image/*" className="hidden" onChange={(e) => onFile(e.target.files?.[0] ?? null)} />
        </label>
      )}
    </div>
  );
}

/* ======================================================================================== */
/* Image                                                                                    */
/* ======================================================================================== */

const SIZES = ["512x512", "768x768", "1024x1024", "832x1216", "1216x832", "1024x576", "576x1024"];

export function ImageStudio({ model }: { model: StudioModel }) {
  const o = model.options ?? {};
  const toast = useToast();
  const [prompt, setPrompt] = useState("");
  const [negative, setNegative] = useState("");
  const [steps, setSteps] = useState<number>(Number(o.steps ?? 25));
  const [guidance, setGuidance] = useState<number>(Number(o.guidance ?? 5));
  const [size, setSize] = useState(`${o.width ?? 1024}x${o.height ?? 1024}`);
  const [seed, setSeed] = useState("");
  const [n, setN] = useState(1);
  const [busy, setBusy] = useState(false);
  const [started, setStarted] = useState(0);
  const [results, setResults] = useState<{ src: string; prompt: string; seconds: number }[]>([]);

  async function generate() {
    if (!prompt.trim()) return;
    setBusy(true);
    const t0 = performance.now();
    setStarted(t0);
    try {
      const res = await gateway.post("/v1/images/generations", {
        model: model.name, prompt, negative_prompt: negative || undefined, steps, guidance, size, n,
        seed: seed ? Number(seed) : undefined,
      });
      const data = await res.json();
      const seconds = Math.round((performance.now() - t0) / 100) / 10;
      setResults((r) => [...data.data.map((d: { b64_json: string }) => ({ src: `data:image/png;base64,${d.b64_json}`, prompt, seconds })), ...r]);
    } catch (e) {
      toast(e instanceof Error ? e.message : "Generation failed", "error");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[380px_minmax(0,1fr)]">
      <div className="panel space-y-4 p-4">
        <Field label="Prompt">
          <Textarea rows={4} value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="A lighthouse on a basalt cliff at blue hour, film grain"
            onKeyDown={(e) => e.key === "Enter" && (e.metaKey || e.ctrlKey) && generate()} />
        </Field>
        <Field label="Negative prompt">
          <Input value={negative} onChange={(e) => setNegative(e.target.value)} placeholder="blurry, low quality" />
        </Field>
        <Slider label="Steps" value={steps} min={1} max={60} onChange={setSteps} />
        <Slider label="Guidance" value={guidance} min={0} max={15} step={0.5} onChange={setGuidance} format={(v) => v.toFixed(1)} />
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          <div className="col-span-2 sm:col-span-1"><Field label="Size">
            <Select value={size} onChange={(e) => setSize(e.target.value)}>
              {[...new Set([size, ...SIZES])].map((s) => <option key={s}>{s}</option>)}
            </Select>
          </Field></div>
          <Field label="Seed">
            <Input value={seed} onChange={(e) => setSeed(e.target.value.replace(/\D/g, ""))} placeholder="random" />
          </Field>
          <Field label="Images">
            <Select value={n} onChange={(e) => setN(Number(e.target.value))}>
              {[1, 2, 3, 4].map((v) => <option key={v}>{v}</option>)}
            </Select>
          </Field>
        </div>
        <Button variant="primary" size="lg" className="w-full" busy={busy} onClick={generate} disabled={!prompt.trim()}>
          <Wand2 className="size-4" /> Generate
        </Button>
        {busy && <Elapsed since={started} />}
      </div>
      <div>
        {results.length === 0 ? (
          <div className="panel grid h-full min-h-[360px] place-items-center p-8 text-center text-fg-3">
            <div>
              <div className="display text-[40px]">{model.name}</div>
              <p className="mt-1 text-[13.5px]">Images show up here. Ctrl/⌘+Enter generates.</p>
            </div>
          </div>
        ) : (
          <div className="grid gap-3 sm:grid-cols-2">
            {results.map((r, i) => (
              <figure key={i} className="panel overflow-hidden animate-[rise_.35s_ease_both]">
                <img src={r.src} alt={r.prompt} className="w-full" />
                <figcaption className="flex items-center gap-2 px-3 py-2 text-[12px] text-fg-2">
                  <span className="min-w-0 flex-1 truncate">{r.prompt}</span>
                  <span className="font-mono text-[10.5px] text-fg-3">{r.seconds}s</span>
                  <a href={r.src} download={`kaggleui-${Date.now()}.png`} className="text-fg-2 hover:text-fg">
                    <Download className="size-3.5" />
                  </a>
                </figcaption>
              </figure>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function Elapsed({ since }: { since: number }) {
  const [, tick] = useState(0);
  useEffect(() => {
    const t = window.setInterval(() => tick((x) => x + 1), 200);
    return () => window.clearInterval(t);
  }, []);
  return <div className="text-center font-mono text-[11px] text-fg-3">{((performance.now() - since) / 1000).toFixed(1)}s</div>;
}

/* ======================================================================================== */
/* Video & World                                                                            */
/* ======================================================================================== */

export interface Segment {
  keys: string[];
  camera?: [number, number];
  frames: number;
  label: string;
}

const KEYS: { key: string; label: string; hint: string }[] = [
  { key: "forward", label: "W", hint: "forward" },
  { key: "left", label: "A", hint: "strafe left" },
  { key: "back", label: "S", hint: "back" },
  { key: "right", label: "D", hint: "strafe right" },
  { key: "jump", label: "␣", hint: "jump" },
  { key: "sprint", label: "⇧", hint: "sprint" },
  { key: "attack", label: "LMB", hint: "attack / mine" },
  { key: "use", label: "RMB", hint: "use / place" },
];

export function ActionBuilder({ segments, onChange }: { segments: Segment[]; onChange: (s: Segment[]) => void }) {
  const [frames, setFrames] = useState(4);
  const add = (s: Omit<Segment, "frames">) => onChange([...segments, { ...s, frames }]);
  const look: { cam: [number, number]; icon: typeof ArrowLeft; label: string }[] = [
    { cam: [0, -0.5], icon: ArrowLeft, label: "look left" },
    { cam: [-0.5, 0], icon: ArrowUp, label: "look up" },
    { cam: [0.5, 0], icon: ArrowDown, label: "look down" },
    { cam: [0, 0.5], icon: ArrowRight, label: "look right" },
  ];
  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between">
        <span className="readout">Controls → timeline</span>
        <label className="flex items-center gap-2 whitespace-nowrap text-[12px] text-fg-2">
          frames per press
          <Select value={frames} onChange={(e) => setFrames(Number(e.target.value))} className="h-8 w-16 px-2 text-[12px]">
            {[2, 4, 8, 12, 16].map((f) => <option key={f}>{f}</option>)}
          </Select>
        </label>
      </div>
      <div className="grid grid-cols-4 gap-1.5">
        {KEYS.map((k) => (
          <button key={k.key} title={k.hint} onClick={() => add({ keys: [k.key], label: k.hint })}
            className="rounded-lg border border-line bg-bg-2 py-2 font-mono text-[13px] hover:border-signal hover:text-signal">
            {k.label}
          </button>
        ))}
        {look.map(({ cam, icon: Icon, label }) => (
          <button key={label} title={label} onClick={() => add({ keys: [], camera: cam, label })}
            className="grid place-items-center rounded-lg border border-line bg-bg-2 py-2 text-fg-2 hover:border-signal hover:text-signal">
            <Icon className="size-4" />
          </button>
        ))}
      </div>
      <div className="flex min-h-10 flex-wrap gap-1.5 rounded-lg border border-line bg-bg p-2">
        {segments.length === 0 && <span className="text-[12px] text-fg-3">Press controls to build an action sequence (idle → walk forward).</span>}
        {segments.map((s, i) => (
          <span key={i} className="inline-flex items-center gap-1 rounded-md bg-panel-3 py-0.5 pl-2 pr-1 font-mono text-[11px]">
            {s.label} ×{s.frames}
            <button onClick={() => onChange(segments.filter((_, j) => j !== i))} className="rounded p-0.5 text-fg-3 hover:text-worst">
              <X className="size-3" />
            </button>
          </span>
        ))}
      </div>
    </div>
  );
}

/** Submit a /v1/videos job, poll it, and return the finished clip as an object URL. */
export async function runVideoJob(
  body: Record<string, unknown>,
  onProgress: (p: { id: string; progress: number; message: string }) => void,
): Promise<string> {
  const res = await gateway.post("/v1/videos", body);
  const v = await res.json();
  onProgress({ id: v.id, progress: 0, message: v.start_frame_model ? `First frame painted with ${v.start_frame_model}` : "Queued" });
  for (;;) {
    await new Promise((r) => setTimeout(r, 1500));
    const s = await (await gateway.get(`/v1/videos/${v.id}`)).json();
    onProgress({ id: v.id, progress: (s.progress ?? 0) / 100, message: s.message ?? s.status });
    if (s.status === "completed") break;
    if (s.status === "failed") throw new Error(s.error?.message ?? "Generation failed");
  }
  const blob = await (await gateway.get(`/v1/videos/${v.id}/content`)).blob();
  return URL.createObjectURL(blob);
}

export function VideoStudio({ model }: { model: StudioModel }) {
  const o = model.options ?? {};
  const toast = useToast();
  const [prompt, setPrompt] = useState("");
  const [image, setImage] = useState<File | null>(null);
  const [frames, setFrames] = useState<number>(Number(o.frames ?? 33));
  const [steps, setSteps] = useState<number>(Number(o.steps ?? 25));
  const [seed, setSeed] = useState("");
  const [job, setJob] = useState<{ id: string; progress: number; message: string } | null>(null);
  const [videos, setVideos] = useState<{ url: string; prompt: string }[]>([]);

  async function generate() {
    try {
      const body: Record<string, unknown> = { model: model.name, prompt, frames, steps, seed: seed ? Number(seed) : undefined };
      if (image) body.input_reference_b64 = await fileToB64(image);
      const url = await runVideoJob(body, setJob);
      setVideos((vs) => [{ url, prompt }, ...vs]);
    } catch (e) {
      toast(e instanceof Error ? e.message : "Generation failed", "error");
    } finally {
      setJob(null);
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[400px_minmax(0,1fr)]">
      <div className="panel space-y-4 p-4">
        <Field label="Prompt">
          <Textarea rows={4} value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="A red fox trotting through fresh snow, golden hour, slow motion" />
        </Field>
        <ImagePick file={image} onFile={setImage} label="Start image (image-to-video models)" />
        <Slider label="Frames" value={frames} min={9} max={161} onChange={setFrames} />
        <Slider label="Steps" value={steps} min={1} max={60} onChange={setSteps} />
        <Field label="Seed">
          <Input value={seed} onChange={(e) => setSeed(e.target.value.replace(/\D/g, ""))} placeholder="random" />
        </Field>
        <Button variant="primary" size="lg" className="w-full" busy={!!job} onClick={generate} disabled={!!job || (!prompt.trim() && !image)}>
          <Wand2 className="size-4" /> Generate
        </Button>
        {job && <JobProgress job={job} />}
        <p className="text-[12px] text-fg-3">Video is slow on T4s: a few minutes per clip at default settings.</p>
      </div>
      <ClipList videos={videos} placeholder={model.name} />
    </div>
  );
}

export function JobProgress({ job }: { job: { progress: number; message: string } }) {
  return (
    <div>
      <ProgressBar value={job.progress || null} />
      <div className="mt-1.5 flex justify-between gap-3 text-[12px] text-fg-2">
        <span className="min-w-0 truncate">{job.message}</span>
        <span className="font-mono">{Math.round(job.progress * 100)}%</span>
      </div>
    </div>
  );
}

export function ClipList({ videos, placeholder }: { videos: { url: string; prompt: string; frame?: string }[]; placeholder: string }) {
  if (videos.length === 0) {
    return (
      <div className="panel grid h-full min-h-[360px] place-items-center p-8 text-center text-fg-3">
        <div>
          <div className="display text-[40px]">{placeholder}</div>
          <p className="mt-1 text-[13.5px]">Clips show up here.</p>
        </div>
      </div>
    );
  }
  return (
    <div className="space-y-3">
      {videos.map((v, i) => (
        <figure key={v.url} className="panel overflow-hidden animate-[rise_.35s_ease_both]">
          <video src={v.url} controls autoPlay={i === 0} loop className="w-full bg-black" />
          <figcaption className="flex items-center gap-2 px-3 py-2 text-[12px] text-fg-2">
            {v.frame && <img src={v.frame} alt="Starting frame" title="Starting frame" className="h-8 rounded border border-line" />}
            <span className="min-w-0 flex-1 truncate">{v.prompt}</span>
            <a href={v.url} download={`kaggleui-${Date.now()}.mp4`} className="hover:text-fg">
              <Download className="size-3.5" />
            </a>
          </figcaption>
        </figure>
      ))}
    </div>
  );
}

/* ======================================================================================== */
/* Text to speech                                                                           */
/* ======================================================================================== */

export function TtsStudio({ model }: { model: StudioModel }) {
  const isAudio = model.modality === "audio";
  const toast = useToast();
  const [text, setText] = useState("");
  const [voices, setVoices] = useState<string[]>([]);
  const [voice, setVoice] = useState("");
  const [voiceInfo, setVoiceInfo] = useState<{ mode?: string; note?: string; supports_reference?: boolean }>({});
  const [voiceError, setVoiceError] = useState("");
  const [speed, setSpeed] = useState(1);
  const [format, setFormat] = useState("mp3");
  const [reference, setReference] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [clips, setClips] = useState<{ url: string; text: string; voice: string; format: string }[]>([]);

  useEffect(() => {
    setReference(null);
    if (isAudio) return;
    let cancelled = false;
    setVoiceError("");
    gateway.get(`/v1/audio/voices?model=${encodeURIComponent(model.name)}`).then((r) => r.json()).then((d) => {
      if (cancelled) return;
      setVoices(d.voices ?? []);
      setVoiceInfo(d);
      setVoice((v) => d.voices?.includes(v) ? v : d.voices?.[0] || "");
    }).catch(error => { if (!cancelled) setVoiceError(error instanceof Error ? error.message : "Could not load voice choices."); });
    return () => { cancelled = true; };
  }, [model.name, isAudio]);

  async function speak() {
    setBusy(true);
    try {
      const res = await gateway.post(isAudio ? "/v1/audio/generations" : "/v1/audio/speech", { model: model.name, input: text, ...(isAudio ? { prompt: text } : { voice: voice || undefined, speed }), response_format: format,
        ...(reference ? { reference_audio_b64: await fileToB64(reference) } : {}) });
      const blob = await res.blob();
      setClips((c) => [{ url: URL.createObjectURL(blob), text, voice, format: blob.type.includes("wav") ? "wav" : format }, ...c]);
    } catch (e) {
      toast(e instanceof Error ? e.message : "Audio generation failed", "error");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_360px]">
      <div className="panel space-y-4 p-4">
        <Field label={isAudio ? "Audio prompt" : "Text"}>
          <Textarea rows={8} value={text} onChange={(e) => setText(e.target.value)} placeholder={model.adapter === "midasheng" ? "Describe an audio scene, music, effects or speech. Use <|caption|>, <|asr|>, <|speech|> for structured prompts." : isAudio ? "Describe the music, sounds or ambience to generate." : "Type something to hear it."} />
        </Field>
        {!isAudio && voiceInfo.note && <p className="text-[12px] leading-relaxed text-fg-2">{voiceInfo.note}</p>}
        {!isAudio && voiceError && <p role="alert" className="text-[12px] text-tight">{voiceError}</p>}
        {!isAudio && (voiceInfo.supports_reference || model.engine === "chatterbox" || model.adapter?.startsWith("vibevoice")) && <Field label="Reference voice (optional)" hint="Upload an audio clip to guide the speaker’s voice.">
          <div className="flex items-center gap-2"><Input type="file" accept="audio/*" onChange={e => setReference(e.target.files?.[0] ?? null)} />
            {reference && <Button variant="ghost" size="sm" onClick={() => setReference(null)}>Clear</Button>}</div>
        </Field>}
        <div className={isAudio ? "max-w-xs" : "grid gap-3 sm:grid-cols-3"}>
          {!isAudio && <Field label="Voice">
            <Select value={voice} onChange={(e) => setVoice(e.target.value)} disabled={voices.length <= 1}>
              {voices.length ? voices.map((v) => <option key={v} value={v}>{voiceInfo.mode === "speakers" ? `Speaker ${v}` : v === "default" ? voiceInfo.mode === "single" ? "Fixed speaker" : voiceInfo.mode === "reference" ? "Default / reference voice" : "Model default" : v}</option>) : <option value="">{voiceInfo.mode === "prompt" ? "Prompt controlled" : "No voice list available"}</option>}
            </Select>
          </Field>}
          {!isAudio && <Slider label="Speed" value={speed} min={0.5} max={2} step={0.05} onChange={setSpeed} format={(v) => `${v.toFixed(2)}×`} />}
          <Field label="Format">
            <Select value={format} onChange={(e) => setFormat(e.target.value)}>
              {["mp3", "wav", "flac", "opus"].map((f) => <option key={f}>{f}</option>)}
            </Select>
          </Field>
        </div>
        <Button variant="primary" size="lg" busy={busy} onClick={speak} disabled={!text.trim()}>
          <Wand2 className="size-4" /> {isAudio ? "Generate audio" : "Speak"}
        </Button>
      </div>
      <div className="space-y-2">
        {clips.length === 0 ? (
          <div className="panel p-6 text-center text-[13.5px] text-fg-3">Generated audio shows up here.</div>
        ) : (
          clips.map((c, i) => (
            <div key={c.url} className="panel p-3 animate-[rise_.3s_ease_both]">
              <div className="mb-2 flex items-center gap-2 text-[12px] text-fg-2">
                <span className="font-mono text-[10.5px] text-fg-3">{c.voice}</span>
                <span className="min-w-0 flex-1 truncate">{c.text}</span>
                <a href={c.url} download={`kaggleui-speech.${c.format}`} className="hover:text-fg">
                  <Download className="size-3.5" />
                </a>
              </div>
              <audio src={c.url} controls autoPlay={i === 0} className="w-full" />
            </div>
          ))
        )}
      </div>
    </div>
  );
}

/* ======================================================================================== */
/* Speech to text                                                                           */
/* ======================================================================================== */

interface Transcript {
  text: string;
  language?: string | null;
  duration?: number | null;
  segments?: { start: number | null; end: number | null; text: string }[];
}

function ts(t: number | null | undefined): string {
  if (t == null) return "--:--";
  const m = Math.floor(t / 60);
  return `${String(m).padStart(2, "0")}:${(t % 60).toFixed(1).padStart(4, "0")}`;
}

export function SttStudio({ model }: { model: StudioModel }) {
  const toast = useToast();
  const [file, setFile] = useState<File | null>(null);
  const [language, setLanguage] = useState("");
  const [task, setTask] = useState<"transcribe" | "translate">("transcribe");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Transcript | null>(null);
  const [recording, setRecording] = useState(false);
  const recorder = useRef<MediaRecorder | null>(null);
  const fileUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  useEffect(() => () => (fileUrl ? URL.revokeObjectURL(fileUrl) : undefined), [fileUrl]);

  async function toggleRecord() {
    if (recording) {
      recorder.current?.stop();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      rec.ondataavailable = (e) => chunks.push(e.data);
      rec.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunks, { type: rec.mimeType || "audio/webm" });
        setFile(new File([blob], `recording.${blob.type.includes("ogg") ? "ogg" : "webm"}`, { type: blob.type }));
        setRecording(false);
      };
      rec.start();
      recorder.current = rec;
      setRecording(true);
    } catch {
      toast("Microphone access was blocked", "error");
    }
  }

  async function run() {
    if (!file) return;
    setBusy(true);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("model", model.name);
      form.append("response_format", "verbose_json");
      if (language) form.append("language", language);
      const res = await gateway.form(task === "translate" ? "/v1/audio/translations" : "/v1/audio/transcriptions", form);
      setResult(await res.json());
    } catch (e) {
      toast(e instanceof Error ? e.message : "Transcription failed", "error");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[380px_minmax(0,1fr)]">
      <div className="panel space-y-4 p-4">
        <div className="grid grid-cols-2 gap-2">
          <label className="flex cursor-pointer flex-col items-center gap-2 rounded-xl border border-dashed border-line-2 px-3 py-5 text-center text-[13px] text-fg-2 hover:border-fg-3 hover:text-fg">
            <Upload className="size-5" />
            Upload audio
            <input type="file" accept="audio/*,video/*" className="hidden" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </label>
          <button onClick={toggleRecord}
            className={cx("flex flex-col items-center gap-2 rounded-xl border px-3 py-5 text-[13px]", recording ? "border-worst bg-worst-soft text-worst" : "border-dashed border-line-2 text-fg-2 hover:border-fg-3 hover:text-fg")}>
            {recording ? <Square className="size-5" /> : <Mic className="size-5" />}
            {recording ? "Stop recording" : "Record"}
          </button>
        </div>
        {file && (
          <div className="rounded-lg border border-line bg-bg-2 p-2.5">
            <div className="mb-1.5 truncate font-mono text-[11.5px] text-fg-2">{file.name}</div>
            <audio src={fileUrl ?? undefined} controls className="w-full" />
          </div>
        )}
        <div className="grid grid-cols-2 gap-2">
          <Field label="Language">
            <Input value={language} onChange={(e) => setLanguage(e.target.value)} placeholder="auto" />
          </Field>
          <Field label="Task">
            <Select value={task} onChange={(e) => setTask(e.target.value as typeof task)}>
              <option value="transcribe">Transcribe</option>
              <option value="translate">Translate to English</option>
            </Select>
          </Field>
        </div>
        <Button variant="primary" size="lg" className="w-full" busy={busy} onClick={run} disabled={!file}>
          Transcribe
        </Button>
      </div>
      <div className="panel min-h-[360px] p-5">
        {!result ? (
          <div className="grid h-full place-items-center text-center text-[13.5px] text-fg-3">The transcript shows up here.</div>
        ) : (
          <div className="animate-[rise_.3s_ease_both]">
            <div className="mb-3 flex flex-wrap items-center gap-3">
              <span className="readout">Transcript</span>
              {result.language && <span className="font-mono text-[11px] text-fg-3">{result.language}</span>}
              {result.duration ? <span className="font-mono text-[11px] text-fg-3">{result.duration.toFixed(1)}s</span> : null}
              <span className="ml-auto">
                <CopyButton text={result.text} label="Copy" />
              </span>
            </div>
            <p className="mb-5 text-[15px] leading-relaxed">{result.text || <span className="text-fg-3">(no speech detected)</span>}</p>
            {result.segments && result.segments.length > 1 && (
              <div className="space-y-1 border-t border-line pt-4">
                {result.segments.map((s, i) => (
                  <div key={i} className="flex gap-3 text-[13px]">
                    <span className="shrink-0 font-mono text-[11px] text-fg-3">{ts(s.start)}</span>
                    <span className="text-fg-2">{s.text}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
