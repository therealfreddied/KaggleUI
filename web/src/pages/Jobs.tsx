import { useQuery } from "@tanstack/react-query";
import { ListChecks } from "lucide-react";
import { useState } from "react";

import { Button, Chip, Empty, SectionTitle } from "../components/ui";
import { api } from "../lib/api";
import { duration } from "../lib/format";
import type { Job } from "../lib/types";

const TONE: Record<string, "plain" | "info" | "warn" | "signal"> = {
  running: "signal",
  queued: "info",
  succeeded: "plain",
  failed: "warn",
  cancelled: "warn",
  interrupted: "warn",
};

function Logs({ job }: { job: Job }) {
  const logs = useQuery({
    queryKey: ["job-logs", job.id, job.state],
    queryFn: () => api.jobLogs(job.id),
    refetchInterval: ["queued", "running"].includes(job.state) ? 2000 : false,
  });
  return (
    <pre className="max-h-[46vh] overflow-auto rounded-lg border border-line bg-bg-2 p-3 text-[12px] leading-[1.55]">
      {(logs.data?.lines ?? []).join("\n") || "No output yet."}
    </pre>
  );
}

export default function JobsPage() {
  const [selected, setSelected] = useState<string | null>(null);
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: () => api.jobs({ limit: 100 }), refetchInterval: 3000 });
  const rows = jobs.data ?? [];
  const current = rows.find((j) => j.id === selected) ?? rows[0];

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between gap-4">
        <div>
          <h1 className="text-[22px] font-semibold tracking-[-0.02em]">Jobs</h1>
          <p className="mt-1 text-[13.5px] text-muted">
            Persistent work: installs, downloads, benchmarks, training and teacher runs. Logs survive a restart.
          </p>
        </div>
        <Button variant="ghost" onClick={() => jobs.refetch()}>Refresh</Button>
      </div>

      {rows.length === 0 ? (
        <Empty icon={<ListChecks className="size-5" />} title="No jobs yet">
          Deploy a model or start a training run and it appears here with its log.
        </Empty>
      ) : (
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
          <div className="overflow-hidden rounded-xl border border-line">
            <table className="w-full text-left text-[13px]">
              <thead className="bg-bg-2 text-muted">
                <tr>
                  <th className="px-3 py-2 font-medium">Kind</th>
                  <th className="px-3 py-2 font-medium">State</th>
                  <th className="px-3 py-2 font-medium">Progress</th>
                  <th className="px-3 py-2 font-medium">Elapsed</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((job) => (
                  <tr key={job.id} onClick={() => setSelected(job.id)}
                    className={`cursor-pointer border-t border-line hover:bg-bg-2 ${current?.id === job.id ? "bg-bg-2" : ""}`}>
                    <td className="px-3 py-2">{job.kind}</td>
                    <td className="px-3 py-2"><Chip tone={TONE[job.state] ?? "plain"}>{job.state}</Chip></td>
                    <td className="px-3 py-2 text-muted">{job.progress == null ? "—" : `${Math.round(job.progress * 100)}%`}</td>
                    <td className="px-3 py-2 text-muted">
                      {duration((job.ended_at ?? Date.now() / 1000) - job.created_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {current && (
            <div className="space-y-3">
              <SectionTitle title={`${current.kind} ${current.id}`}
                description={current.error ?? current.message ?? undefined} />
              <div className="flex gap-2">
                {["queued", "running"].includes(current.state) && (
                  <Button size="sm" onClick={() => api.cancelJob(current.id)}>Cancel</Button>
                )}
                {current.resume_supported && ["failed", "interrupted", "cancelled"].includes(current.state) && (
                  <Button size="sm" onClick={() => api.resumeJob(current.id)}>Resume</Button>
                )}
              </div>
              <Logs job={current} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

