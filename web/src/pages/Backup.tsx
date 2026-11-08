import { StorageSections } from "./Storage";
import { WorkspaceSections } from "./Workspace";

export default function Backup() {
  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <div>
        <div className="eyebrow">Durability</div>
        <h1 className="display mt-1 text-[30px]">Backup</h1>
        <p className="mt-1 max-w-2xl text-[13.5px] text-fg-2">
          Restore dashboard preferences, deployment definitions and storage connections on a fresh instance. Model files travel
          separately; runtime state is always rebuilt locally.
        </p>
      </div>
      <WorkspaceSections />
      <StorageSections />
    </div>
  );
}