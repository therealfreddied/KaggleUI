import { useState } from "react";
import { useLocation, useNavigate } from "react-router";

import { ModelDetail } from "../components/ModelDetail";
import { ModelImporter } from "../components/ModelImporter";
import { Drawer, SectionTitle } from "../components/ui";
import type { CatalogItem } from "../lib/types";

export default function Imports() {
  const location = useLocation();
  const nav = useNavigate();
  const [open, setOpen] = useState<CatalogItem | null>(null);
  return <div>
    <SectionTitle title="Imports" description="Your own checkpoints, kept separate from the public catalog." />
    <div className="max-w-3xl space-y-8">
      <ModelImporter initialUrl={(location.state as { url?: string } | null)?.url} onCustomize={setOpen} />
    </div>
    <Drawer open={!!open} onClose={() => setOpen(null)} title="Customize import">
      {open && <ModelDetail key={open.key} item={open} actionLabel="Continue to deployment" onAction={choice => nav("/deploy/new", { state: { choice } })} />}
    </Drawer>
  </div>;
}
