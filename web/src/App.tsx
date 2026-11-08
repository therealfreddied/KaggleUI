import { useQuery } from "@tanstack/react-query";
import { Moon, Sun } from "lucide-react";
import { lazy, Suspense } from "react";
import { BrowserRouter, Navigate, Route, Routes, useLocation } from "react-router";

import { ErrorBoundary } from "./components/ErrorBoundary";

import { Brand, Shell } from "./components/Shell";
import { Button, Empty } from "./components/ui";
import { CommunityLinks } from "./components/CommunityLinks";
import { api } from "./lib/api";
import { LiveProvider, useLive } from "./lib/live";
import { useTheme } from "./lib/theme";
const Launch = lazy(() => import("./pages/Launch"));
const StoragePage = lazy(() => import("./pages/Storage"));
const Dashboard = lazy(() => import("./pages/Dashboard"));
const Imports = lazy(() => import("./pages/Imports"));
import { AccessLogin, AdminLogin } from "./pages/Login";

const Models = lazy(() => import("./pages/Models"));
const Deployments = lazy(() => import("./pages/Deployments"));
const Studio = lazy(() => import("./pages/Studio"));
const Access = lazy(() => import("./pages/Access"));
const System = lazy(() => import("./pages/System"));
const Create = lazy(() => import("./pages/Create"));
const Jobs = lazy(() => import("./pages/Jobs"));
const Workspace = lazy(() => import("./pages/Workspace"));

function Loading() {
  return (
    <div className="flex min-h-[40vh] items-center justify-center">
      <span className="inline-block h-5 w-2.5 bg-signal animate-blink" />
    </div>
  );
}

function VersionBanner({ server }: { server: string }) {
  if (server === __APP_VERSION__) return null;
  return (
    <div className="mb-5 rounded-lg border border-tight/40 bg-tight-soft px-4 py-3 text-[13.5px] text-tight">
      The page is KaggleUI {__APP_VERSION__} but the server is still {server}. In the notebook, re-run the KaggleUI cell. It
      restarts the server.
    </div>
  );
}

function AdminApp() {
  const { overview, error, refresh } = useLive();
  const loc = useLocation();
  if (!overview) return error ? <div className="mx-auto max-w-lg p-10"><Empty title="Connection interrupted">{error.message}<Button className="mt-4" onClick={refresh}>Reconnect</Button></Empty></div> : <Loading />;
  return (
    <Shell>
      <VersionBanner server={overview.version} />
      <ErrorBoundary resetKey={loc.pathname}>
      <Suspense fallback={<Loading />}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/deploy/new" element={<Launch />} />
          <Route path="/imports" element={<Imports />} />
          <Route path="/host" element={<Navigate to="/" replace />} />
          <Route path="/models" element={<Models />} />
          <Route path="/deployments" element={<Deployments />} />
          <Route path="/storage" element={<StoragePage />} />
          <Route path="/studio" element={<Studio admin />} />
          <Route path="/studio/:id" element={<Studio admin />} />
          <Route path="/access" element={<Access />} />
          <Route path="/system" element={<System />} />
          <Route path="/create" element={<Create />} />
          <Route path="/jobs" element={<Jobs />} />
          <Route path="/workspace" element={<Workspace />} />
          <Route path="/login" element={<Navigate to="/studio" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
      </ErrorBoundary>
    </Shell>
  );
}

function GuestStudio() {
  const [theme, toggle] = useTheme();
  return (
    <div className="min-h-screen">
      <header className="flex items-center justify-between border-b border-line px-4 py-3 sm:px-6">
        <Brand small />
        <button onClick={toggle} className="rounded-md p-2 text-fg-2 hover:bg-panel-2" aria-label="Toggle theme">
          {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
        </button>
      </header>
      <main className="mx-auto w-full max-w-[1320px] px-4 py-7 sm:px-6 lg:px-8">
        <ErrorBoundary>
          <Suspense fallback={<Loading />}>
            <Studio admin={false} />
          </Suspense>
        </ErrorBoundary>
      </main>
      <footer className="workspace-footer"><CommunityLinks /></footer>
    </div>
  );
}

export default function App() {
  const session = useQuery({ queryKey: ["session"], queryFn: api.session, retry: 1 });
  if (session.isLoading) return <Loading />;
  if (session.error) return <div className="mx-auto max-w-lg px-6 py-24"><Empty title="Can't reach KaggleUI">The server may be restarting, or the notebook session may have ended.<p className="mt-2 break-words text-worst">{session.error.message}</p><Button className="mt-5" onClick={() => session.refetch()}>Try again</Button></Empty></div>;
  const s = session.data;
  const path = location.pathname;

  if (s?.compatibility?.blocked && !s.admin && path !== "/host") return <div className="mx-auto max-w-lg p-10"><Empty title="This server needs an update">{s.compatibility.message}<p className="mt-3">Ask the host to update KaggleUI.</p><a href="/host" className="mt-4 inline-block underline">Host sign-in</a></Empty></div>;

  if (s?.admin) {
    return (
      <BrowserRouter>
        <LiveProvider>
          <AdminApp />
        </LiveProvider>
      </BrowserRouter>
    );
  }
  if (path !== "/host" && !new URLSearchParams(location.search).has("bad_token")) {
    return (
      <BrowserRouter>
        <Suspense fallback={<Loading />}>
          <Routes>
            <Route path="/studio/*" element={s?.access ? <GuestStudio /> : <Navigate to={`/login?next=${encodeURIComponent(path)}`} replace />} />
            <Route path="/login" element={s?.access ? <Navigate to="/studio" replace /> : <AccessLogin />} />
            <Route path="*" element={<Navigate to={s?.access ? "/studio" : "/login"} replace />} />
          </Routes>
        </Suspense>
      </BrowserRouter>
    );
  }
  return <AdminLogin />;
}
