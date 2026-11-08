#!/usr/bin/env bash
# Install the latest KaggleUI release (wheel includes the prebuilt web UI).
#   !curl -fsSL https://raw.githubusercontent.com/therealfreddied/KaggleUI/main/install.sh | bash
set -euo pipefail
REPO="therealfreddied/KaggleUI"
URL=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
  | python3 -c "import json,sys; a=[x['browser_download_url'] for x in json.load(sys.stdin)['assets'] if x['name'].endswith('.whl')]; print(a[0])")
WHEEL="${URL##*/}"
NEW_VERSION=$(echo "$WHEEL" | cut -d- -f2)
echo "Installing ${WHEEL}"
pip install -q --disable-pip-version-check "${URL}"

# If an older KaggleUI server is still running in this session, stop the app but retain its tunnel.
# Otherwise the notebook's
# already-imported (old) kaggleui.launch() would keep reusing it while it serves the new web UI.
python3 - "$NEW_VERSION" <<'EOF'
import sys

from kaggleui.config import Paths
from kaggleui.launcher import _read_state, _running, stop

new = sys.argv[1]
paths = Paths()
st = _read_state(paths)
if _running(st) and st.get("version") != new:
    print(f"Stopping the running KaggleUI ({st.get('version') or 'older version'}) so {new} starts fresh. "
          "Relaunch your models afterwards.")
    stop(str(paths.home), _keep_tunnel=True)
EOF
echo "Done. Next: import kaggleui; kaggleui.launch()"
