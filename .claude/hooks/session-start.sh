#!/bin/bash
set -euo pipefail

# Fresh remote sessions clone the repo into a new container, so local git
# config and .git/hooks never survive between sessions. Reinstall both here
# so commits keep landing under the repo owner's identity, not Claude's.

git config user.name "Pedro Nascimento de Lima"
git config user.email "5301006+pedroliman@users.noreply.github.com"

mkdir -p .git/hooks
cat > .git/hooks/commit-msg << 'HOOK'
#!/bin/bash
set -euo pipefail

# Strip any Claude co-authorship / session trailers before the commit lands.
grep -viE '^(Co-Authored-By:.*Claude|Claude-Session:)' "$1" > "$1.tmp" || true
mv "$1.tmp" "$1"

# Collapse blank lines left behind at the end of the message.
perl -i -0pe 's/\n+\z/\n/' "$1" 2>/dev/null || true
HOOK
chmod +x .git/hooks/commit-msg
