#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

(cd frontend && pnpm i && pnpm run build)

(cd node && pnpm i && pnpm exec tsc --noEmit)

cd node
exec pnpm start
