#!/usr/bin/env bash
set -euo pipefail

# Build and deploy the Next.js app locally, then upload the prebuilt output to Vercel.
# Usage:
#   scripts/deploy-prebuilt.sh [BACKEND_URL] [preview|prod]
# Examples:
#   scripts/deploy-prebuilt.sh https://sentimizer.onrender.com prod
#   scripts/deploy-prebuilt.sh https://sentimizer.onrender.com preview

BACKEND_URL="${1:-https://sentimizer.onrender.com}"
CHANNEL="${2:-prod}"

# cd to repo root (this script lives in scripts/)
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_DIR="$ROOT_DIR/sentiment-sphere"

if ! command -v vercel >/dev/null 2>&1; then
  echo "[!] vercel CLI not found. Install with: npm i -g vercel" >&2
  exit 1
fi

# Prefer pnpm if available, fallback to npm
if command -v pnpm >/dev/null 2>&1; then
  PKG_MGR=pnpm
else
  PKG_MGR=npm
fi

echo "[i] Using backend URL: $BACKEND_URL"
echo "[i] Deployment channel: $CHANNEL"

cd "$APP_DIR"

# Ensure project is linked. If not, vercel will prompt interactively.
if [ ! -f .vercel/project.json ]; then
  echo "[i] Project not linked yet. Running 'vercel link'..."
  vercel link
fi

echo "[i] Pulling Vercel project settings (production env)"
vercel pull --yes --environment=production

# Ensure NEXT_PUBLIC_API_URL is set in .env.local
touch .env.local
if grep -q '^NEXT_PUBLIC_API_URL=' .env.local; then
  sed -i.bak "s#^NEXT_PUBLIC_API_URL=.*#NEXT_PUBLIC_API_URL=$BACKEND_URL#g" .env.local && rm -f .env.local.bak
else
  echo "NEXT_PUBLIC_API_URL=$BACKEND_URL" >> .env.local
fi

echo "[i] Installing dependencies with $PKG_MGR"
if [ "$PKG_MGR" = "pnpm" ]; then
  pnpm install
else
  npm install
fi

echo "[i] Building locally with increased Node heap"
export NODE_OPTIONS=--max-old-space-size=8192
vercel build

echo "[i] Deploying prebuilt output to Vercel ($CHANNEL)"
if [ "$CHANNEL" = "prod" ]; then
  vercel deploy --prebuilt --prod
else
  vercel deploy --prebuilt
fi

