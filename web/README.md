# Matcha UI

Standalone Next.js frontend for running Matcha docking jobs and visualizing denoising trajectories.

## Development

1. Start the Matcha backend from the repository root:

```bash
uv run matcha-server --host 127.0.0.1 --port 8899
```

2. Start the frontend:

```bash
cd web
cp .env.example .env.local
npm install
npm run dev
```

3. Open `http://127.0.0.1:3000/matcha`.
