# Klike Interactive Lab · frontend

Interface web da demonstração do case, em React, TypeScript e Vite. Ela só consome a API em `backend/`: scores, classes, evidências e ressalvas vêm do pacote Python `klike`.

```bash
npm install
npm run dev        # http://localhost:5173, com /api encaminhado para a API local na porta 8000
npm test           # testes (Vitest e Testing Library)
npm run lint       # oxlint
npm run typecheck  # TypeScript
npm run build      # build de produção em dist/
```

Em produção, `VITE_API_URL` aponta para a API publicada, por exemplo `https://klike-interactive-lab-api.onrender.com`. Veja [`docs/interactive_lab.md`](../docs/interactive_lab.md).
