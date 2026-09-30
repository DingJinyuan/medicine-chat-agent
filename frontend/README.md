# Frontend

MediSense / Clinical AI 的前端：Next.js 16 (App Router) + TypeScript + Tailwind + shadcn/ui + framer-motion。

## 路由

| 路径 | 说明 |
|---|---|
| `/` | 首页选择界面（专家版 / 患者版两个入口） |
| `/patient` | 患者版 MediSense（分诊安全问答） |
| `/expert` | 专家版 Clinical AI（多 Agent 临床 RAG + 事实核查） |
| `/api/chat` | 代理转发到后端 `/api/chat` |

## 开发

```bash
npm install   # 下载依赖（node_modules）
npm run dev   # 启动 http://localhost:3000
npm run build # 生产构建
```

## 环境变量（`.env.local`）

```bash
MEDISENSE_BACKEND_URL=http://localhost:8000   # 后端地址
MEDISENSE_API_KEY=medisense-secret-2026       # 必须 = 后端 .env 的 APP_API_KEY
```

`MEDISENSE_API_KEY` 与后端 `.env` 的 `APP_API_KEY` 不一致时，`/api/chat` 会返回 401。
