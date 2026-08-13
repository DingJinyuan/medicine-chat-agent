import { NextRequest, NextResponse } from "next/server";

// Server-only env vars (no NEXT_PUBLIC_ prefix) so the backend API key never
// reaches the browser bundle -- the browser only ever talks to this route.
const BACKEND_URL = process.env.MEDISENSE_BACKEND_URL ?? "http://localhost:8000";
const API_KEY = process.env.MEDISENSE_API_KEY ?? "dev-local-key";

export async function POST(request: NextRequest) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "invalid JSON body" }, { status: 400 });
  }

  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND_URL}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
      },
      body: JSON.stringify(body),
    });
  } catch {
    return NextResponse.json({ error: "could not reach MediSense backend" }, { status: 502 });
  }

  const data = await upstream.json().catch(() => ({ error: "invalid response from backend" }));
  return NextResponse.json(data, { status: upstream.status });
}
