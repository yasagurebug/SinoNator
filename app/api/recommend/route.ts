import { NextResponse } from "next/server";
import { recommend } from "@/lib/recommend";
import type { RecommendRequest } from "@/lib/types";

export const runtime = "nodejs";

export async function POST(req: Request) {
  try {
    const body = (await req.json()) as Partial<RecommendRequest>;
    if (!body || !Array.isArray(body.answers)) {
      return NextResponse.json(
        { error: { code: "INVALID_REQUEST", message: "answers must be an array" } },
        { status: 400 }
      );
    }
    const response = recommend({ set_id: body.set_id, answers: body.answers });
    return NextResponse.json(response, { status: 200 });
  } catch (error) {
    const message = error instanceof Error ? error.message : "unknown error";
    if (message.startsWith("duplicate question_id")) {
      return NextResponse.json({ error: { code: "INVALID_REQUEST", message } }, { status: 400 });
    }
    if (message.startsWith("unknown question_id") || message.startsWith("invalid answer")) {
      return NextResponse.json({ error: { code: "INVALID_ANSWER", message } }, { status: 422 });
    }
    if (message.startsWith("set_id mismatch")) {
      return NextResponse.json({ error: { code: "SET_NOT_FOUND", message } }, { status: 404 });
    }
    return NextResponse.json({ error: { code: "INTERNAL_ERROR", message } }, { status: 500 });
  }
}
