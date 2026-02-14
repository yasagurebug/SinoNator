"use client";

import { useCallback, useEffect, useState } from "react";
import type { RecommendResponse } from "@/lib/types";

type AnswerRow = { question_id: string; answer: "yes" | "no" };

function buildXShareUrl(title: string, videoUrl: string): string {
  const appUrl =
    typeof window === "undefined"
      ? "https://example.com/"
      : new URL("/", window.location.href).toString();
  const params = new URLSearchParams({
    text: `シノネーター：${appUrl}が選ぶ今日の１本：\n${title}\n${videoUrl}`
  });
  return `https://x.com/intent/tweet?${params.toString()}`;
}

async function requestRecommend(answers: AnswerRow[]) {
  const res = await fetch("/api/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ answers })
  });
  const json = await res.json();
  if (!res.ok) {
    const msg = json?.error?.message ?? "failed to request recommend api";
    throw new Error(msg);
  }
  return json as RecommendResponse;
}

export default function HomePage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [answers, setAnswers] = useState<AnswerRow[]>([]);
  const [response, setResponse] = useState<RecommendResponse | null>(null);

  const isFinished = Boolean(response?.is_finished);
  const currentQuestion = response?.next_question ?? null;

  const fetchWithAnswers = useCallback(async (nextAnswers: AnswerRow[]) => {
    setLoading(true);
    setError(null);
    try {
      const data = await requestRecommend(nextAnswers);
      setResponse(data);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "unknown error";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchWithAnswers([]);
  }, [fetchWithAnswers]);

  const onAnswer = async (answer: "yes" | "no") => {
    if (!currentQuestion || loading || isFinished) {
      return;
    }
    const next = [...answers, { question_id: currentQuestion.question_id, answer }];
    setAnswers(next);
    await fetchWithAnswers(next);
  };

  const onRetry = async () => {
    await fetchWithAnswers(answers);
  };

  const onRestart = async () => {
    setAnswers([]);
    await fetchWithAnswers([]);
  };

  return (
    <main className="page-shell">
      <section className="hero">
        <img
          className="orb-image"
          src="/icon.png"
          alt="配信者アイコン（差し替え用）"
          width={98}
          height={98}
        />
        <h1>シノネーター</h1>
      </section>

      <section className="board">

        {error ? (
          <div className="error-box">
            <p>通信エラー: {error}</p>
            <button onClick={onRetry} className="btn btn-retry">
              再試行
            </button>
          </div>
        ) : null}

        {!isFinished && currentQuestion ? (
          <div className="question-card">
            <h2>{currentQuestion.text}</h2>
            <div className="actions">
              <button
                className="btn btn-yes"
                onClick={() => onAnswer("yes")}
                disabled={loading}
              >
                はい
              </button>
              <button
                className="btn btn-no"
                onClick={() => onAnswer("no")}
                disabled={loading}
              >
                いいえ
              </button>
            </div>
          </div>
        ) : null}

        {isFinished && response?.recommended_archive ? (
          <div className="result-card">
            <h2>今日の1本</h2>
            <a href={response.recommended_archive.video_url} target="_blank" rel="noreferrer">
              {response.recommended_archive.title}
            </a>
            <img
              className="video-thumb"
              src={`https://i.ytimg.com/vi/${response.recommended_archive.video_id}/hqdefault.jpg`}
              alt={response.recommended_archive.title}
              loading="lazy"
            />
            <div className="result-actions">
              <a
                className="btn btn-share"
                href={buildXShareUrl(
                  response.recommended_archive.title,
                  response.recommended_archive.video_url
                )}
                target="_blank"
                rel="noreferrer"
              >
                Xでシェア
              </a>
              <button className="btn btn-retry" onClick={onRestart}>
                もう一回やる
              </button>
            </div>
          </div>
        ) : null}
      </section>
    </main>
  );
}
