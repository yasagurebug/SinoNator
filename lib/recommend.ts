import { randomInt } from "crypto";
import type {
  AnswerValue,
  Archive,
  CandidateArchive,
  RecommendRequest,
  RecommendResponse,
  QuestionType,
  RuntimeQuestion,
  RuntimeV2,
} from "./types";
import runtimeJson from "@/public/data/runtime_v2.json";
import uiRecommendJson from "@/config/ui_recommend.json";

type UiRecommendConfig = {
  limits?: {
    min_questions?: number;
    max_questions?: number;
    finalize_candidate_size?: number;
  };
  selection?: {
    no_repeat?: boolean;
    exploration_epsilon?: number;
    temperature?: number;
  };
  stagnation_limit?: number;
  weights?: {
    recommended_label?: number;
    default?: number;
  };
  scoring?: {
    match_reward?: number;
    mismatch_penalty?: number;
    split_weight_scale?: number;
    split_weight_power?: number;
    final_temperature?: number;
    final_exploration_epsilon?: number;
    hard_candidate_boost?: number;
  };
};

type UiRecommendResolved = {
  limits: {
    min_questions: number;
    max_questions: number;
    finalize_candidate_size: number;
  };
  selection: {
    no_repeat: boolean;
    exploration_epsilon: number;
    temperature: number;
  };
  stagnation_limit: number;
  weights: {
    recommended_label: number;
    default: number;
  };
  scoring: {
    match_reward: number;
    mismatch_penalty: number;
    split_weight_scale: number;
    split_weight_power: number;
    final_temperature: number;
    final_exploration_epsilon: number;
    hard_candidate_boost: number;
  };
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function loadRuntime(requestSetId?: string): RuntimeV2 {
  const runtime = runtimeJson as RuntimeV2;
  const resolvedSetId = requestSetId ?? runtime.set_id;
  if (resolvedSetId !== runtime.set_id) {
    throw new Error(`set_id mismatch: request=${resolvedSetId}, runtime=${runtime.set_id}`);
  }
  return runtime;
}

function loadUiConfig(): UiRecommendResolved {
  const raw = uiRecommendJson as UiRecommendConfig;
  const limits = raw.limits ?? {};
  const selection = raw.selection ?? {};
  const weights = raw.weights ?? {};
  const scoring = raw.scoring ?? {};
  return {
    limits: {
      min_questions: Math.max(1, Number(limits.min_questions ?? 1)),
      max_questions: Math.min(20, Math.max(1, Number(limits.max_questions ?? 20))),
      finalize_candidate_size: Math.max(1, Number(limits.finalize_candidate_size ?? 5)),
    },
    selection: {
      no_repeat: Boolean(selection.no_repeat ?? true),
      exploration_epsilon: clamp(Number(selection.exploration_epsilon ?? 0.12), 0.0, 0.8),
      temperature: Math.max(0.05, Number(selection.temperature ?? 0.55)),
    },
    stagnation_limit: Math.max(1, Number(raw.stagnation_limit ?? 5)),
    weights: {
      recommended_label: Math.max(0.01, Number(weights.recommended_label ?? 2.5)),
      default: Math.max(0.01, Number(weights.default ?? 1.0)),
    },
    scoring: {
      match_reward: Math.max(0.01, Number(scoring.match_reward ?? 2.0)),
      mismatch_penalty: Math.max(0.01, Number(scoring.mismatch_penalty ?? 1.6)),
      split_weight_scale: clamp(Number(scoring.split_weight_scale ?? 1.2), 0.0, 4.0),
      split_weight_power: clamp(Number(scoring.split_weight_power ?? 1.0), 0.2, 3.0),
      final_temperature: Math.max(0.05, Number(scoring.final_temperature ?? 1.0)),
      final_exploration_epsilon: clamp(Number(scoring.final_exploration_epsilon ?? 0.05), 0.0, 0.5),
      hard_candidate_boost: Math.max(1.0, Number(scoring.hard_candidate_boost ?? 1.3)),
    },
  };
}

function validateAnswers(runtime: RuntimeV2, answers: RecommendRequest["answers"]): void {
  const seen = new Set<string>();
  const questionMap = new Map(runtime.questions.map((q) => [q.question_id, q]));
  for (const row of answers) {
    if (seen.has(row.question_id)) {
      throw new Error(`duplicate question_id: ${row.question_id}`);
    }
    seen.add(row.question_id);
    const q = questionMap.get(row.question_id);
    if (!q || !q.is_active) {
      throw new Error(`unknown question_id: ${row.question_id}`);
    }
    if (row.answer !== "yes" && row.answer !== "no") {
      throw new Error(`invalid answer: ${row.answer}`);
    }
  }
}

function randomUnitFloat(): number {
  return randomInt(0, 1_000_000_000) / 1_000_000_000;
}

function inferQuestionType(q: RuntimeQuestion): QuestionType {
  const raw = String(q.question_type ?? "")
    .trim()
    .toLowerCase();
  if (raw === "core" || raw === "hook" || raw === "semantic") {
    return raw;
  }
  const cid = String(q.concept_id ?? "")
    .trim()
    .toLowerCase();
  if (cid.startsWith("core_")) {
    return "core";
  }
  if (cid.startsWith("hook_")) {
    return "hook";
  }
  return "semantic";
}

function toCandidate(archive: Archive, score: number): CandidateArchive {
  return {
    video_id: archive.video_id,
    title: archive.title,
    video_url: archive.video_url,
    score: Number(score.toFixed(4)),
    recommended_label: archive.recommended_label,
  };
}

function weightedPick(
  archives: Archive[],
  cfg: UiRecommendResolved,
  fitScoreMap: Map<string, number>,
  hardCandidateIds: Set<string>
): Archive {
  const eps = cfg.scoring.final_exploration_epsilon;
  const temperature = cfg.scoring.final_temperature;
  const n = archives.length;
  const scores = archives.map((a) => fitScoreMap.get(a.video_id) ?? 0);
  const maxScore = scores.length > 0 ? Math.max(...scores) : 0;

  const modelWeights = archives.map((a, i) => {
    const score = scores[i];
    const scoreWeight = Math.exp((score - maxScore) / temperature);
    const labelWeight = a.recommended_label.trim() ? cfg.weights.recommended_label : cfg.weights.default;
    const hardBoost = hardCandidateIds.has(a.video_id) ? cfg.scoring.hard_candidate_boost : 1.0;
    return Math.max(1e-12, scoreWeight * labelWeight * hardBoost);
  });
  const modelTotal = modelWeights.reduce((s, w) => s + w, 0);
  const uniformProb = 1 / n;

  let r = randomUnitFloat();
  for (let i = 0; i < archives.length; i += 1) {
    const modelProb = modelTotal > 0 ? modelWeights[i] / modelTotal : uniformProb;
    const p = (1 - eps) * modelProb + eps * uniformProb;
    r -= p;
    if (r <= 0) {
      return archives[i];
    }
  }
  return archives[archives.length - 1];
}

function chooseWeightedQuestion(
  cfg: UiRecommendResolved,
  pool: RuntimeQuestion[],
  candidateIds: Set<string>,
  yesSetMap: Map<string, Set<string>>
): RuntimeQuestion {
  const eps = cfg.selection.exploration_epsilon;
  const temperature = cfg.selection.temperature;

  const size = candidateIds.size;
  const n = pool.length;
  const rows: Array<{ q: RuntimeQuestion; modelWeight: number }> = [];

  for (const q of pool) {
    const yesSet = yesSetMap.get(q.question_id) ?? new Set<string>();
    let yesCount = 0;
    for (const id of candidateIds) {
      if (yesSet.has(id)) {
        yesCount += 1;
      }
    }
    const noCount = size - yesCount;
    const splitNorm = size > 0 ? (2 * Math.min(yesCount, noCount)) / size : 0; // 0..1
    const deadPenalty = yesCount === 0 || noCount === 0 ? 0.2 : 1.0;
    const base = Math.exp(Math.min(20, splitNorm / temperature));
    const w = base * deadPenalty;
    rows.push({ q, modelWeight: w });
  }

  const modelTotal = rows.reduce((s, r) => s + r.modelWeight, 0);
  const uniformProb = 1 / n;
  const finalProbs = rows.map((r) => {
    const modelProb = modelTotal > 0 ? r.modelWeight / modelTotal : uniformProb;
    return (1 - eps) * modelProb + eps * uniformProb;
  });

  let r = randomUnitFloat();
  for (let i = 0; i < rows.length; i += 1) {
    r -= finalProbs[i];
    if (r <= 0) {
      return rows[i].q;
    }
  }
  return rows[rows.length - 1].q;
}

function chooseNextQuestion(
  cfg: UiRecommendResolved,
  runtime: RuntimeV2,
  askedQuestionIds: Set<string>,
  askedCoreAlready: boolean,
  candidateIds: Set<string>,
  yesSetMap: Map<string, Set<string>>
): RuntimeQuestion | null {
  const noRepeat = cfg.selection.no_repeat;
  let pool = runtime.questions.filter((q) => q.is_active && (!noRepeat || !askedQuestionIds.has(q.question_id)));
  if (askedCoreAlready) {
    pool = pool.filter((q) => inferQuestionType(q) !== "core");
  }
  if (pool.length === 0) {
    return null;
  }
  return chooseWeightedQuestion(cfg, pool, candidateIds, yesSetMap);
}

function buildFitScoreMap(
  answers: RecommendRequest["answers"],
  archiveIds: string[],
  yesSetMap: Map<string, Set<string>>,
  questionMap: Map<string, RuntimeQuestion>,
  cfg: UiRecommendResolved
): Map<string, number> {
  const scoreMap = new Map<string, number>(archiveIds.map((id) => [id, 0]));
  for (const row of answers) {
    const q = questionMap.get(row.question_id);
    if (!q) {
      continue;
    }
    const yesSet = yesSetMap.get(row.question_id);
    if (!yesSet) {
      continue;
    }
    const splitScoreRaw =
      typeof q.split_score === "number"
        ? q.split_score
        : q.yes_count + q.no_count > 0
          ? (2 * Math.min(q.yes_count, q.no_count)) / (q.yes_count + q.no_count)
          : 0;
    const splitScore = clamp(splitScoreRaw, 0, 1);
    const qWeight = 1 + cfg.scoring.split_weight_scale * Math.pow(splitScore, cfg.scoring.split_weight_power);
    const matchDelta = cfg.scoring.match_reward * qWeight;
    const mismatchDelta = -cfg.scoring.mismatch_penalty * qWeight;

    for (const id of archiveIds) {
      const inYes = yesSet.has(id);
      const matched = (row.answer === "yes" && inYes) || (row.answer === "no" && !inYes);
      const prev = scoreMap.get(id) ?? 0;
      scoreMap.set(id, prev + (matched ? matchDelta : mismatchDelta));
    }
  }
  return scoreMap;
}

export function recommend(req: RecommendRequest): RecommendResponse {
  const runtime = loadRuntime(req.set_id);
  const cfg = loadUiConfig();
  const answers = req.answers ?? [];
  validateAnswers(runtime, answers);

  const archives = runtime.archives;
  const archiveMap = new Map(archives.map((a) => [a.video_id, a]));
  const questionMap = new Map(runtime.questions.map((q) => [q.question_id, q]));
  const yesSetMap = new Map(runtime.questions.map((q) => [q.question_id, new Set(q.yes_video_ids)]));

  let candidateIds = new Set<string>(archives.map((a) => a.video_id));
  let lastNonEmptyCandidateIds = new Set(candidateIds);
  let stagnationStreak = 0;

  for (const row of answers) {
    const prevCount = candidateIds.size;
    const q = questionMap.get(row.question_id)!;
    const yesSet = yesSetMap.get(q.question_id) ?? new Set<string>();
    const next = new Set<string>();

    if (row.answer === "yes") {
      for (const id of candidateIds) {
        if (yesSet.has(id)) {
          next.add(id);
        }
      }
    } else {
      for (const id of candidateIds) {
        if (!yesSet.has(id)) {
          next.add(id);
        }
      }
    }

    candidateIds = next;
    if (candidateIds.size > 0) {
      lastNonEmptyCandidateIds = new Set(candidateIds);
    }

    if (candidateIds.size < prevCount) {
      stagnationStreak = 0;
    } else {
      stagnationStreak += 1;
    }
  }

  if (candidateIds.size === 0) {
    candidateIds = new Set(lastNonEmptyCandidateIds);
  }

  const askedCount = answers.length;
  const askedIds = new Set(answers.map((a) => a.question_id));
  const askedCoreAlready = answers.some((a) => {
    const q = questionMap.get(a.question_id);
    return q ? inferQuestionType(q) === "core" : false;
  });
  const minQuestions = cfg.limits.min_questions;
  const maxQuestions = Math.max(minQuestions, cfg.limits.max_questions);
  const finalizeCandidateSize = cfg.limits.finalize_candidate_size;
  const noRepeat = cfg.selection.no_repeat;

  const candidateLeThreshold = askedCount >= minQuestions && candidateIds.size <= finalizeCandidateSize;
  const maxQuestionsReached = askedCount >= maxQuestions;
  const remainingQuestionCount = runtime.questions.filter(
    (q) =>
      q.is_active &&
      (!noRepeat || !askedIds.has(q.question_id)) &&
      (!askedCoreAlready || inferQuestionType(q) !== "core")
  ).length;
  const noMoreQuestions = remainingQuestionCount === 0;
  const stagnationReached =
    askedCount >= minQuestions &&
    stagnationStreak >= cfg.stagnation_limit &&
    candidateIds.size <= Math.max(finalizeCandidateSize * 2, 12);
  const isFinished = candidateLeThreshold || maxQuestionsReached || noMoreQuestions || stagnationReached;

  const allArchiveIds = archives.map((a) => a.video_id);
  const fitScoreMap = buildFitScoreMap(answers, allArchiveIds, yesSetMap, questionMap, cfg);
  const candidateArchives = [...candidateIds]
    .map((id) => toCandidate(archiveMap.get(id)!, fitScoreMap.get(id) ?? 0))
    .sort((a, b) => b.score - a.score || a.title.localeCompare(b.title, "ja"))
    .slice(0, finalizeCandidateSize);

  if (!isFinished) {
    const next = chooseNextQuestion(cfg, runtime, askedIds, askedCoreAlready, candidateIds, yesSetMap);
    if (!next) {
      const finalPool = archives;
      const picked = weightedPick(finalPool, cfg, fitScoreMap, candidateIds);
      return {
        is_finished: true,
        set_id: runtime.set_id,
        next_question: null,
        recommended_archive: toCandidate(picked, fitScoreMap.get(picked.video_id) ?? 0),
        candidate_archives: candidateArchives,
        meta: {
          asked_count: askedCount,
          candidate_count: candidateIds.size,
          finish_reason: "no_more_questions",
        },
      };
    }

    return {
      is_finished: false,
      set_id: runtime.set_id,
      next_question: {
        question_id: next.question_id,
        text: next.text,
        kind: next.kind,
        question_type: inferQuestionType(next),
        question_mode: "filter",
      },
      candidate_archives: candidateArchives,
      meta: {
        asked_count: askedCount,
        candidate_count: candidateIds.size,
      },
    };
  }

  const finalPool = archives;
  const picked = weightedPick(finalPool, cfg, fitScoreMap, candidateIds);
  const pickedCandidate = toCandidate(picked, fitScoreMap.get(picked.video_id) ?? 0);

  return {
    is_finished: true,
    set_id: runtime.set_id,
    next_question: null,
    recommended_archive: pickedCandidate,
    candidate_archives: candidateArchives,
    meta: {
      asked_count: askedCount,
      candidate_count: candidateIds.size,
      finish_reason: candidateLeThreshold
        ? "candidate_le_1"
        : stagnationReached
          ? "stagnation_reached"
          : noMoreQuestions
            ? "no_more_questions"
            : "max_questions_reached",
    },
  };
}
