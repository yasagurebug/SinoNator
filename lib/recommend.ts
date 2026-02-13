import { readFileSync } from "fs";
import path from "path";
import { randomInt } from "crypto";
import type {
  ActiveSet,
  Archive,
  ArchiveTag,
  AnswerValue,
  CandidateArchive,
  Question,
  QuestionSet,
  RecommendRequest,
  RecommendResponse
} from "./types";

type DataBundle = {
  activeSet: ActiveSet;
  archives: Archive[];
  archiveTags: ArchiveTag[];
  questionSet: QuestionSet;
};

const MAX_QUESTIONS = 15;
const STAGNATION_LIMIT = 2;

function loadJson<T>(...parts: string[]): T {
  const filePath = path.join(process.cwd(), ...parts);
  return JSON.parse(readFileSync(filePath, "utf-8")) as T;
}

function loadDataBundle(requestSetId?: string): DataBundle {
  const activeSet = loadJson<ActiveSet>("public", "data", "active_set.json");
  const questionSet = loadJson<QuestionSet>("public", "data", "question_set.json");
  const archivesPayload = loadJson<{ items: Archive[] }>("public", "data", "archives.json");
  const archiveTagsPayload = loadJson<{ items: ArchiveTag[] }>("public", "data", "archive_tags.json");

  const resolvedSetId = requestSetId ?? activeSet.set_id;
  if (resolvedSetId !== questionSet.set_id) {
    throw new Error(`set_id mismatch: request=${resolvedSetId}, question_set=${questionSet.set_id}`);
  }
  return {
    activeSet: { ...activeSet, set_id: resolvedSetId },
    archives: archivesPayload.items,
    archiveTags: archiveTagsPayload.items,
    questionSet
  };
}

function validateAnswers(questionSet: QuestionSet, answers: RecommendRequest["answers"]): void {
  const seen = new Set<string>();
  const questionMap = new Map(questionSet.questions.map((q) => [q.question_id, q]));
  for (const row of answers) {
    if (seen.has(row.question_id)) {
      throw new Error(`duplicate question_id: ${row.question_id}`);
    }
    seen.add(row.question_id);
    const q = questionMap.get(row.question_id);
    if (!q) {
      throw new Error(`unknown question_id: ${row.question_id}`);
    }
    if (row.answer !== "yes" && row.answer !== "no") {
      throw new Error(`invalid answer: ${row.answer}`);
    }
  }
}

function buildTagConfidenceMap(archiveTags: ArchiveTag[]): Map<string, Map<string, number>> {
  const map = new Map<string, Map<string, number>>();
  for (const row of archiveTags) {
    if (!map.has(row.video_id)) {
      map.set(row.video_id, new Map<string, number>());
    }
    map.get(row.video_id)!.set(row.tag_id, row.confidence);
  }
  return map;
}

function randomUnitFloat(): number {
  return randomInt(0, 1_000_000_000) / 1_000_000_000;
}

function chooseNextQuestion(
  questionSet: QuestionSet,
  askedQuestionIds: Set<string>
): Question | null {
  const pool = questionSet.questions.filter((q) => q.is_active && !askedQuestionIds.has(q.question_id));
  if (pool.length === 0) {
    return null;
  }
  const fixed = pool.filter((q) => q.kind === "fixed");
  const variable = pool.filter((q) => q.kind === "variable");
  const hasVariable = variable.length > 0;
  const hasFixed = fixed.length > 0;

  let target: Question[];
  if (hasVariable && hasFixed) {
    const p = questionSet.selection.variable_weight;
    target = randomUnitFloat() < p ? variable : fixed;
  } else {
    target = hasVariable ? variable : fixed;
  }
  const index = randomInt(0, target.length);
  return target[index] ?? null;
}

function toCandidate(
  archive: Archive,
  score: number
): CandidateArchive {
  return {
    video_id: archive.video_id,
    title: archive.title,
    video_url: archive.video_url,
    score: Number(score.toFixed(4)),
    recommended_label: archive.recommended_label
  };
}

function weightedPick(
  archives: Archive[]
): Archive {
  const RECOMMENDED_WEIGHT = 2.5;
  const DEFAULT_WEIGHT = 1.0;
  const weights = archives.map((a) =>
    a.recommended_label.trim() ? RECOMMENDED_WEIGHT : DEFAULT_WEIGHT
  );
  const total = weights.reduce((s, w) => s + w, 0);
  let r = randomUnitFloat() * total;
  for (let i = 0; i < archives.length; i += 1) {
    r -= weights[i];
    if (r <= 0) {
      return archives[i];
    }
  }
  return archives[archives.length - 1];
}

export function recommend(req: RecommendRequest): RecommendResponse {
  const bundle = loadDataBundle(req.set_id);
  const { questionSet, archives, archiveTags, activeSet } = bundle;
  const answers = req.answers ?? [];
  validateAnswers(questionSet, answers);

  const questionMap = new Map(questionSet.questions.map((q) => [q.question_id, q]));
  const archiveMap = new Map(archives.map((a) => [a.video_id, a]));
  const tagMap = buildTagConfidenceMap(archiveTags);

  let candidateIds = new Set<string>(archives.map((a) => a.video_id));
  let lastNonEmptyCandidateIds = new Set(candidateIds);
  const scoreMap = new Map<string, number>(archives.map((a) => [a.video_id, 0]));
  let stagnationStreak = 0;

  for (const row of answers) {
    const prevCandidateCount = candidateIds.size;
    const q = questionMap.get(row.question_id)!;
    const answer: AnswerValue = row.answer;

    if (q.question_mode === "filter") {
      const next = new Set<string>();
      for (const id of candidateIds) {
        const tagConf = tagMap.get(id);
        const hasAny = q.target_tags.some((t) => (tagConf?.get(t) ?? 0) > 0);
        if (answer === "yes" && hasAny) {
          next.add(id);
        }
        if (answer === "no" && !hasAny) {
          next.add(id);
        }
      }
      candidateIds = next;
      if (candidateIds.size > 0) {
        lastNonEmptyCandidateIds = new Set(candidateIds);
      }
    } else {
      const effects = q.effects[answer];
      for (const id of candidateIds) {
        const base = scoreMap.get(id) ?? 0;
        let delta = 0;
        for (const e of effects) {
          const conf = tagMap.get(id)?.get(e.tag_id) ?? 0;
          delta += e.weight * conf;
        }
        scoreMap.set(id, base + delta);
      }
    }

    const currentCandidateCount = candidateIds.size;
    if (currentCandidateCount < prevCandidateCount) {
      stagnationStreak = 0;
    } else {
      stagnationStreak += 1;
    }
  }

  if (candidateIds.size === 0) {
    candidateIds = new Set(lastNonEmptyCandidateIds);
  }

  const askedCount = answers.length;
  const maxQuestionsReached = askedCount >= MAX_QUESTIONS;
  const candidateLe1 = askedCount >= questionSet.limits.min_questions && candidateIds.size <= 1;
  const stagnationReached = askedCount >= questionSet.limits.min_questions && stagnationStreak >= STAGNATION_LIMIT;
  const noMoreQuestions = questionSet.questions.filter((q) => q.is_active).length <= askedCount;
  const isFinished = candidateLe1 || maxQuestionsReached || noMoreQuestions || stagnationReached;

  const candidateArchives = [...candidateIds]
    .map((id) => toCandidate(archiveMap.get(id)!, scoreMap.get(id) ?? 0))
    .sort((a, b) => (b.score - a.score) || a.title.localeCompare(b.title, "ja"))
    .slice(0, 5);

  if (!isFinished) {
    const askedIds = new Set(answers.map((a) => a.question_id));
    const next = chooseNextQuestion(questionSet, askedIds);
    if (!next) {
      return {
        is_finished: true,
        set_id: activeSet.set_id,
        next_question: null,
        recommended_archive: candidateArchives[0],
        candidate_archives: candidateArchives,
        meta: {
          asked_count: askedCount,
          candidate_count: candidateIds.size,
          finish_reason: "no_more_questions"
        }
      };
    }
    return {
      is_finished: false,
      set_id: activeSet.set_id,
      next_question: {
        question_id: next.question_id,
        text: next.text,
        kind: next.kind,
        question_mode: next.question_mode
      },
      candidate_archives: candidateArchives,
      meta: {
        asked_count: askedCount,
        candidate_count: candidateIds.size
      }
    };
  }

  const finalPool = [...candidateIds].map((id) => archiveMap.get(id)!);
  const picked = weightedPick(finalPool);
  const pickedCandidate = toCandidate(picked, scoreMap.get(picked.video_id) ?? 0);

  return {
    is_finished: true,
    set_id: activeSet.set_id,
    next_question: null,
    recommended_archive: pickedCandidate,
    candidate_archives: candidateArchives,
    meta: {
      asked_count: askedCount,
      candidate_count: candidateIds.size,
      finish_reason: candidateLe1
        ? "candidate_le_1"
        : stagnationReached
          ? "stagnation_reached"
          : noMoreQuestions
            ? "no_more_questions"
            : "max_questions_reached"
    }
  };
}
