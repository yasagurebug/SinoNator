export type AnswerValue = "yes" | "no";

export type QuestionKind = "fixed" | "variable";
export type QuestionMode = "filter" | "score";

export type Archive = {
  video_id: string;
  video_url: string;
  title: string;
  start_time: string;
  type_raw: string;
  is_guerrilla: boolean;
  recommended_label: string;
  summary_ai: string;
  has_summary: boolean;
};

export type ArchiveTag = {
  video_id: string;
  tag_id: string;
  confidence: number;
  source: string;
};

export type FilterQuestion = {
  question_id: string;
  kind: QuestionKind;
  is_active: boolean;
  question_mode: "filter";
  text: string;
  target_tags: string[];
  answers: {
    yes: { action: "filter_in" };
    no: { action: "filter_out" };
  };
};

export type ScoreEffect = { tag_id: string; weight: number };

export type ScoreQuestion = {
  question_id: string;
  kind: QuestionKind;
  is_active: boolean;
  question_mode: "score";
  text: string;
  effects: {
    yes: ScoreEffect[];
    no: ScoreEffect[];
  };
};

export type Question = FilterQuestion | ScoreQuestion;

export type QuestionSet = {
  set_id: string;
  name: string;
  generated_at: string;
  tone: "casual";
  limits: {
    min_questions: number;
    max_questions: number;
    finalize_candidate_size: number;
  };
  selection: {
    variable_weight: number;
    fixed_weight: number;
    no_repeat: boolean;
  };
  questions: Question[];
};

export type ActiveSet = {
  set_id: string;
  updated_at: string;
};

export type RecommendRequest = {
  set_id?: string;
  answers: Array<{ question_id: string; answer: AnswerValue }>;
};

export type CandidateArchive = {
  video_id: string;
  title: string;
  video_url: string;
  score: number;
  recommended_label: string;
};

export type RecommendResponse = {
  is_finished: boolean;
  set_id: string;
  next_question: null | {
    question_id: string;
    text: string;
    kind: QuestionKind;
    question_mode: QuestionMode;
  };
  candidate_archives: CandidateArchive[];
  recommended_archive?: CandidateArchive;
  meta: {
    asked_count: number;
    candidate_count: number;
    finish_reason?: "candidate_le_1" | "max_questions_reached" | "no_more_questions" | "stagnation_reached";
  };
};
