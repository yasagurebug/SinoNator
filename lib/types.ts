export type AnswerValue = "yes" | "no";

export type QuestionKind = "fixed" | "variable";
export type QuestionType = "core" | "hook" | "semantic";

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

export type RuntimeQuestion = {
  question_id: string;
  kind: QuestionKind;
  question_type: QuestionType;
  is_active: boolean;
  text: string;
  concept_id: string | null;
  yes_video_ids: string[];
  yes_count: number;
  no_count: number;
  split_score?: number;
};

export type RuntimeV2 = {
  set_id: string;
  name: string;
  generated_at: string;
  archives: Archive[];
  questions: RuntimeQuestion[];
  meta: {
    version: string;
    fixed_count: number;
    variable_count: number;
    source_selected_path: string;
    source_questions_path: string;
    source_archives_path: string;
  };
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
    question_type: QuestionType;
    question_mode: "filter";
  };
  candidate_archives: CandidateArchive[];
  recommended_archive?: CandidateArchive;
  meta: {
    asked_count: number;
    candidate_count: number;
    finish_reason?: "candidate_le_1" | "max_questions_reached" | "no_more_questions" | "stagnation_reached";
  };
};
