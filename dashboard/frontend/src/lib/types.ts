export interface Fact {
  id: string;
  text: string;
  category: string;
  temporal_class: string;
  confidence: string;
  decay_score: number;
  session_count: number;
  importance: number;
  scope: string;
  times_recalled: number;
  times_applied: number;
  recall_utility: number;
  failure_probability: number;
  created_at: string;
  last_seen_at: string;
}

export interface Decision {
  id: string;
  text: string;
  temporal_class: string;
  decay_score: number;
  session_count: number;
  importance: number;
  scope: string;
  times_recalled: number;
  times_applied: number;
  created_at: string;
  last_seen_at: string;
}

export interface Entity {
  id: string;
  name: string;
  entity_type: string;
  session_count: number;
  scope: string;
  first_seen_at: string;
  last_seen_at: string;
}

export interface Relationship {
  id: string;
  from_entity: string;
  to_entity: string;
  rel_type: string;
  description: string;
  strength: number;
  session_count: number;
  scope: string;
  created_at: string;
  last_seen_at: string;
}

export interface Guardrail {
  id: string;
  warning: string;
  rationale: string;
  consequence: string;
  file_paths: string[] | null;
  importance: number;
  scope: string;
  session_count: number;
  times_recalled: number;
  times_applied: number;
  created_at: string;
  last_seen_at: string;
}

export interface Procedure {
  id: string;
  task_description: string;
  steps: string;
  file_paths: string[] | null;
  importance: number;
  scope: string;
  temporal_class: string;
  decay_score: number;
  session_count: number;
  times_recalled: number;
  times_applied: number;
  created_at: string;
  last_seen_at: string;
}

export interface ErrorSolution {
  id: string;
  error_pattern: string;
  error_context: string;
  solution: string;
  file_paths: string[] | null;
  scope: string;
  confidence: string;
  times_applied: number;
  times_recalled: number;
  created_at: string;
  last_applied_at: string;
}

export interface Observation {
  id: string;
  text: string;
  proof_count: number;
  source_fact_ids: string[] | null;
  temporal_class: string;
  decay_score: number;
  session_count: number;
  scope: string;
  importance: number;
  superseded_by: string | null;
  created_at: string;
  last_seen_at: string;
  updated_at: string;
}

export interface Session {
  id: string;
  trigger: string;
  cwd: string;
  message_count: number;
  summary: string | null;
  scope: string;
  created_at: string;
}

export interface Scope {
  scope: string;
  display_name: string;
  count: number;
}

export interface GraphNode {
  id: string;
  label: string;
  entity_type: string;
  session_count: number;
  degree: number;
  cluster: string;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  rel_type: string;
  description: string;
  strength: number;
}

export interface KnowledgeNode {
  id: string;
  label: string;
  node_type: "entity" | "fact" | "decision" | "observation" | "guardrail" | "procedure" | "error_solution" | "file";
  size_metric: number;
  metadata: Record<string, any>;
  cluster: string | null;
  scope: string;
  degree: number;
}

export interface KnowledgeEdge {
  id: string;
  source: string;
  target: string;
  edge_type: string;
  strength: number;
  metadata: Record<string, any>;
}

export interface KnowledgeGraphData {
  nodes: KnowledgeNode[];
  edges: KnowledgeEdge[];
  clusters: { id: string; label: string; node_count: number }[];
  type_counts: Record<string, number>;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: { id: string; node_type: string; text: string; score?: number }[];
}

export interface SearchResult {
  type: string;
  id: string;
  text: string;
  score?: number;
  temporal_class?: string;
  decay_score?: number;
  scope?: string;
}

export interface Stats {
  facts: { total: number; long: number; medium: number; short: number; inactive: number };
  ideas: { total: number; inactive: number };
  entities: { total: number };
  relationships: { total: number };
  decisions: { total: number };
  questions: { total: number; resolved: number };
  sessions: { total: number };
  observations: { total: number; long: number; medium: number; inactive: number };
  guardrails: { total: number };
  procedures: { total: number };
  error_solutions: { total: number };
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  limit?: number;
  offset?: number;
}
