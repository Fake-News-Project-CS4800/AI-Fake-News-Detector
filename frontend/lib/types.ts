// TypeScript types for the API

export interface AnalyzeRequest {
  text: string;
  include_explanation: boolean;
}

export interface AnalyzeResponse {
  text_hash: string;
  label: 'Human' | 'AI' | 'Inconclusive';
  confidence: number;
  probabilities: {
    Human: number;
    AI: number;
    Inconclusive: number;
  };
  reasons: string[];
  model_version: string;
  processing_time_ms: number;
  explanation?: {
    top_tokens: Array<[string, number]>;
    full_analysis: any;
  } | null;
}

export interface Message {
  id: string;
  type: 'user' | 'bot';
  content: string;
  result?: AnalyzeResponse;
  timestamp: Date;
}
