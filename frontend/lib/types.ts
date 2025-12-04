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
  ensemble?: {
    label: string;
    confidence: number;
    agreement_level: string;
    models_agree: boolean;
    roberta: {
      label: string;
      confidence: number;
      probabilities: {
        Human: number;
        AI: number;
        Inconclusive: number;
      };
    };
    gemini: {
      label: string;
      confidence: number;
      reasoning: string;
    };
  } | null;
  style_analysis?: {
    lexical_diversity: {
      ttr: number;
      unique_words: number;
      total_words: number;
      interpretation: string;
    };
    perplexity_proxy: {
      entropy: number;
      normalized_score: number;
      predictability: string;
    };
    sentence_complexity: {
      avg_sentence_length: number;
      std_sentence_length: number;
      total_sentences: number;
      complex_sentences: number;
      complexity_interpretation: string;
    };
    readability: {
      flesch_reading_ease: number;
      grade_level: number;
      interpretation: string;
    };
    vocabulary_richness: {
      avg_word_length: number;
      long_words_count: number;
      long_word_ratio: number;
      interpretation: string;
    };
    punctuation_patterns: {
      commas: number;
      periods: number;
      exclamations: number;
      questions: number;
      semicolons: number;
      colons: number;
      punctuation_density: number;
      interpretation: string;
    };
    text_statistics: {
      character_count: number;
      word_count: number;
      sentence_count: number;
      avg_words_per_sentence: number;
    };
  } | null;
}

export interface Message {
  id: string;
  type: 'user' | 'bot';
  content: string;
  result?: AnalyzeResponse;
  originalText?: string;
  timestamp: Date;
}
