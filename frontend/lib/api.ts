// API client for calling the backend (FastAPI or Gradio)

import { AnalyzeRequest, AnalyzeResponse } from './types';

// Get API URL from environment variable or default to localhost
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Detect if we're using Gradio (HF Spaces) or FastAPI
const isGradio = API_URL.includes('huggingface.co') || API_URL.includes('.hf.space');

export class APIError extends Error {
  constructor(
    message: string,
    public status?: number,
    public details?: any
  ) {
    super(message);
    this.name = 'APIError';
  }
}

/**
 * Analyze text using the backend (FastAPI or Gradio)
 */
export async function analyzeText(
  text: string,
  includeExplanation: boolean = true
): Promise<AnalyzeResponse> {
  try {
    if (isGradio) {
      // Gradio API format
      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          data: [text],
        }),
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new APIError(
          error.detail || `API request failed with status ${response.status}`,
          response.status,
          error
        );
      }

      const gradioResponse = await response.json();

      // Transform Gradio response to match our expected format
      const gradioData = gradioResponse.data[0];

      // Parse probabilities from strings to numbers
      const parsePercent = (str: string) => parseFloat(str.replace('%', '')) / 100;

      return {
        text_hash: gradioData.text_hash,
        label: gradioData.label as 'Human' | 'AI' | 'Inconclusive',
        confidence: parsePercent(gradioData.confidence),
        probabilities: {
          Human: parsePercent(gradioData.probabilities.Human),
          AI: parsePercent(gradioData.probabilities.AI),
          Inconclusive: parsePercent(gradioData.probabilities.Inconclusive),
        },
        reasons: [`Classification: ${gradioData.label} (confidence: ${gradioData.confidence})`],
        model_version: gradioData.model_version,
        processing_time_ms: parseFloat(gradioData.processing_time_ms.replace('ms', '')),
        explanation: null,
        ensemble: null,
        style_analysis: null,
      };
    } else {
      // FastAPI format
      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text,
          include_explanation: includeExplanation,
        } as AnalyzeRequest),
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new APIError(
          error.detail || `API request failed with status ${response.status}`,
          response.status,
          error
        );
      }

      const data: AnalyzeResponse = await response.json();
      return data;
    }
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }

    // Network or other errors
    throw new APIError(
      'Failed to connect to the API. Make sure the backend is running.',
      undefined,
      error
    );
  }
}

/**
 * Check if the API is healthy and running
 */
export async function checkHealth(): Promise<{ status: string; model_loaded: boolean }> {
  try {
    if (isGradio) {
      // Gradio doesn't have a health endpoint, try the main page
      const response = await fetch(API_URL);
      if (response.ok) {
        return { status: 'healthy', model_loaded: true };
      }
      throw new APIError('Gradio space is not running', response.status);
    } else {
      const response = await fetch(`${API_URL}/health`);

      if (!response.ok) {
        throw new APIError(`Health check failed with status ${response.status}`, response.status);
      }

      return await response.json();
    }
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }

    throw new APIError('Backend is not reachable', undefined, error);
  }
}

/**
 * Run adversarial testing on text
 */
export async function runAdversarialTest(
  text: string
): Promise<{
  text_hash: string;
  baseline: {
    label: string;
    confidence: number;
    probabilities: {
      Human: number;
      AI: number;
      Inconclusive: number;
    };
  };
  adversarial_examples: Array<{
    text: string;
    attack_type: string;
    label: string;
    confidence: number;
    label_flipped: boolean;
    confidence_change: number;
    probabilities: {
      Human: number;
      AI: number;
      Inconclusive: number;
    };
  }>;
  summary: {
    total_tests: number;
    label_flips: number;
    flip_rate: number;
    robustness_score: number;
    avg_confidence_change: number;
    max_confidence_change: number;
    stability: string;
  };
}> {
  if (isGradio) {
    throw new APIError(
      'Adversarial testing is not available with Gradio backend. Please use the full FastAPI backend locally for this feature.',
      503
    );
  }

  try {
    const response = await fetch(`${API_URL}/adversarial-test`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text,
        include_explanation: false,
      } as AnalyzeRequest),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new APIError(
        error.detail || `Adversarial test failed with status ${response.status}`,
        response.status,
        error
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }

    throw new APIError(
      'Failed to run adversarial test. Make sure the backend is running.',
      undefined,
      error
    );
  }
}
