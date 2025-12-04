'use client';

import { useState, useEffect } from 'react';
import { runAdversarialTest, APIError } from '@/lib/api';

interface AdversarialTestingModalProps {
  isOpen: boolean;
  onClose: () => void;
  originalText: string;
}

export default function AdversarialTestingModal({
  isOpen,
  onClose,
  originalText
}: AdversarialTestingModalProps) {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen && !results && !loading) {
      runTests();
    }
  }, [isOpen]);

  const runTests = async () => {
    setLoading(true);
    setError(null);
    try {
      const testResults = await runAdversarialTest(originalText);
      setResults(testResults);
    } catch (err) {
      setError(err instanceof APIError ? err.message : 'Failed to run adversarial tests');
      console.error('Adversarial testing error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getLabelColor = (label: string) => {
    switch (label) {
      case 'AI':
        return 'text-red-700';
      case 'Human':
        return 'text-green-700';
      case 'Inconclusive':
        return 'text-yellow-700';
      default:
        return 'text-gray-700';
    }
  };

  const getLabelBgColor = (label: string) => {
    switch (label) {
      case 'AI':
        return 'bg-red-100 border-red-300';
      case 'Human':
        return 'bg-green-100 border-green-300';
      case 'Inconclusive':
        return 'bg-yellow-100 border-yellow-300';
      default:
        return 'bg-gray-100 border-gray-300';
    }
  };

  const getRobustnessColor = (score: number) => {
    if (score >= 0.9) return 'text-green-700';
    if (score >= 0.7) return 'text-blue-700';
    if (score >= 0.5) return 'text-yellow-700';
    return 'text-red-700';
  };

  const getRobustnessBg = (score: number) => {
    if (score >= 0.9) return 'bg-green-100 border-green-300';
    if (score >= 0.7) return 'bg-blue-100 border-blue-300';
    if (score >= 0.5) return 'bg-yellow-100 border-yellow-300';
    return 'bg-red-100 border-red-300';
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-lg shadow-xl max-w-6xl w-full max-h-[90vh] overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="sticky top-0 bg-gradient-to-r from-orange-600 to-red-600 text-white p-6 rounded-t-lg">
          <div className="flex justify-between items-center">
            <div>
              <h2 className="text-2xl font-bold">🛡️ Model Robustness Testing</h2>
              <p className="text-sm text-orange-100 mt-1">
                Testing model stability against realistic writing variations (typos, punctuation, fillers)
              </p>
            </div>
            <button
              onClick={onClose}
              className="text-white hover:text-gray-200 transition-colors"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        <div className="overflow-y-auto p-6">
          {loading && (
            <div className="flex flex-col items-center justify-center py-12">
              <div className="w-16 h-16 border-4 border-orange-200 border-t-orange-600 rounded-full animate-spin mb-4" />
              <p className="text-gray-600">Running robustness tests...</p>
              <p className="text-sm text-gray-500 mt-1">Testing 20+ realistic text variations</p>
            </div>
          )}

          {error && (
            <div className="bg-red-50 border-2 border-red-300 rounded-lg p-4">
              <div className="flex items-start gap-2">
                <span className="text-2xl">❌</span>
                <div>
                  <h3 className="font-semibold text-red-800">Testing Failed</h3>
                  <p className="text-sm text-red-700 mt-1">{error}</p>
                </div>
              </div>
            </div>
          )}

          {results && (
            <div className="space-y-6">
              {/* Robustness Score Summary */}
              <div className={`rounded-lg p-6 border-2 ${getRobustnessBg(results.summary.robustness_score)}`}>
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-bold text-gray-800">Model Robustness Score</h3>
                    <p className="text-sm text-gray-600 mt-1">
                      Measures prediction stability across adversarial perturbations
                    </p>
                  </div>
                  <div className="text-right">
                    <div className={`text-4xl font-bold ${getRobustnessColor(results.summary.robustness_score)}`}>
                      {(results.summary.robustness_score * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm font-medium text-gray-700 mt-1">
                      {results.summary.stability}
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-4 gap-4">
                  <div className="bg-white rounded-lg p-3 border border-gray-200">
                    <div className="text-xs text-gray-500 mb-1">Total Tests</div>
                    <div className="text-2xl font-bold text-gray-800">{results.summary.total_tests}</div>
                  </div>

                  <div className="bg-white rounded-lg p-3 border border-gray-200">
                    <div className="text-xs text-gray-500 mb-1">Label Flips</div>
                    <div className="text-2xl font-bold text-red-600">{results.summary.label_flips}</div>
                  </div>

                  <div className="bg-white rounded-lg p-3 border border-gray-200">
                    <div className="text-xs text-gray-500 mb-1">Flip Rate</div>
                    <div className="text-2xl font-bold text-orange-600">
                      {(results.summary.flip_rate * 100).toFixed(1)}%
                    </div>
                  </div>

                  <div className="bg-white rounded-lg p-3 border border-gray-200">
                    <div className="text-xs text-gray-500 mb-1">Avg Δ Conf</div>
                    <div className="text-2xl font-bold text-blue-600">
                      {(results.summary.avg_confidence_change * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>

              {/* Baseline Prediction */}
              <div className="bg-blue-50 rounded-lg p-4 border-2 border-blue-300">
                <h3 className="text-sm font-bold text-blue-900 mb-3">📍 Baseline Prediction (Original Text)</h3>
                <div className="flex items-center gap-4">
                  <div className={`px-4 py-2 rounded-full border-2 font-bold ${getLabelBgColor(results.baseline.label)}`}>
                    <span className={getLabelColor(results.baseline.label)}>{results.baseline.label}</span>
                  </div>
                  <div className="text-sm text-gray-700">
                    <strong>{(results.baseline.confidence * 100).toFixed(1)}%</strong> confidence
                  </div>
                </div>
              </div>

              {/* Robustness Test Examples */}
              <div>
                <h3 className="text-lg font-bold text-gray-800 mb-4">🎯 Test Results - Realistic Variations</h3>

                <div className="space-y-3">
                  {results.adversarial_examples.map((example: any, idx: number) => (
                    <div
                      key={idx}
                      className={`rounded-lg p-4 border-2 ${
                        example.label_flipped
                          ? 'bg-red-50 border-red-300'
                          : 'bg-gray-50 border-gray-200'
                      }`}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-mono font-semibold text-gray-600 bg-white px-2 py-1 rounded border">
                            #{idx + 1}
                          </span>
                          <span className="text-xs font-semibold text-purple-700 bg-purple-100 px-2 py-1 rounded">
                            {example.attack_type}
                          </span>
                          {example.label_flipped && (
                            <span className="text-xs font-bold text-red-700 bg-red-100 px-2 py-1 rounded animate-pulse">
                              ⚠️ LABEL FLIPPED
                            </span>
                          )}
                        </div>

                        <div className="flex items-center gap-3">
                          <div className={`px-3 py-1 rounded-full text-sm font-bold border ${getLabelBgColor(example.label)}`}>
                            <span className={getLabelColor(example.label)}>{example.label}</span>
                          </div>
                          <div className="text-sm text-gray-600">
                            {(example.confidence * 100).toFixed(1)}%
                          </div>
                        </div>
                      </div>

                      <div className="bg-white rounded p-3 border border-gray-200 mb-2">
                        <p className="text-sm text-gray-700 font-mono break-words">
                          {example.text.length > 200 ? example.text.substring(0, 200) + '...' : example.text}
                        </p>
                      </div>

                      <div className="flex items-center gap-4 text-xs">
                        <div className="flex items-center gap-1">
                          <span className="text-gray-500">Confidence Change:</span>
                          <span className={`font-bold ${example.confidence_change > 0.1 ? 'text-orange-600' : 'text-gray-600'}`}>
                            {(example.confidence_change * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Educational Info */}
              <div className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg p-6 border-2 border-blue-200">
                <h3 className="text-lg font-bold text-gray-800 mb-3">📚 Understanding Robustness Testing</h3>

                <div className="space-y-3 text-sm text-gray-700">
                  <div className="bg-white rounded-lg p-3 border border-blue-200">
                    <h4 className="font-semibold text-blue-800 mb-1">What is model robustness?</h4>
                    <p>
                      Model robustness measures how stable predictions are when text is slightly modified in realistic ways.
                      A robust model should give similar predictions despite typos, casual punctuation, or natural writing variations.
                    </p>
                  </div>

                  <div className="bg-white rounded-lg p-3 border border-blue-200">
                    <h4 className="font-semibold text-blue-800 mb-1">Why does it matter?</h4>
                    <p>
                      Real-world text isn't perfect - people make typos, use informal punctuation, add filler words, and write casually.
                      A production-ready model must handle these natural variations reliably. Low robustness means predictions could
                      flip based on minor typos, making the model unreliable for real-world use.
                    </p>
                  </div>

                  <div className="bg-white rounded-lg p-3 border border-blue-200">
                    <h4 className="font-semibold text-blue-800 mb-1">Realistic attack types tested:</h4>
                    <ul className="list-disc list-inside mt-2 space-y-1 text-xs">
                      <li><strong>Natural typos:</strong> common misspellings (teh→the), doubled letters, missing letters</li>
                      <li><strong>Punctuation variations:</strong> extra commas/periods, missing punctuation, ellipses (...)</li>
                      <li><strong>Whitespace errors:</strong> extra spaces, missing spaces after punctuation</li>
                      <li><strong>Filler words:</strong> adding conversational fillers like "actually", "basically", "you know"</li>
                      <li><strong>Contractions:</strong> converting between "can't" ↔ "cannot", "it's" ↔ "it is"</li>
                      <li><strong>Capitalization:</strong> all lowercase writing, missing initial capitals</li>
                      <li><strong>Article changes:</strong> adding/removing "a", "an", "the"</li>
                    </ul>
                    <p className="text-xs text-gray-600 mt-2 italic">
                      Note: These tests simulate realistic human writing variations, not adversarial attacks like l33t speak.
                    </p>
                  </div>

                  <div className="bg-white rounded-lg p-3 border border-blue-200">
                    <h4 className="font-semibold text-blue-800 mb-1">Interpreting the score:</h4>
                    <p className="text-xs">
                      <strong>90-100%:</strong> Very Stable - Excellent robustness, production-ready<br />
                      <strong>70-90%:</strong> Stable - Good robustness with minor vulnerabilities<br />
                      <strong>50-70%:</strong> Moderately Stable - Some susceptibility to attacks<br />
                      <strong>&lt;50%:</strong> Unstable - Highly vulnerable, needs improvement
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
