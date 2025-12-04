'use client';

import { AnalyzeResponse } from '@/lib/types';
import StyleAnalysisModal from './StyleAnalysisModal';
import AdversarialTestingModal from './AdversarialTestingModal';
import { useState } from 'react';

interface ResultCardProps {
  result: AnalyzeResponse;
  originalText?: string;
}

export default function ResultCard({ result, originalText }: ResultCardProps) {
  const { label, confidence, probabilities, reasons, processing_time_ms, ensemble } = result;
  const [isStyleModalOpen, setIsStyleModalOpen] = useState(false);
  const [isAdversarialModalOpen, setIsAdversarialModalOpen] = useState(false);

  // Color coding based on label
  const getLabelColor = () => {
    switch (label) {
      case 'AI':
        return 'bg-red-100 text-red-800 border-red-300';
      case 'Human':
        return 'bg-green-100 text-green-800 border-green-300';
      case 'Inconclusive':
        return 'bg-yellow-100 text-yellow-800 border-yellow-300';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  const getIcon = () => {
    switch (label) {
      case 'AI':
        return '🤖';
      case 'Human':
        return '✍️';
      case 'Inconclusive':
        return '❓';
      default:
        return '📝';
    }
  };

  return (
    <div className="space-y-4 p-6 bg-white rounded-lg shadow-md border border-gray-200">
      {/* Main Label */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-4xl">{getIcon()}</span>
          <div>
            <div className={`inline-block px-4 py-2 rounded-full border-2 font-bold text-lg ${getLabelColor()}`}>
              {label}
            </div>
            <p className="text-sm text-gray-500 mt-1">
              Confidence: {(confidence * 100).toFixed(1)}%
            </p>
          </div>
        </div>
        <div className="text-xs text-gray-400">
          {processing_time_ms.toFixed(0)}ms
        </div>
      </div>

      {/* Ensemble Information */}
      {ensemble && (
        <div className="border-t border-gray-200 pt-4">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-lg">🔬</span>
            <h4 className="text-sm font-semibold text-gray-700">Ensemble Analysis</h4>
            <span className={`px-2 py-1 rounded text-xs font-medium ${
              ensemble.models_agree
                ? 'bg-green-100 text-green-700'
                : 'bg-orange-100 text-orange-700'
            }`}>
              {ensemble.agreement_level}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3 mb-3">
            {/* RoBERTa Prediction */}
            <div className="bg-blue-50 rounded-lg p-3 border border-blue-200">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-semibold text-blue-800">RoBERTa Model</span>
              </div>
              <div className="text-sm font-bold text-blue-900">{ensemble.roberta.label}</div>
              <div className="text-xs text-blue-700">
                {(ensemble.roberta.confidence * 100).toFixed(1)}% confidence
              </div>
            </div>

            {/* Gemini Prediction */}
            <div className="bg-purple-50 rounded-lg p-3 border border-purple-200">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-semibold text-purple-800">Gemini AI</span>
              </div>
              <div className="text-sm font-bold text-purple-900">{ensemble.gemini.label}</div>
              <div className="text-xs text-purple-700">
                {(ensemble.gemini.confidence * 100).toFixed(1)}% confidence
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Probability Bars */}
      <div className="space-y-2">
        <h4 className="text-sm font-semibold text-gray-700">Probabilities:</h4>
        {Object.entries(probabilities).map(([key, value]) => (
          <div key={key} className="space-y-1">
            <div className="flex justify-between text-sm">
              <span className="text-gray-600">{key}</span>
              <span className="font-medium text-gray-900">{(value * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div
                className={`h-2.5 rounded-full transition-all duration-500 ${
                  key === 'Human' ? 'bg-green-500' :
                  key === 'AI' ? 'bg-red-500' :
                  'bg-yellow-500'
                }`}
                style={{ width: `${value * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Reasons/Explanation */}
      {reasons && reasons.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-semibold text-gray-700">Analysis:</h4>
          <ul className="space-y-1">
            {reasons.map((reason, idx) => (
              <li key={idx} className="text-sm text-gray-600 flex items-start gap-2">
                <span className="text-blue-500 mt-0.5">•</span>
                <span>{reason}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Adversarial Testing Button */}
      {originalText && (
        <>
          <div className="border-t border-gray-200 pt-4 mt-4">
            <button
              onClick={() => setIsAdversarialModalOpen(true)}
              className="w-full flex items-center justify-between p-4 bg-gradient-to-r from-orange-50 to-red-50 hover:from-orange-100 hover:to-red-100 rounded-lg border-2 border-orange-200 hover:border-orange-300 transition-all"
            >
              <div className="flex items-center gap-3">
                <span className="text-2xl">🛡️</span>
                <div className="text-left">
                  <div className="font-semibold text-gray-800">Test Model Robustness</div>
                  <div className="text-xs text-gray-600">Run adversarial attacks and measure stability</div>
                </div>
              </div>
              <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </button>
          </div>

          {/* Adversarial Testing Modal */}
          <AdversarialTestingModal
            isOpen={isAdversarialModalOpen}
            onClose={() => setIsAdversarialModalOpen(false)}
            originalText={originalText}
          />
        </>
      )}

      {/* Style Analysis Button */}
      {result.style_analysis && (
        <>
          <div className="border-t border-gray-200 pt-4 mt-4">
            <button
              onClick={() => setIsStyleModalOpen(true)}
              className="w-full flex items-center justify-between p-4 bg-gradient-to-r from-blue-50 to-purple-50 hover:from-blue-100 hover:to-purple-100 rounded-lg border-2 border-blue-200 hover:border-blue-300 transition-all"
            >
              <div className="flex items-center gap-3">
                <span className="text-2xl">📊</span>
                <div className="text-left">
                  <div className="font-semibold text-gray-800">View Writing Style Analysis</div>
                  <div className="text-xs text-gray-600">Detailed linguistic metrics and patterns</div>
                </div>
              </div>
              <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </button>
          </div>

          {/* Style Analysis Modal */}
          <StyleAnalysisModal
            styleAnalysis={result.style_analysis}
            isOpen={isStyleModalOpen}
            onClose={() => setIsStyleModalOpen(false)}
          />
        </>
      )}
    </div>
  );
}
