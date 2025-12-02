'use client';

import { AnalyzeResponse } from '@/lib/types';

interface ResultCardProps {
  result: AnalyzeResponse;
}

export default function ResultCard({ result }: ResultCardProps) {
  const { label, confidence, probabilities, reasons, processing_time_ms } = result;

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
            <div className={`inline-block px-4 py-2 shadow-sm rounded-full border-2 font-bold text-lg ${getLabelColor()}`}>
              {label}
            </div>
            <p className="text-sm text-gray-600 mt-1">
              Confidence: {(confidence * 100).toFixed(1)}%
            </p>
          </div>
        </div>
        <div className="text-xs text-gray-400">
          {processing_time_ms.toFixed(0)}ms
        </div>
      </div>

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
                className={`h-2.5 rounded-full transition-all duration-500 ${key === 'Human' ? 'bg-green-700' :
                  key === 'AI' ? 'bg-red-700' :
                    'bg-yellow-700'
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
                <span className="text-blue-400 mt-0.5">•</span>
                <span>{reason}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
