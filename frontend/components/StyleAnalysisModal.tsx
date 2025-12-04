'use client';

import { AnalyzeResponse } from '@/lib/types';
import { useState, useEffect } from 'react';

interface StyleAnalysisModalProps {
  styleAnalysis: NonNullable<AnalyzeResponse['style_analysis']>;
  isOpen: boolean;
  onClose: () => void;
}

export default function StyleAnalysisModal({ styleAnalysis, isOpen, onClose }: StyleAnalysisModalProps) {
  // Close modal on ESC key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }
    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const MetricBar = ({ value, max, label, color }: { value: number; max: number; label: string; color: string }) => {
    const percentage = Math.min((value / max) * 100, 100);

    return (
      <div className="space-y-1">
        <div className="flex justify-between text-xs">
          <span className="text-gray-600">{label}</span>
          <span className="font-medium text-gray-900">{value.toFixed(2)}</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className={`h-2 rounded-full transition-all duration-500 ${color}`}
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>
    );
  };

  const SectionCard = ({
    icon,
    title,
    children,
    explanation
  }: {
    icon: string;
    title: string;
    children: React.ReactNode;
    explanation: string;
  }) => (
    <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
      <div className="flex items-start gap-2 mb-3">
        <span className="text-xl">{icon}</span>
        <div className="flex-1">
          <h5 className="text-sm font-semibold text-gray-700">{title}</h5>
          <p className="text-xs text-gray-500 mt-1 italic">{explanation}</p>
        </div>
      </div>
      {children}
    </div>
  );

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center gap-3">
            <span className="text-2xl">📊</span>
            <div>
              <h3 className="text-xl font-bold text-gray-800">Writing Style Analysis</h3>
              <p className="text-sm text-gray-500">Detailed linguistic metrics and patterns</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content - Scrollable */}
        <div className="overflow-y-auto p-6 space-y-5">
          {/* Quick Stats */}
          <div className="grid grid-cols-4 gap-3">
            <div className="bg-blue-50 rounded-lg p-4 border border-blue-200 text-center">
              <div className="text-xs font-semibold text-blue-800 mb-1">Total Words</div>
              <div className="text-3xl font-bold text-blue-900">
                {styleAnalysis.text_statistics.word_count}
              </div>
            </div>
            <div className="bg-green-50 rounded-lg p-4 border border-green-200 text-center">
              <div className="text-xs font-semibold text-green-800 mb-1">Sentences</div>
              <div className="text-3xl font-bold text-green-900">
                {styleAnalysis.text_statistics.sentence_count}
              </div>
            </div>
            <div className="bg-purple-50 rounded-lg p-4 border border-purple-200 text-center">
              <div className="text-xs font-semibold text-purple-800 mb-1">Grade Level</div>
              <div className="text-3xl font-bold text-purple-900">
                {styleAnalysis.readability.grade_level.toFixed(1)}
              </div>
            </div>
            <div className="bg-orange-50 rounded-lg p-4 border border-orange-200 text-center">
              <div className="text-xs font-semibold text-orange-800 mb-1">Characters</div>
              <div className="text-3xl font-bold text-orange-900">
                {styleAnalysis.text_statistics.character_count}
              </div>
            </div>
          </div>

          {/* Lexical Diversity */}
          <SectionCard
            icon="📚"
            title="Lexical Diversity"
            explanation="Measures vocabulary variety. Formula: Unique words / Total words. Higher scores indicate more diverse vocabulary (more human-like). AI often repeats similar words."
          >
            <MetricBar
              value={styleAnalysis.lexical_diversity.ttr}
              max={1}
              label="Type-Token Ratio (TTR)"
              color="bg-blue-500"
            />
            <div className="mt-3 p-3 bg-blue-50 rounded border border-blue-200">
              <p className="text-xs font-semibold text-blue-800">
                {styleAnalysis.lexical_diversity.interpretation}
              </p>
              <div className="grid grid-cols-2 gap-2 mt-2 text-xs text-blue-700">
                <div>✓ Unique words: {styleAnalysis.lexical_diversity.unique_words}</div>
                <div>✓ Total words: {styleAnalysis.lexical_diversity.total_words}</div>
              </div>
            </div>
            <div className="mt-2 text-xs text-gray-600 bg-gray-100 p-2 rounded">
              <strong>What it means:</strong> A score of 0.7-1.0 suggests varied vocabulary (human-like), while 0.3-0.5 suggests repetitive patterns (AI-like).
            </div>
          </SectionCard>

          {/* Predictability */}
          <SectionCard
            icon="🎲"
            title="Predictability (Entropy)"
            explanation="Measures how predictable the text is based on word frequency distribution. Higher entropy = more varied and unpredictable (human-like). Lower entropy = formulaic patterns (AI-like)."
          >
            <MetricBar
              value={styleAnalysis.perplexity_proxy.normalized_score}
              max={10}
              label="Entropy Score (Higher = More Varied)"
              color="bg-green-500"
            />
            <div className="mt-3 p-3 bg-green-50 rounded border border-green-200">
              <p className="text-xs font-semibold text-green-800">
                {styleAnalysis.perplexity_proxy.predictability}
              </p>
              <div className="text-xs text-green-700 mt-1">
                Raw entropy: {styleAnalysis.perplexity_proxy.entropy.toFixed(2)}
              </div>
            </div>
            <div className="mt-2 text-xs text-gray-600 bg-gray-100 p-2 rounded">
              <strong>What it means:</strong> Scores below 3 indicate very predictable text (AI-like). Scores above 7 indicate highly varied text with more surprises (human-like).
            </div>
          </SectionCard>

          {/* Readability */}
          <SectionCard
            icon="📖"
            title="Readability"
            explanation="Flesch Reading Ease score (0-100). Formula: 206.835 - 1.015 × (words/sentences) - 84.6 × (syllables/words). Higher scores = easier to read. Shows sophistication level."
          >
            <MetricBar
              value={styleAnalysis.readability.flesch_reading_ease}
              max={100}
              label="Flesch Reading Ease"
              color="bg-purple-500"
            />
            <div className="mt-3 p-3 bg-purple-50 rounded border border-purple-200">
              <p className="text-xs font-semibold text-purple-800">
                {styleAnalysis.readability.interpretation}
              </p>
              <div className="text-xs text-purple-700 mt-1">
                Grade Level: {styleAnalysis.readability.grade_level.toFixed(1)} (education level needed to understand)
              </div>
            </div>
            <div className="mt-2 text-xs text-gray-600 bg-gray-100 p-2 rounded">
              <strong>What it means:</strong> 90-100 = very easy (5th grade), 60-70 = plain English (8-9th grade), 30-50 = difficult (college), 0-30 = very difficult (professional).
            </div>
          </SectionCard>

          {/* Sentence Complexity */}
          <SectionCard
            icon="✏️"
            title="Sentence Structure"
            explanation="Analyzes sentence complexity through length and structure. AI often writes consistently. Humans naturally vary sentence length and use complex conjunctions (however, moreover, etc.)."
          >
            <MetricBar
              value={styleAnalysis.sentence_complexity.avg_sentence_length}
              max={40}
              label="Avg Sentence Length (words)"
              color="bg-yellow-500"
            />
            <div className="mt-3 p-3 bg-yellow-50 rounded border border-yellow-200">
              <p className="text-xs font-semibold text-yellow-800">
                {styleAnalysis.sentence_complexity.complexity_interpretation}
              </p>
              <div className="grid grid-cols-2 gap-2 mt-2 text-xs text-yellow-700">
                <div>Total sentences: {styleAnalysis.sentence_complexity.total_sentences}</div>
                <div>Complex sentences: {styleAnalysis.sentence_complexity.complex_sentences}</div>
                <div>Std deviation: {styleAnalysis.sentence_complexity.std_sentence_length.toFixed(2)}</div>
              </div>
            </div>
            <div className="mt-2 text-xs text-gray-600 bg-gray-100 p-2 rounded">
              <strong>What it means:</strong> {"<"}10 words = simple (AI-like brevity), 10-20 = moderate, 20-30 = complex (human-like), 30+ = academic/professional.
            </div>
          </SectionCard>

          {/* Vocabulary */}
          <SectionCard
            icon="🎓"
            title="Vocabulary Richness"
            explanation="Measures word sophistication by analyzing average word length and count of long words (7+ letters). AI might use unnecessarily complex words to sound smarter."
          >
            <MetricBar
              value={styleAnalysis.vocabulary_richness.avg_word_length}
              max={10}
              label="Avg Word Length (characters)"
              color="bg-indigo-500"
            />
            <div className="mt-3 p-3 bg-indigo-50 rounded border border-indigo-200">
              <p className="text-xs font-semibold text-indigo-800">
                {styleAnalysis.vocabulary_richness.interpretation}
              </p>
              <div className="grid grid-cols-2 gap-2 mt-2 text-xs text-indigo-700">
                <div>Long words (7+ letters): {styleAnalysis.vocabulary_richness.long_words_count}</div>
                <div>Long word ratio: {(styleAnalysis.vocabulary_richness.long_word_ratio * 100).toFixed(1)}%</div>
              </div>
            </div>
            <div className="mt-2 text-xs text-gray-600 bg-gray-100 p-2 rounded">
              <strong>What it means:</strong> 3-4 chars avg = basic, 4-5 = standard, 5-6+ = sophisticated. Long word ratio {">"} 30% indicates academic/professional writing.
            </div>
          </SectionCard>

          {/* Punctuation */}
          <SectionCard
            icon="📝"
            title="Punctuation Patterns"
            explanation="Analyzes how punctuation is used. Punctuation shows personality and emotion. AI text is often more neutral with fewer exclamations/questions. High semicolon use suggests academic writing."
          >
            <div className="grid grid-cols-3 gap-3 mb-3">
              <div className="text-center p-3 bg-gray-100 rounded">
                <div className="text-2xl font-bold text-gray-800">{styleAnalysis.punctuation_patterns.commas}</div>
                <div className="text-xs text-gray-600">Commas</div>
              </div>
              <div className="text-center p-3 bg-gray-100 rounded">
                <div className="text-2xl font-bold text-gray-800">{styleAnalysis.punctuation_patterns.periods}</div>
                <div className="text-xs text-gray-600">Periods</div>
              </div>
              <div className="text-center p-3 bg-gray-100 rounded">
                <div className="text-2xl font-bold text-gray-800">{styleAnalysis.punctuation_patterns.exclamations}</div>
                <div className="text-xs text-gray-600">Exclamations</div>
              </div>
              <div className="text-center p-3 bg-gray-100 rounded">
                <div className="text-2xl font-bold text-gray-800">{styleAnalysis.punctuation_patterns.questions}</div>
                <div className="text-xs text-gray-600">Questions</div>
              </div>
              <div className="text-center p-3 bg-gray-100 rounded">
                <div className="text-2xl font-bold text-gray-800">{styleAnalysis.punctuation_patterns.semicolons}</div>
                <div className="text-xs text-gray-600">Semicolons</div>
              </div>
              <div className="text-center p-3 bg-gray-100 rounded">
                <div className="text-2xl font-bold text-gray-800">{styleAnalysis.punctuation_patterns.colons}</div>
                <div className="text-xs text-gray-600">Colons</div>
              </div>
            </div>
            <div className="p-3 bg-pink-50 rounded border border-pink-200">
              <p className="text-xs font-semibold text-pink-800">
                {styleAnalysis.punctuation_patterns.interpretation}
              </p>
              <div className="text-xs text-pink-700 mt-1">
                Density: {(styleAnalysis.punctuation_patterns.punctuation_density * 100).toFixed(2)}%
              </div>
            </div>
            <div className="mt-2 text-xs text-gray-600 bg-gray-100 p-2 rounded">
              <strong>What it means:</strong> No exclamations/questions = formal/AI-like. Many = conversational/human emotion. High semicolons = academic.
            </div>
          </SectionCard>
        </div>

        {/* Footer */}
        <div className="border-t border-gray-200 p-4 bg-gray-50">
          <button
            onClick={onClose}
            className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
          >
            Close Analysis
          </button>
        </div>
      </div>
    </div>
  );
}
