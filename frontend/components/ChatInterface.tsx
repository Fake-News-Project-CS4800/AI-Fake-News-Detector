'use client';

import { useState, useRef, useEffect } from 'react';
import { analyzeText, APIError } from '@/lib/api';
import { Message } from '@/lib/types';
import ResultCard from './ResultCard';


export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const resultsRef = useRef<HTMLDivElement>(null);
  const [count, setCount] = useState(0);

  // Auto-scroll to results when messages change
  useEffect(() => {
    if (messages.length > 0 && resultsRef.current) {
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }, 500);
    }
  }, [messages]);

  function increment() {
    setCount(c => {
      const next = c + 1;
      return next === 0 ? next + 1 : next
    });   // increment the counter
  }


  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!input.trim() || isLoading) return;

    increment();

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [userMessage, ...prev]);
    setInput('');
    setIsLoading(true);

    try {
      // Call the API
      const result = await analyzeText(input, true);

      // Add bot response
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: `Analysis complete! This text appears to be **${result.label}**-written.`,
        result,
        timestamp: new Date(),
      };

      setMessages((prev) => [botMessage, ...prev]);
    } catch (error) {
      console.error('Analysis failed:', error);

      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: error instanceof APIError
          ? `❌ Error: ${error.message}`
          : '❌ Failed to analyze text. Please make sure the backend is running.',
        timestamp: new Date(),
      };

      setMessages((prev) => [errorMessage, ...prev]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col max-w-3xl mx-auto space-y-4 ml-0"
      style={{
        padding: '0px 20px',
      }}
    >
      {/* Input Area */}

      <form onSubmit={handleSubmit} className="relative">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Paste your text here to check if it's AI-generated"
          className="w-full p-4 pr-24 text-gray-600 placeholder-gray-500 border-1 border-gray-200 rounded-lg shadow-xl focus:ring-1 focus:ring-blue-100 focus:border-blue-100 resize-none"
          rows={3}
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={!input.trim() || isLoading}
          className="absolute bottom-4 right-4 p-3 bg-blue-500 text-black rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
        >
          {isLoading ? (
            <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"
            />
          ) : (
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
              />
            </svg>
          )}
        </button>
      </form>
      <p
        className={'text-black'}
      >{count === 0 ? "" : count})</p>


      {/* Messages Area */}
      {messages.length > 0 && (
        <div className="space-y-4 p-4">

          {messages.map((message, index) => (
            <div
              key={message.id}
              ref={message.type === 'bot' ? resultsRef : null}
              className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] rounded-lg p-4 ${message.type === 'user'
                  ? 'bg-blue-400 text-black shadow-xl '
                  : 'bg-white text-gray-900 shadow-sm border border-gray-200 '
                  }`}
              >

                {message.type === 'user' ? (
                  <p className="whitespace-pre-wrap break-words">{message.content}</p>
                ) : (
                  <div className="space-y-4">
                    <p className="text-sm">{message.content}</p>
                    {message.result && <ResultCard result={message.result} />}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {isLoading && (
        <div className="p-4">
          <div className="flex justify-start">
            <div className="bg-white rounded-lg p-4 shadow-xl border border-gray-200">
              <div className="flex items-center gap-2">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
                <span className="text-sm text-gray-500">Analyzing...</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
