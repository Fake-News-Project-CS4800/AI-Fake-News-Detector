import ChatInterface from '@/components/ChatInterface';

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <header className="text-center mb-8">
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-2">
            AI Fake News Detector
          </h1>
          <p className="text-gray-600 text-lg">
            Detect if text was written by AI or a human using advanced machine learning
          </p>
        </header>

        {/* Chat Interface */}
        <ChatInterface />

        {/* Footer Info */}
        <footer className="mt-8 text-center text-sm text-gray-500">
          <p>
            Powered by <span className="font-semibold">RoBERTa</span> •
            Model: <span className="font-semibold">Hello-SimpleAI/chatgpt-detector-roberta</span>
          </p>
        </footer>
      </div>
    </main>
  );
}
