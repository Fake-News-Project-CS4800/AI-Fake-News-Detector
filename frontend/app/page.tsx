import ChatInterface from '@/components/ChatInterface';

export default function Home() {
  return (
    <main className="min-h-screen "
      style={{
        backgroundColor: '#F6EEEE',
        backgroundImage: `
      repeating-linear-gradient(
        to bottom,
        transparent 0,
        transparent 23px,
        rgba(0, 120, 255, 0.18) 23px,
        rgba(0, 120, 255, 0.18) 24px
      ),
      linear-gradient(
        to right,
        rgba(255, 0, 0, 0.4) 0px,
        rgba(255, 0, 0, 0.4) 2px,
        transparent 2px
      ),
      linear-gradient(
        to left,
        rgba(255, 0, 0, 0.4) 0px,
        rgba(255, 0, 0, 0.4) 2px,
        transparent 2px
      ),
      repeating-radial-gradient(
        circle 10px,
        rgba(0,0,0,.1) 5px,
        rgba(0,0,0,.25) 5px,
        transparent 10px,
        transparent 70px
        )
    `,
        backgroundPosition: "0px 67px, 40px 0px, 0px 0px, 5px 0px",
        backgroundSize: "100% 45px, 100% 100%, 95% 100%, 32px 90px",
        backgroundRepeat: "repeat-y, no-repeat, no-repeat, repeat-y",

      }}
    >
      <div className="container mx-auto px-8 py-1 shadow-xl mix-blend-multiply"

      >
        {/* Header */}
        <header className=" "
        >
          <h1 className="text-4xl md:text-4xl font-bold text-gray-600 text-center mb-4">
            Assignment: Detect AI vs Human
          </h1>
          <p className="text-gray-600 text-2xl text-left ">
            Section 1: True or False
          </p>
        </header>

        {/* Chat Interface */}
        <ChatInterface />

        {/* Footer Info */}
        <footer className="mt-8 text-center text-sm text-gray-600">
          <p>
            Powered by <span className="font-semibold">RoBERTa</span> •
            Model: <span className="font-semibold">Hello-SimpleAI/chatgpt-detector-roberta</span>
          </p>
        </footer>
      </div>
    </main>
  );
}
