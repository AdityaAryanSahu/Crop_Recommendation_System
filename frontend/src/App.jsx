// src/App.jsx

import React, { useState } from 'react';
import InferenceForm from './components/InferenceForm';
import ResultDisplay from './components/ResultDisplay';
import './App.css';

function App() {
    const [result, setResult] = useState(null);

    return (
        <div className="App">
            <header className="App-header">
                <h1>Multi-Modal Crop Recommendation System</h1>
                <p className="subtitle">Fusion of IoT, Weather, and Image Data.</p>
            </header>

            <main className={`main-content-area ${result ? 'has-result' : ''}`}>
                <InferenceForm setResult={setResult} />
                {result && <ResultDisplay result={result} />}
            </main>

            <footer className="App-footer">
                <p>© 2025 Multi-Modal Agricultural Inference System</p>
            </footer>
        </div>
    );
}

export default App;