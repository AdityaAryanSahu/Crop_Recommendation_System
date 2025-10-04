// src/App.jsx

import React, { useState } from 'react';
import InferenceForm from './components/InferenceForm';
import ResultDisplay from './components/ResultDisplay'; 
import './App.css'; 

function App() {
    // State lifted here to allow ResultDisplay to be rendered next to the form
    const [result, setResult] = useState(null);

    return (
        <div className="App">
            <header className="App-header">
                <h1>Multi-Modal Crop Recommendation System</h1>
                <p className="subtitle">Fusion of IoT, Weather, and Image Data.</p>
            </header>
            
            <main className="main-content-area"> 
                
                {/* 1. Form Component (Left Side) */}
                <InferenceForm setResult={setResult} />
                
                {/* 2. Result Component (Right Side) */}
                {/* It will be styled with .result-box and conditionally rendered */}
                {result && <ResultDisplay result={result} />}
            </main>

            <footer className="App-footer">
                <p>© 2025 Multi-Modal Agricultural Inference System</p>
            </footer>
        </div>
    );
}

export default App;