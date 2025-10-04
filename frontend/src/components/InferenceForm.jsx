// src/components/InferenceForm.jsx

import React, { useState, useCallback } from 'react';
import './Inference.css'; 

const API_BASE_URL = process.env.REACT_APP_API_URL; 

// --- Agronomic Limits (Based on Pydantic/Domain Knowledge) ---
const FIELD_LIMITS = {
    N: { min: 0, max: 100, label: "ppm" },
    P: { min: 0, max: 100, label: "ppm" },
    K: { min: 0, max: 100, label: "ppm" },
    ph: { min: 0, max: 14, label: "pH units" },
    temperature: { min: -10, max: 60, label: "°C" },
    Humidity: { min: 0, max: 100, label: "%" },
    rainfall: { min: 0, max: 500, label: "mm" }
};

const InferenceForm = ({ setResult }) => { 
    
    const [mode, setMode] = useState('manual'); 
    const [formData, setFormData] = useState({
        N: '', P: '', K: '', ph: '',
        temperature: '', Humidity: '', rainfall: '', 
        city: '',
    });
    const [imageFile, setImageFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [fieldPrompts, setFieldPrompts] = useState({}); // State for field-specific feedback

    // --- Clear Form Logic ---
    const clearForm = () => {
        setFormData({ N: '', P: '', K: '', ph: '', temperature: '', Humidity: '', rainfall: '', city: '' });
        setImageFile(null);
        setResult(null); 
        setError(null);
        setFieldPrompts({}); // Clear prompts
        const fileInput = document.getElementById('soil-image-input');
        if (fileInput) fileInput.value = null;
    };

    // --- Input Validation Handler (UX Gimmick) ---
    const handleInputValidation = (e) => {
        const { name, value } = e.target;
        const numValue = parseFloat(value);
        
        if (value === '' || isNaN(numValue) || !FIELD_LIMITS[name]) {
            setFieldPrompts(prev => ({ ...prev, [name]: null }));
            return;
        }

        const { min, max, label } = FIELD_LIMITS[name];
        let message = null;

        if (numValue < min) {
            message = `⚠️ Value is too low. Min: ${min} ${label}.`;
        } else if (numValue > max) {
            message = `⚠️ Value is too high. Max: ${max} ${label}.`;
        } else {
            message = `✅ Value is within range.`;
        }

        setFieldPrompts(prev => ({ ...prev, [name]: message }));
    };


    // --- Core Handlers (Rest remains the same) ---
    const handleInputChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleFileChange = (e) => {
        setImageFile(e.target.files[0]);
    };
    const generateRandomId = () => {
    return 'TEST-' + Math.random().toString(36).substring(2, 9).toUpperCase();
};

    const handleSubmit = useCallback(async (e) => {
        e.preventDefault();
        setLoading(true);
        setResult(null);
        setError(null);

        if (!imageFile) {
            setError("Please upload a soil image.");
            setLoading(false);
            return;
        }

        try {
            const apiEndpoint = mode === 'manual' ? `${API_BASE_URL}/manual` : `${API_BASE_URL}/auto`;
            
            const submissionData = new FormData();
            submissionData.append('image_file', imageFile);
            submissionData.append('device_id', generateRandomId());
            
            for (const key in formData) {
                const value = formData[key];
                const isAutoFetched = (key === 'temperature' || key === 'Humidity' || key === 'rainfall');
                
                if (mode === 'auto' && isAutoFetched) {
                    continue; 
                }
                submissionData.append(key, value || ''); 
            }
            
            const response = await fetch(apiEndpoint, { method: 'POST', body: submissionData });
            const data = await response.json();

            if (!response.ok) {
                if (response.status === 422 && data.detail) {
                    const firstError = data.detail[0];
                    setError(`Validation Failed for '${firstError.loc[1]}' field: ${firstError.msg}`);
                } else {
                    setError(data.detail || `An error occurred (Status: ${response.status}).`);
                }
            } else {
                setResult(data);
            }

        } catch (err) {
            setError("Network or parsing error. Check your FastAPI server connection.");
            console.error(err);
        } finally {
            setLoading(false);
        }
    }, [formData, imageFile, mode, setResult]);

    const isManual = mode === 'manual';
    
    // --- Helper function to render a single input group ---
    const renderInput = (name, label, isRequired, isManualOnly = false) => {
        const limits = FIELD_LIMITS[name];
        const prompt = fieldPrompts[name];
        
        if (isManualOnly && !isManual) return null; // Skip rendering if in Auto Mode

        return (
            <div className="input-group" key={name}>
                <label>{label}:</label>
                <input 
                    type="number" 
                    name={name} 
                    value={formData[name]} 
                    onChange={handleInputChange} 
                    onBlur={handleInputValidation} // Attach validation handler
                    required={isRequired}
                    step="0.01"
                    placeholder={`Range: ${limits.min} to ${limits.max} ${limits.label}`}
                />
                {/* Dynamic Feedback Display */}
                {prompt && (
                    <p className={`field-prompt ${prompt.startsWith('⚠️') ? 'warning' : 'success'}`}>
                        {prompt}
                    </p>
                )}
            </div>
        );
    };

    return (
        <div className="container form-card">
            
            {/* --- Mode Selector --- */}
            <div className="mode-selector">
                <button 
                    className={`mode-btn ${isManual ? 'active' : ''}`}
                    onClick={() => setMode('manual')}
                >
                    Manual Input Mode
                </button>
                <button 
                    className={`mode-btn ${!isManual ? 'active' : ''}`}
                    onClick={() => setMode('auto')}
                >
                    Automatic (Weather Fetch) Mode
                </button>
            </div>

            {error && <div className="status error">Error: {error}</div>}

            {/* --- Form --- */}
            <form onSubmit={handleSubmit} className="inference-form">
                <h2>{isManual ? 'Manual Data Entry' : 'Automatic Mode'}</h2>
                
                {/* 1. NPK + pH Inputs (Core Soil Data) */}
                <h3>Core Field Measurements (IoT)</h3>
                {renderInput('N', 'Nitrogen (N) Content', true)}
                {renderInput('P', 'Phosphorus (P) Content', true)}
                {renderInput('K', 'Potassium (K) Content', true)}
                {renderInput('ph', 'Soil pH', true)}

                {/* 2. Weather Data Section (CONDITIONAL RENDERING) */}
                {isManual && (
                    <div className="weather-section">
                        <h3>Manual Weather Input (All Fields Required)</h3>
                        {renderInput('temperature', 'Temperature (C)', true, true)}
                        {renderInput('Humidity', 'Humidity (%)', true, true)}
                        {renderInput('rainfall', 'Rainfall (mm)', true, true)}
                    </div>
                )}
                
                {/* 3. City and Image (Context/File Upload) */}
                <h3 style={{marginTop: '20px'}}>Context & Image Input</h3>
                
                <div className="input-group">
                    <label>City Name ({!isManual ? 'REQUIRED for Weather Fetch' : 'Optional for Traceability'}):</label>
                    <input 
                        type="text" 
                        name="city" 
                        value={formData.city} 
                        onChange={handleInputChange} 
                        required={!isManual}
                        placeholder={!isManual ? "e.g., Mumbai or New Delhi" : "Optional for logging"}
                    />
                </div>

                <div className="input-group file-upload">
                    <label>Upload Soil Image (Required):</label>
                    <div className="file-upload-container" style={{position: 'relative'}}>
                        
                        <div className="custom-file-button">
                            {imageFile ? 'File Selected' : 'Choose File'}
                        </div>
                        
                        <input 
                            type="file" 
                            onChange={handleFileChange} 
                            accept="image/*" 
                            required 
                            id="soil-image-input" 
                        />
                        
                        {imageFile && <span className="selected-file-name"> ({imageFile.name})</span>}
                    </div>
                </div>
                
                {/* --- Submission Buttons --- */}
                <div className="button-group">
                    <button type="submit" disabled={loading} className="submit-btn">
                        {loading ? 'Processing Inference...' : 'Get Recommendation'}
                    </button>
                    
                    <button 
                        type="button" 
                        onClick={clearForm}
                        className="clear-btn"
                    >
                        Clear All
                    </button>
                </div>
            </form>
        </div>
    );
};

export default InferenceForm;