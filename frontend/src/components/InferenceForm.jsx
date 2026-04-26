// src/components/InferenceForm.jsx

import React, { useState, useCallback } from 'react';
import './Inference.css';

const API_BASE_URL = process.env.REACT_APP_API_URL ?? "http://127.0.0.1:8000/api/v1/inference";

const FIELD_LIMITS = {
    N:           { min: 0,   max: 100, label: "ppm" },
    P:           { min: 0,   max: 100, label: "ppm" },
    K:           { min: 0,   max: 100, label: "ppm" },
    ph:          { min: 0,   max: 14,  label: "pH"  },
    temperature: { min: -10, max: 60,  label: "°C"  },
    Humidity:    { min: 0,   max: 100, label: "%"   },
    rainfall:    { min: 0,   max: 500, label: "mm"  },
};

const STEP_REQUIRED_FIELDS = {
    0: ['N', 'P', 'K', 'ph'],
    1: ['temperature', 'Humidity', 'rainfall'],
    2: [],
};

const STEPS = [
    { label: 'Soil',    icon: '🌱' },
    { label: 'Weather', icon: '🌤' },
    { label: 'Context', icon: '📍' },
];

const generateRandomId = () =>
    'TEST-' + Math.random().toString(36).substring(2, 9).toUpperCase();

const InferenceForm = ({ setResult }) => {
    const [mode, setMode]       = useState('manual');
    const [step, setStep]       = useState(0);
    const [formData, setFormData] = useState({
        N: '', P: '', K: '', ph: '',
        temperature: '', Humidity: '', rainfall: '', city: '',
    });
    const [imageFile, setImageFile]         = useState(null);
    const [loading, setLoading]             = useState(false);
    const [error, setError]                 = useState(null);
    const [fieldStates, setFieldStates]     = useState({});
    const [fieldMessages, setFieldMessages] = useState({});
    const [stepError, setStepError]         = useState(null);

    const isManual   = mode === 'manual';
    const totalSteps = isManual ? 3 : 2;

    // Auto mode skips step index 1 (weather), so map display step → logical step
    const logicalStep = (!isManual && step >= 1) ? step + 1 : step;

    const clearForm = () => {
        setFormData({ N: '', P: '', K: '', ph: '', temperature: '', Humidity: '', rainfall: '', city: '' });
        setImageFile(null);
        setResult(null);
        setError(null);
        setFieldStates({});
        setFieldMessages({});
        setStepError(null);
        setStep(0);
        const fi = document.getElementById('soil-image-input');
        if (fi) fi.value = null;
    };

    const handleModeSwitch = (newMode) => {
        setMode(newMode);
        setStep(0);
        setStepError(null);
        setError(null);
    };

    const handleInputChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
        setStepError(null);
    };

    const handleInputValidation = (e) => {
        const { name, value } = e.target;
        const numValue = parseFloat(value);

        if (value === '' || isNaN(numValue) || !FIELD_LIMITS[name]) {
            setFieldStates(prev => ({ ...prev, [name]: null }));
            setFieldMessages(prev => ({ ...prev, [name]: null }));
            return;
        }

        const { min, max, label } = FIELD_LIMITS[name];

        if (numValue < min) {
            setFieldStates(prev => ({ ...prev, [name]: 'invalid' }));
            setFieldMessages(prev => ({ ...prev, [name]: { type: 'warning', text: `Below minimum (${min} ${label})` } }));
        } else if (numValue > max) {
            setFieldStates(prev => ({ ...prev, [name]: 'invalid' }));
            setFieldMessages(prev => ({ ...prev, [name]: { type: 'warning', text: `Exceeds maximum (${max} ${label})` } }));
        } else {
            setFieldStates(prev => ({ ...prev, [name]: 'valid' }));
            setFieldMessages(prev => ({ ...prev, [name]: null }));
        }
    };

    const handleFileChange = (e) => {
        setImageFile(e.target.files[0] || null);
        setStepError(null);
    };

    const validateCurrentStep = () => {
        const required = STEP_REQUIRED_FIELDS[logicalStep] || [];
        const missing  = required.filter(f => !formData[f] || formData[f] === '');
        const invalid  = required.filter(f => fieldStates[f] === 'invalid');

        if (missing.length > 0) {
            setStepError(`Please fill in all fields before continuing.`);
            return false;
        }
        if (invalid.length > 0) {
            setStepError(`Fix out-of-range values before continuing.`);
            return false;
        }
        setStepError(null);
        return true;
    };

    const handleNext = () => {
        if (!validateCurrentStep()) return;
        setStep(s => s + 1);
    };

    const handleBack = () => {
        setStepError(null);
        setStep(s => s - 1);
    };

    const handleSubmit = useCallback(async (e) => {
        e.preventDefault();
        if (!imageFile) {
            setError("Please upload a soil image to continue.");
            return;
        }
        if (!isManual && !formData.city) {
            setError("City name is required for Auto Weather Fetch.");
            return;
        }

        setLoading(true);
        setResult(null);
        setError(null);

        try {
            const apiEndpoint = isManual
                ? `${API_BASE_URL}/manual`
                : `${API_BASE_URL}/auto`;

            const submissionData = new FormData();
            submissionData.append('image_file', imageFile);
            submissionData.append('device_id', generateRandomId());

            for (const key in formData) {
                const isAutoFetched = ['temperature', 'Humidity', 'rainfall'].includes(key);
                if (!isManual && isAutoFetched) continue;
                submissionData.append(key, formData[key] || '');
            }

            const response = await fetch(apiEndpoint, { method: 'POST', body: submissionData });
            const data     = await response.json();

            if (!response.ok) {
                if (response.status === 422 && data.detail) {
                    const firstError = data.detail[0];
                    setError(`Validation failed for '${firstError.loc[1]}': ${firstError.msg}`);
                } else {
                    setError(data.detail || `Error (Status: ${response.status})`);
                }
            } else {
                setResult(data);
            }
        } catch (err) {
            setError("Network error — check your FastAPI server connection.");
            console.error(err);
        } finally {
            setLoading(false);
        }
    }, [formData, imageFile, isManual, setResult]);

    const renderInput = (name, label) => {
        const limits = FIELD_LIMITS[name];
        const state  = fieldStates[name];
        const msg    = fieldMessages[name];

        return (
            <div className="input-group" key={name}>
                <label>{label} <span className="required-star">*</span></label>
                <div className="input-wrapper">
                    <input
                        type="number"
                        name={name}
                        value={formData[name]}
                        onChange={handleInputChange}
                        onBlur={handleInputValidation}
                        required
                        step="0.01"
                        placeholder={`${limits.min} – ${limits.max}`}
                        className={state || ''}
                    />
                    <span className="unit-badge">{limits.label}</span>
                </div>
                {msg && (
                    <p className={`field-prompt ${msg.type}`}>
                       {msg.text}
                    </p>
                )}
            </div>
        );
    };

    const renderStepContent = () => {
        if (logicalStep === 0) return (
            <div className="step-panel">
                <p className="step-description">Enter your IoT sensor readings from the field.</p>
                <div className="input-grid">
                    {renderInput('N',  'Nitrogen (N)')}
                    {renderInput('P',  'Phosphorus (P)')}
                    {renderInput('K',  'Potassium (K)')}
                    {renderInput('ph', 'Soil pH')}
                </div>
            </div>
        );

        if (logicalStep === 1) return (
            <div className="step-panel">
                <p className="step-description">Enter current weather conditions at your field location.</p>
                <div className="input-grid">
                    {renderInput('temperature', 'Temperature')}
                    {renderInput('Humidity',    'Humidity')}
                    <div className="span-full">{renderInput('rainfall', 'Rainfall')}</div>
                </div>
            </div>
        );

        if (logicalStep === 2) return (
            <div className="step-panel">
                <p className="step-description">
                    {isManual
                        ? 'Optionally name your location, then upload a soil photo.'
                        : 'Enter your city for weather fetch, then upload a soil photo.'}
                </p>
                <div className="input-grid">
                    <div className="input-group">
                        <label>
                            City
                            {!isManual && <span className="required-star"> *</span>}
                            <span style={{ color: 'var(--text-muted)', fontWeight: 400, fontSize: '0.78rem' }}>
                                {isManual ? ' (optional)' : ''}
                            </span>
                        </label>
                        <div className="input-wrapper">
                            <input
                                type="text"
                                name="city"
                                value={formData.city}
                                onChange={handleInputChange}
                                required={!isManual}
                                placeholder={isManual ? 'e.g. Manipal' : 'e.g. Mumbai'}
                                style={{ paddingRight: '12px' }}
                            />
                        </div>
                    </div>

                    <div className="input-group">
                        <label>
                            Soil Image <span className="required-star">*</span>
                        </label>
                        <div className={`file-upload-zone ${imageFile ? 'has-file' : ''}`}>
                            <input
                                type="file"
                                onChange={handleFileChange}
                                accept="image/*"
                                id="soil-image-input"
                            />
                            <span className="upload-icon">{imageFile ? '✅' : '📷'}</span>
                            {imageFile ? (
                                <>
                                    <span className="upload-label">Image ready</span>
                                    <span className="selected-file-name">{imageFile.name}</span>
                                </>
                            ) : (
                                <span className="upload-label"><span>Click to upload</span></span>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        );
    };

    const visibleSteps = isManual ? STEPS : [STEPS[0], STEPS[2]];
    const isLastStep   = step === totalSteps - 1;

    return (
        <div className="form-card">

            {/* Mode Switcher */}
            <div className="mode-selector">
                <button type="button" className={`mode-btn ${isManual ? 'active' : ''}`} onClick={() => handleModeSwitch('manual')}>
                    Manual Input
                </button>
                <button type="button" className={`mode-btn ${!isManual ? 'active' : ''}`} onClick={() => handleModeSwitch('auto')}>
                    Auto Weather Fetch
                </button>
            </div>

            {/* Step Indicator */}
            <div className="step-indicator">
                {visibleSteps.map((s, i) => (
                    <React.Fragment key={i}>
                        <button
                            type="button"
                            className={`step-dot ${i === step ? 'active' : ''} ${i < step ? 'done' : ''}`}
                            onClick={() => i < step && setStep(i)}
                        >
                            <span className="step-dot-icon">{i < step ? '✓' : s.icon}</span>
                            <span className="step-dot-label">{s.label}</span>
                        </button>
                        {i < visibleSteps.length - 1 && (
                            <div className={`step-connector ${i < step ? 'done' : ''}`} />
                        )}
                    </React.Fragment>
                ))}
            </div>

            {/* Errors */}
            {(error || stepError) && (
                <div className="status error">{error || stepError}</div>
            )}

            <form onSubmit={handleSubmit} className="inference-form">
                {renderStepContent()}

                <div className="button-group">
                    {step > 0 && (
                        <button type="button" onClick={handleBack} className="back-btn">
                            ← Back
                        </button>
                    )}
                    {!isLastStep ? (
                        <button type="button" onClick={handleNext} className="submit-btn">
                            Next →
                        </button>
                    ) : (
                        <button type="submit" disabled={loading} className="submit-btn">
                            {loading ? ' Processing…' : 'Get Recommendation'}
                        </button>
                    )}
                    <button type="button" onClick={clearForm} className="clear-btn">
                        Clear
                    </button>
                </div>
            </form>
        </div>
    );
};

export default InferenceForm;