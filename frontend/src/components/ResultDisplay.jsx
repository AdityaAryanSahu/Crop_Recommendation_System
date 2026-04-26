// src/components/ResultDisplay.jsx

import React from 'react';
import './Inference.css';

// Maps crop names to emojis for a friendlier display

const ResultDisplay = ({ result }) => {
    const finalCrop   = result?.crop_pred || 'N/A';
    const soilPred    = result?.soil_pred || 'N/A';
    const rawPred     = result?.raw_tabular_prediction || 'N/A';
    const isCompatible = result?.compatibility ?? false;
    const alternatives = result?.alternatives || [];
    const message     = result?.mssg || 'Successfully retrieved results.';
    const dateTime    = result?.date_time
        ? new Date(result.date_time).toLocaleString()
        : 'N/A';
    const deviceId = result?.device_id || 'N/A';

    const cropName = String(finalCrop).toUpperCase();
    

    return (
        <div className="result-box">
            <h2>Final Recommendation</h2>

            {/* Status + message */}
            <div className="result-status-row">
                <span className={`status-badge ${isCompatible ? 'compatible' : 'incompatible'}`}>
                    {isCompatible ? ' Compatible' : ' Fallback Applied'}
                </span>
            </div>
            <p className="message">{message}</p>

            {/* Crop hero */}
            <div className="crop-hero">
                <p className="crop-label">Recommended Crop</p>
                <h1 className="final-crop-name">{cropName}</h1>
            </div>

            {/* Detail rows */}
            <div className="detail-section">
                <div className="detail-section-title">Traceability &amp; Details</div>

                <div className="detail-row">
                    <span className="detail-key">Predicted Soil Type</span>
                    <span className="soil-chip">{String(soilPred).toUpperCase()}</span>
                </div>

                <div className="detail-row">
                    <span className="detail-key">Raw Tabular Prediction</span>
                    <span className="detail-val">{String(rawPred).toUpperCase()}</span>
                </div>

                <div className="detail-row">
                    <span className="detail-key">Compatibility</span>
                    <span className="detail-val" style={{ color: isCompatible ? 'var(--accent-green)' : 'var(--accent-amber)' }}>
                        {isCompatible ? 'COMPATIBLE' : 'FALLBACK'}
                    </span>
                </div>
            </div>

            {/* Alternatives (only when fallback applied) */}
            {!isCompatible && alternatives.length > 0 && (
                <div className="alternatives-list">
                    <h4>Top alternatives for {String(soilPred).toUpperCase()} soil</h4>
                    <div className="alt-pills">
                        {alternatives.map((crop, i) => (
                            <span key={i} className="alt-pill">
                                 {String(crop).toUpperCase()}
                            </span>
                        ))}
                    </div>
                </div>
            )}

            {/* Trace footer */}
            <div className="device-info">
                <small>
                    {dateTime}<br />
                    Trace ID: {deviceId}
                </small>
            </div>
        </div>
    );
};

export default ResultDisplay;