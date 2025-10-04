// src/components/ResultDisplay.jsx

import React from 'react';
import './Inference.css'; // Uses the common component CSS

const ResultDisplay = ({ result }) => {
    
    // Safely extract results using optional chaining and defaults
    const finalCrop = result?.crop_pred || 'N/A';
    const soilPred = result?.soil_pred || 'N/A';
    const rawPred = result?.raw_tabular_prediction || 'N/A';
    const isCompatible = result?.compatibility ?? false; 
    const alternatives = result?.alternatives || [];
    const message = result?.mssg || 'Successfully retrieved results.';
    const dateTime = result?.date_time ? new Date(result.date_time).toLocaleString() : 'N/A';
    const deviceId = result?.device_id || 'N/A';

    return (
        <div className="result-box visible"> 
            <h2>Final Recommendation</h2>
            
            <p className="message">{message}</p>

            <div className="final-recommendation">
                <h3>Recommended Crop:</h3>
                {/* Use explicit String() coercion and optional chaining for safety */}
                <h1 className="final-crop-name">{String(finalCrop)?.toUpperCase() || 'ERROR'}</h1>
            </div>

            <div className="fusion-details">
                <h3>Traceability & Details:</h3>
                
                <p>
                    <strong>Compatibility Status: </strong> 
                    <span className={`status-tag ${isCompatible ? 'compatible' : 'incompatible'}`}>
                        {isCompatible ? ' COMPATIBLE' : ' FALLBACK APPLIED '}
                    </span>
                </p>

                <p>
                    <strong>Predicted Soil Type: </strong> 
                    <span className="soil-type">
                        {String(soilPred)?.toUpperCase() || 'N/A'}
                    </span>
                </p>
                
                <p>
                    <strong>Raw Tabular Prediction: </strong> 
                    <span className="raw-pred-name">
                        {String(rawPred)?.toUpperCase() || 'N/A'}
                    </span>
                </p>

                {
                    !isCompatible && alternatives.length > 0 && (
                        <div className="alternatives-list">
                            <h4>Top 5 Alternatives for {soilPred}:</h4>
                            <ul>
                                {alternatives.map((crop, index) => (
                                    <li key={index}>{String(crop)?.toUpperCase()}</li>
                                ))}
                            </ul>
                        </div>
                    )
                }
            </div>
            
            <p className="device-info">
                <small>Inference Time: {dateTime} | Trace ID: {deviceId}</small>
            </p>
        </div>
    );
};

export default ResultDisplay;