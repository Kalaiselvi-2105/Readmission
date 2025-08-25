import React, { useState } from 'react';
import { toast } from 'react-hot-toast';
import { apiService } from '../services/api';
import { useAuthStore } from '../stores/authStore';
import { Upload, Download, FileText, Activity } from 'lucide-react';

const BatchPredictions = () => {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [csvData, setCsvData] = useState('');
  const { canPredict } = useAuthStore();

  const handleFileUpload = (e) => {
    const uploadedFile = e.target.files[0];
    if (uploadedFile && uploadedFile.type === 'text/csv') {
      setFile(uploadedFile);
      readCSVFile(uploadedFile);
    } else {
      toast.error('Please upload a valid CSV file');
    }
  };

  const readCSVFile = (file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      setCsvData(e.target.result);
    };
    reader.readAsText(file);
  };

  const handleCSVInput = (e) => {
    setCsvData(e.target.value);
  };

  const parseCSV = (csvText) => {
    const lines = csvText.trim().split('\n');
    if (lines.length < 2) {
      throw new Error('CSV must have at least a header row and one data row');
    }

    const headers = lines[0].split(',').map(h => h.trim());
    const patients = [];

    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',').map(v => v.trim());
      if (values.length !== headers.length) {
        continue; // Skip malformed rows
      }

      const patient = {};
      headers.forEach((header, index) => {
        let value = values[index];
        
        // Convert numeric values
        if (['age', 'charlson_index', 'prior_readmit_30d', 'prior_readmit_365d', 
             'ed_visits_180d', 'los_days', 'procedures_count', 'meds_count', 
             'days_to_followup', 'deprivation_index', 'icd_count'].includes(header)) {
          value = value === '' ? 0 : parseInt(value) || 0;
        }
        
        // Convert boolean values
        if (['icu_stay', 'high_risk_meds_flag', 'followup_scheduled', 
             'icd_I50', 'icd_E11', 'icd_I25', 'icd_I10', 'icd_N18'].includes(header)) {
          value = value.toLowerCase() === 'true' || value === '1' || value === 'yes';
        }
        
        // Convert float values
        if (header === 'deprivation_index') {
          value = value === '' ? 0.0 : parseFloat(value) || 0.0;
        }

        patient[header] = value;
      });

      // Validate required fields
      if (patient.patient_id && patient.admission_id && patient.age && patient.sex && 
          patient.insurance && patient.charlson_index !== undefined && patient.los_days !== undefined) {
        patients.push(patient);
      }
    }

    return patients;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!canPredict) {
      toast.error('You do not have permission to make predictions');
      return;
    }

    if (!csvData.trim()) {
      toast.error('Please provide CSV data');
      return;
    }

    setIsLoading(true);
    
    try {
      const patients = parseCSV(csvData);
      
      if (patients.length === 0) {
        toast.error('No valid patient data found in CSV');
        return;
      }

      toast.success(`Processing ${patients.length} patients...`);

      const response = await apiService.batchPredictReadmission(patients);
      setPredictions(response.data);
      toast.success(`Batch prediction completed! Processed ${patients.length} patients`);
    } catch (error) {
      console.error('Batch prediction error:', error);
      toast.error(error.response?.data?.detail || 'Batch prediction failed');
    } finally {
      setIsLoading(false);
    }
  };

  const downloadResults = () => {
    if (!predictions) return;

    const csvContent = [
      ['Patient ID', 'Risk Score', 'Risk Category', 'Confidence', 'Timestamp'],
      ...predictions.predictions.map(p => [
        p.patient_id,
        p.risk_score,
        p.risk_category,
        p.confidence,
        p.prediction_timestamp
      ])
    ].map(row => row.join(',')).join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `batch_predictions_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const getRiskLevelColor = (riskLevel) => {
    const colors = {
      'Low': 'bg-green-100 text-green-800',
      'Medium-Low': 'bg-blue-100 text-blue-800',
      'Medium': 'bg-yellow-100 text-yellow-800',
      'Medium-High': 'bg-orange-100 text-orange-800',
      'High': 'bg-red-100 text-red-800'
    };
    return colors[riskLevel] || 'bg-gray-100 text-gray-800';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center">
          <Upload className="h-8 w-8 text-green-600 mr-3" />
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Batch Predictions</h1>
            <p className="text-gray-600 mt-1">
              Upload CSV file with multiple patients for batch readmission risk prediction
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Section */}
        <div>
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Upload Patient Data</h2>
            
            <form onSubmit={handleSubmit} className="space-y-4">
              {/* File Upload */}
              <div>
                <label className="form-label">Upload CSV File</label>
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  className="form-input"
                  disabled={isLoading}
                />
                <p className="text-xs text-gray-500 mt-1">
                  File should contain patient data with headers matching the schema
                </p>
              </div>

              {/* CSV Input */}
              <div>
                <label className="form-label">Or Paste CSV Data</label>
                <textarea
                  value={csvData}
                  onChange={handleCSVInput}
                  className="form-input h-64 font-mono text-sm"
                  placeholder="patient_id,admission_id,age,sex,insurance,charlson_index,los_days,zip_code&#10;P001,A001,65,M,Private,2,5,12345&#10;P002,A002,72,F,Medicare,3,7,67890"
                  disabled={isLoading}
                />
                <p className="text-xs text-gray-500 mt-1">
                  Include headers: patient_id, admission_id, age, sex, insurance, charlson_index, los_days, zip_code
                </p>
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={isLoading || !canPredict || !csvData.trim()}
                className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Processing...
                  </>
                ) : (
                  'Process Batch Predictions'
                )}
              </button>
            </form>

            {/* CSV Template */}
            <div className="mt-6 p-4 bg-gray-50 rounded-lg">
              <h3 className="text-sm font-medium text-gray-900 mb-2">CSV Template</h3>
              <div className="text-xs text-gray-600 font-mono">
                <p>patient_id,admission_id,age,sex,insurance,charlson_index,prior_readmit_30d,prior_readmit_365d,ed_visits_180d,los_days,icu_stay,procedures_count,meds_count,high_risk_meds_flag,discharge_disposition,followup_scheduled,days_to_followup,zip_code,deprivation_index,icd_count,icd_I50,icd_E11,icd_I25,icd_I10,icd_N18</p>
                <p className="mt-1">P001,A001,65,M,Private,2,0,1,2,5,false,2,5,false,Home,true,7,12345,0.5,3,false,true,false,true,false</p>
              </div>
            </div>
          </div>
        </div>

        {/* Results Section */}
        <div>
          {predictions ? (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900">Batch Results</h2>
                <button
                  onClick={downloadResults}
                  className="btn-secondary"
                >
                  <Download className="h-4 w-4 mr-2" />
                  Download CSV
                </button>
              </div>

              {/* Summary Stats */}
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="text-center p-3 bg-blue-50 rounded-lg">
                  <p className="text-2xl font-bold text-blue-600">{predictions.total_patients}</p>
                  <p className="text-sm text-blue-600">Total Patients</p>
                </div>
                <div className="text-center p-3 bg-green-50 rounded-lg">
                  <p className="text-2xl font-bold text-green-600">
                    {predictions.predictions.filter(p => !p.error).length}
                  </p>
                  <p className="text-sm text-green-600">Successful</p>
                </div>
              </div>

              {/* Risk Distribution */}
              <div className="mb-4">
                <h3 className="text-sm font-medium text-gray-900 mb-2">Risk Distribution</h3>
                <div className="space-y-2">
                  {['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'].map(risk => {
                    const count = predictions.predictions.filter(p => 
                      !p.error && p.risk_category === risk
                    ).length;
                    const percentage = predictions.total_patients > 0 ? 
                      ((count / predictions.total_patients) * 100).toFixed(1) : 0;
                    
                    return (
                      <div key={risk} className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">{risk}</span>
                        <div className="flex items-center space-x-2">
                          <div className="w-20 bg-gray-200 rounded-full h-2">
                            <div 
                              className={`h-2 rounded-full ${getRiskLevelColor(risk).split(' ')[0]}`}
                              style={{ width: `${percentage}%` }}
                            ></div>
                          </div>
                          <span className="text-sm font-medium text-gray-900 w-8 text-right">
                            {count}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Recent Results */}
              <div>
                <h3 className="text-sm font-medium text-gray-900 mb-2">Recent Results</h3>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {predictions.predictions.slice(0, 10).map((pred, index) => (
                    <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-900 truncate">
                          {pred.patient_id}
                        </p>
                        <p className="text-xs text-gray-500">
                          {pred.error ? 'Error' : `${(pred.risk_score * 100).toFixed(1)}% risk`}
                        </p>
                      </div>
                      {!pred.error && (
                        <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getRiskLevelColor(pred.risk_category)}`}>
                          {pred.risk_category}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Errors */}
              {predictions.errors && predictions.errors.length > 0 && (
                <div className="mt-4 p-3 bg-red-50 rounded-lg">
                  <h3 className="text-sm font-medium text-red-900 mb-2">Errors ({predictions.errors.length})</h3>
                  <div className="text-xs text-red-700 space-y-1">
                    {predictions.errors.slice(0, 5).map((error, index) => (
                      <p key={index} className="truncate">
                        {error.patient_id}: {error.error}
                      </p>
                    ))}
                    {predictions.errors.length > 5 && (
                      <p className="text-red-600">... and {predictions.errors.length - 5} more errors</p>
                    )}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <div className="text-center text-gray-500">
                <Upload className="h-12 w-12 mx-auto mb-3 text-gray-300" />
                <p className="text-sm">No batch results yet</p>
                <p className="text-xs">Upload CSV data and process to see results</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default BatchPredictions;
