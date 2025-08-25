import React, { useState } from 'react';
import { toast } from 'react-hot-toast';
import { apiService } from '../services/api';
import { useAuthStore } from '../stores/authStore';
import { Activity, User, FileText, AlertTriangle } from 'lucide-react';

const Predictions = () => {
  const [formData, setFormData] = useState({
    patient_id: '',
    admission_id: '',
    age: '',
    sex: 'M',
    insurance: '',
    charlson_index: '',
    prior_readmit_30d: '0',
    prior_readmit_365d: '0',
    ed_visits_180d: '0',
    los_days: '',
    icu_stay: false,
    procedures_count: '0',
    meds_count: '0',
    high_risk_meds_flag: false,
    discharge_disposition: 'Home',
    followup_scheduled: false,
    days_to_followup: '0',
    zip_code: '',
    deprivation_index: '0.0',
    icd_count: '0',
    icd_I50: false,
    icd_E11: false,
    icd_I25: false,
    icd_I10: false,
    icd_N18: false
  });

  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const { canPredict } = useAuthStore();

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!canPredict) {
      toast.error('You do not have permission to make predictions');
      return;
    }

    setIsLoading(true);
    
    try {
      // Convert form data to proper types
      const patientData = {
        ...formData,
        age: parseInt(formData.age),
        charlson_index: parseInt(formData.charlson_index),
        prior_readmit_30d: parseInt(formData.prior_readmit_30d),
        prior_readmit_365d: parseInt(formData.prior_readmit_365d),
        ed_visits_180d: parseInt(formData.ed_visits_180d),
        los_days: parseInt(formData.los_days),
        procedures_count: parseInt(formData.procedures_count),
        meds_count: parseInt(formData.meds_count),
        days_to_followup: parseInt(formData.days_to_followup),
        deprivation_index: parseFloat(formData.deprivation_index),
        icd_count: parseInt(formData.icd_count)
      };

      const response = await apiService.predictReadmission(patientData);
      setPrediction(response.data);
      toast.success('Prediction completed successfully!');
    } catch (error) {
      console.error('Prediction error:', error);
      toast.error(error.response?.data?.detail || 'Prediction failed');
    } finally {
      setIsLoading(false);
    }
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

  const getConfidenceColor = (confidence) => {
    const colors = {
      'High': 'text-green-600',
      'Medium': 'text-yellow-600',
      'Low': 'text-red-600'
    };
    return colors[confidence] || 'text-gray-600';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center">
          <Activity className="h-8 w-8 text-blue-600 mr-3" />
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Single Patient Prediction</h1>
            <p className="text-gray-600 mt-1">
              Enter patient information to predict readmission risk
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Prediction Form */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Patient Information</h2>
            
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Basic Information */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="form-label">Patient ID *</label>
                  <input
                    type="text"
                    name="patient_id"
                    value={formData.patient_id}
                    onChange={handleInputChange}
                    className="form-input"
                    required
                  />
                </div>
                <div>
                  <label className="form-label">Admission ID *</label>
                  <input
                    type="text"
                    name="admission_id"
                    value={formData.admission_id}
                    onChange={handleInputChange}
                    className="form-input"
                    required
                  />
                </div>
                <div>
                  <label className="form-label">Age *</label>
                  <input
                    type="number"
                    name="age"
                    value={formData.age}
                    onChange={handleInputChange}
                    className="form-input"
                    min="0"
                    max="120"
                    required
                  />
                </div>
                <div>
                  <label className="form-label">Sex *</label>
                  <select
                    name="sex"
                    value={formData.sex}
                    onChange={handleInputChange}
                    className="form-input"
                    required
                  >
                    <option value="M">Male</option>
                    <option value="F">Female</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
                <div>
                  <label className="form-label">Insurance *</label>
                  <input
                    type="text"
                    name="insurance"
                    value={formData.insurance}
                    onChange={handleInputChange}
                    className="form-input"
                    required
                  />
                </div>
                <div>
                  <label className="form-label">Charlson Index *</label>
                  <input
                    type="number"
                    name="charlson_index"
                    value={formData.charlson_index}
                    onChange={handleInputChange}
                    className="form-input"
                    min="0"
                    max="10"
                    required
                  />
                </div>
              </div>

              {/* Medical History */}
              <div>
                <h3 className="text-md font-medium text-gray-900 mb-3">Medical History</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="form-label">Prior Readmissions (30d)</label>
                    <input
                      type="number"
                      name="prior_readmit_30d"
                      value={formData.prior_readmit_30d}
                      onChange={handleInputChange}
                      className="form-input"
                      min="0"
                    />
                  </div>
                  <div>
                    <label className="form-label">Prior Readmissions (365d)</label>
                    <input
                      type="number"
                      name="prior_readmit_365d"
                      value={formData.prior_readmit_365d}
                      onChange={handleInputChange}
                      className="form-input"
                      min="0"
                    />
                  </div>
                  <div>
                    <label className="form-label">ED Visits (180d)</label>
                    <input
                      type="number"
                      name="ed_visits_180d"
                      value={formData.ed_visits_180d}
                      onChange={handleInputChange}
                      className="form-input"
                      min="0"
                    />
                  </div>
                </div>
              </div>

              {/* Current Stay */}
              <div>
                <h3 className="text-md font-medium text-gray-900 mb-3">Current Stay</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="form-label">Length of Stay (days) *</label>
                    <input
                      type="number"
                      name="los_days"
                      value={formData.los_days}
                      onChange={handleInputChange}
                      className="form-input"
                      min="0"
                      required
                    />
                  </div>
                  <div>
                    <label className="form-label">ICU Stay</label>
                    <div className="mt-2">
                      <label className="inline-flex items-center">
                        <input
                          type="checkbox"
                          name="icu_stay"
                          checked={formData.icu_stay}
                          onChange={handleInputChange}
                          className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                        <span className="ml-2 text-sm text-gray-700">Yes</span>
                      </label>
                    </div>
                  </div>
                  <div>
                    <label className="form-label">Procedures Count</label>
                    <input
                      type="number"
                      name="procedures_count"
                      value={formData.procedures_count}
                      onChange={handleInputChange}
                      className="form-input"
                      min="0"
                    />
                  </div>
                </div>
              </div>

              {/* Medications */}
              <div>
                <h3 className="text-md font-medium text-gray-900 mb-3">Medications</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="form-label">Medications Count</label>
                    <input
                      type="number"
                      name="meds_count"
                      value={formData.meds_count}
                      onChange={handleInputChange}
                      className="form-input"
                      min="0"
                    />
                  </div>
                  <div>
                    <label className="form-label">High Risk Medications</label>
                    <div className="mt-2">
                      <label className="inline-flex items-center">
                        <input
                          type="checkbox"
                          name="high_risk_meds_flag"
                          checked={formData.high_risk_meds_flag}
                          onChange={handleInputChange}
                          className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                        <span className="ml-2 text-sm text-gray-700">Yes</span>
                      </label>
                    </div>
                  </div>
                </div>
              </div>

              {/* Discharge & Follow-up */}
              <div>
                <h3 className="text-md font-medium text-gray-900 mb-3">Discharge & Follow-up</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="form-label">Discharge Disposition *</label>
                    <select
                      name="discharge_disposition"
                      value={formData.discharge_disposition}
                      onChange={handleInputChange}
                      className="form-input"
                      required
                    >
                      <option value="Home">Home</option>
                      <option value="SNF">SNF</option>
                      <option value="Rehab">Rehab</option>
                      <option value="Hospice">Hospice</option>
                      <option value="Other">Other</option>
                    </select>
                  </div>
                  <div>
                    <label className="form-label">Follow-up Scheduled</label>
                    <div className="mt-2">
                      <label className="inline-flex items-center">
                        <input
                          type="checkbox"
                          name="followup_scheduled"
                          checked={formData.followup_scheduled}
                          onChange={handleInputChange}
                          className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                        <span className="ml-2 text-sm text-gray-700">Yes</span>
                      </label>
                    </div>
                  </div>
                  <div>
                    <label className="form-label">Days to Follow-up</label>
                    <input
                      type="number"
                      name="days_to_followup"
                      value={formData.days_to_followup}
                      onChange={handleInputChange}
                      className="form-input"
                      min="0"
                    />
                  </div>
                </div>
              </div>

              {/* Demographics */}
              <div>
                <h3 className="text-md font-medium text-gray-900 mb-3">Demographics</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="form-label">Zip Code *</label>
                    <input
                      type="text"
                      name="zip_code"
                      value={formData.zip_code}
                      onChange={handleInputChange}
                      className="form-input"
                      required
                    />
                  </div>
                  <div>
                    <label className="form-label">Deprivation Index</label>
                    <input
                      type="number"
                      name="deprivation_index"
                      value={formData.deprivation_index}
                      onChange={handleInputChange}
                      className="form-input"
                      step="0.1"
                      min="0"
                    />
                  </div>
                </div>
              </div>

              {/* ICD Codes */}
              <div>
                <h3 className="text-md font-medium text-gray-900 mb-3">ICD Diagnosis Codes</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="form-label">Total ICD Count</label>
                    <input
                      type="number"
                      name="icd_count"
                      value={formData.icd_count}
                      onChange={handleInputChange}
                      className="form-input"
                      min="0"
                    />
                  </div>
                  <div className="col-span-2">
                    <div className="grid grid-cols-2 gap-4">
                      <label className="inline-flex items-center">
                        <input
                          type="checkbox"
                          name="icd_I50"
                          checked={formData.icd_I50}
                          onChange={handleInputChange}
                          className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                        <span className="ml-2 text-sm text-gray-700">I50 (Heart Failure)</span>
                      </label>
                      <label className="inline-flex items-center">
                        <input
                          type="checkbox"
                          name="icd_E11"
                          checked={formData.icd_E11}
                          onChange={handleInputChange}
                          className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                        <span className="ml-2 text-sm text-gray-700">E11 (Type 2 Diabetes)</span>
                      </label>
                      <label className="inline-flex items-center">
                        <input
                          type="checkbox"
                          name="icd_I25"
                          checked={formData.icd_I25}
                          onChange={handleInputChange}
                          className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                        <span className="ml-2 text-sm text-gray-700">I25 (Chronic Ischemic Heart)</span>
                      </label>
                      <label className="inline-flex items-center">
                        <input
                          type="checkbox"
                          name="icd_I10"
                          checked={formData.icd_I10}
                          onChange={handleInputChange}
                          className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                        <span className="ml-2 text-sm text-gray-700">I10 (Essential Hypertension)</span>
                      </label>
                      <label className="inline-flex items-center">
                        <input
                          type="checkbox"
                          name="icd_N18"
                          checked={formData.icd_N18}
                          onChange={handleInputChange}
                          className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                        <span className="ml-2 text-sm text-gray-700">N18 (Chronic Kidney Disease)</span>
                      </label>
                    </div>
                  </div>
                </div>
              </div>

              {/* Submit Button */}
              <div className="pt-4">
                <button
                  type="submit"
                  disabled={isLoading || !canPredict}
                  className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLoading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Processing...
                    </>
                  ) : (
                    'Predict Readmission Risk'
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>

        {/* Prediction Results */}
        <div className="lg:col-span-1">
          {prediction ? (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 sticky top-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Prediction Results</h2>
              
              <div className="space-y-4">
                {/* Risk Score */}
                <div className="text-center p-4 bg-gray-50 rounded-lg">
                  <p className="text-sm text-gray-600 mb-1">Risk Score</p>
                  <p className="text-3xl font-bold text-blue-600">
                    {(prediction.risk_score * 100).toFixed(1)}%
                  </p>
                </div>

                {/* Risk Category */}
                <div className="text-center">
                  <p className="text-sm text-gray-600 mb-2">Risk Category</p>
                  <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getRiskLevelColor(prediction.risk_category)}`}>
                    {prediction.risk_category}
                  </span>
                </div>

                {/* Confidence Level */}
                <div className="text-center">
                  <p className="text-sm text-gray-600 mb-2">Confidence</p>
                  <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-100 ${getConfidenceColor(prediction.confidence)}`}>
                    {prediction.confidence}
                  </span>
                </div>

                {/* Additional Info */}
                <div className="border-t border-gray-200 pt-4 space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Patient ID:</span>
                    <span className="font-medium">{prediction.patient_id}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Features Used:</span>
                    <span className="font-medium">{prediction.features_used}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Model Version:</span>
                    <span className="font-medium">{prediction.model_version}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Timestamp:</span>
                    <span className="font-medium">
                      {new Date(prediction.prediction_timestamp).toLocaleString()}
                    </span>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="pt-4 space-y-2">
                  <button
                    onClick={() => window.location.href = '/explanations'}
                    className="w-full btn-secondary"
                  >
                    <FileText className="h-4 w-4 mr-2 inline" />
                    Get Explanation
                  </button>
                  <button
                    onClick={() => setPrediction(null)}
                    className="w-full btn-secondary"
                  >
                    New Prediction
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <div className="text-center text-gray-500">
                <Activity className="h-12 w-12 mx-auto mb-3 text-gray-300" />
                <p className="text-sm">No prediction yet</p>
                <p className="text-xs">Fill out the form and submit to see results</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Predictions;
