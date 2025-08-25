import React, { useState } from 'react';
import { toast } from 'react-hot-toast';
import { apiService } from '../services/api';
import { useAuthStore } from '../stores/authStore';
import { FileText, BarChart3, TrendingUp, TrendingDown } from 'lucide-react';

const Explanations = () => {
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

  const [explanation, setExplanation] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [topFeatures, setTopFeatures] = useState(10);
  const { canExplain } = useAuthStore();

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!canExplain) {
      toast.error('You do not have permission to get explanations');
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

      const response = await apiService.explainPrediction(patientData, topFeatures);
      setExplanation(response.data);
      toast.success('Explanation generated successfully!');
    } catch (error) {
      console.error('Explanation error:', error);
      toast.error(error.response?.data?.detail || 'Explanation failed');
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

  const formatFeatureName = (featureName) => {
    // Convert feature names to readable format
    const nameMap = {
      'age': 'Age',
      'sex_M': 'Male',
      'sex_F': 'Female',
      'sex_Other': 'Other Gender',
      'charlson_index': 'Charlson Comorbidity Index',
      'prior_readmit_30d': 'Prior Readmissions (30 days)',
      'prior_readmit_365d': 'Prior Readmissions (365 days)',
      'ed_visits_180d': 'ED Visits (180 days)',
      'los_days': 'Length of Stay',
      'icu_stay': 'ICU Stay',
      'procedures_count': 'Number of Procedures',
      'meds_count': 'Number of Medications',
      'high_risk_meds_flag': 'High Risk Medications',
      'discharge_disposition_Home': 'Discharge to Home',
      'discharge_disposition_SNF': 'Discharge to SNF',
      'discharge_disposition_Rehab': 'Discharge to Rehab',
      'discharge_disposition_Hospice': 'Discharge to Hospice',
      'discharge_disposition_Other': 'Other Discharge',
      'followup_scheduled': 'Follow-up Scheduled',
      'days_to_followup': 'Days to Follow-up',
      'deprivation_index': 'Area Deprivation Index',
      'icd_count': 'Number of ICD Codes',
      'icd_I50': 'Heart Failure (I50)',
      'icd_E11': 'Type 2 Diabetes (E11)',
      'icd_I25': 'Chronic Ischemic Heart Disease (I25)',
      'icd_I10': 'Essential Hypertension (I10)',
      'icd_N18': 'Chronic Kidney Disease (N18)'
    };
    
    return nameMap[featureName] || featureName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center">
          <FileText className="h-8 w-8 text-purple-600 mr-3" />
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Prediction Explanations</h1>
            <p className="text-gray-600 mt-1">
              Understand how the model makes predictions using SHAP explanations
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Explanation Form */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Patient Information</h2>
            
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Top Features Selection */}
              <div className="mb-4">
                <label className="form-label">Number of Top Features to Show</label>
                <select
                  value={topFeatures}
                  onChange={(e) => setTopFeatures(parseInt(e.target.value))}
                  className="form-input w-32"
                >
                  <option value={5}>5</option>
                  <option value={10}>10</option>
                  <option value={15}>15</option>
                  <option value={20}>20</option>
                </select>
              </div>

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

              {/* Submit Button */}
              <div className="pt-4">
                <button
                  type="submit"
                  disabled={isLoading || !canExplain}
                  className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLoading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Generating Explanation...
                    </>
                  ) : (
                    'Generate Explanation'
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>

        {/* Explanation Results */}
        <div className="lg:col-span-1">
          {explanation ? (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 sticky top-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Explanation Results</h2>
              
              <div className="space-y-4">
                {/* Risk Score */}
                <div className="text-center p-4 bg-gray-50 rounded-lg">
                  <p className="text-sm text-gray-600 mb-1">Risk Score</p>
                  <p className="text-3xl font-bold text-blue-600">
                    {(explanation.risk_score * 100).toFixed(1)}%
                  </p>
                </div>

                {/* Risk Category */}
                <div className="text-center">
                  <p className="text-sm text-gray-600 mb-2">Risk Category</p>
                  <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getRiskLevelColor(explanation.risk_category)}`}>
                    {explanation.risk_category}
                  </span>
                </div>

                {/* Feature Contributions */}
                <div>
                  <h3 className="text-sm font-medium text-gray-900 mb-3">Top Feature Contributions</h3>
                  <div className="space-y-2">
                    {explanation.feature_contributions.map((feature, index) => (
                      <div key={index} className="p-2 bg-gray-50 rounded">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm font-medium text-gray-900">
                            {formatFeatureName(feature.feature_name)}
                          </span>
                          <span className={`text-xs font-medium ${
                            feature.contribution > 0 ? 'text-red-600' : 'text-green-600'
                          }`}>
                            {feature.contribution > 0 ? '+' : ''}{feature.contribution.toFixed(4)}
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div 
                            className={`h-2 rounded-full ${
                              feature.contribution > 0 ? 'bg-red-500' : 'bg-green-500'
                            }`}
                            style={{ 
                              width: `${Math.min(Math.abs(feature.contribution) * 100, 100)}%`,
                              marginLeft: feature.contribution > 0 ? '0' : 'auto'
                            }}
                          ></div>
                        </div>
                        <div className="flex items-center mt-1">
                          {feature.contribution > 0 ? (
                            <TrendingUp className="h-3 w-3 text-red-600 mr-1" />
                          ) : (
                            <TrendingDown className="h-3 w-3 text-green-600 mr-1" />
                          )}
                          <span className="text-xs text-gray-500">
                            {feature.contribution > 0 ? 'Increases' : 'Decreases'} risk
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Additional Info */}
                <div className="border-t border-gray-200 pt-4 space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Patient ID:</span>
                    <span className="font-medium">{explanation.patient_id}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Features Analyzed:</span>
                    <span className="font-medium">{explanation.feature_contributions.length}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Timestamp:</span>
                    <span className="font-medium">
                      {new Date(explanation.explanation_timestamp).toLocaleString()}
                    </span>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="pt-4 space-y-2">
                  <button
                    onClick={() => setExplanation(null)}
                    className="w-full btn-secondary"
                  >
                    New Explanation
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <div className="text-center text-gray-500">
                <FileText className="h-12 w-12 mx-auto mb-3 text-gray-300" />
                <p className="text-sm">No explanation yet</p>
                <p className="text-xs">Fill out the form and submit to see SHAP explanations</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Explanations;
