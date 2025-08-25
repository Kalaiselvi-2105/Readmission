import axios from 'axios';

// Create axios instance
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth-storage');
    if (token) {
      try {
        const authData = JSON.parse(token);
        if (authData.state?.token) {
          config.headers.Authorization = `Bearer ${authData.state.token}`;
        }
      } catch (error) {
        console.error('Error parsing auth token:', error);
      }
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid, redirect to login
      localStorage.removeItem('auth-storage');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// API endpoints
export const endpoints = {
  // Auth
  login: '/auth/login',
  me: '/auth/me',
  
  // Predictions
  predict: '/predict',
  batchPredict: '/batch_predict',
  explain: '/explain',
  
  // Monitoring
  health: '/health',
  metrics: '/metrics',
  modelInfo: '/model_info',
  rateLimitInfo: '/rate_limit_info',
  
  // Admin
  users: '/admin/users',
  adminStats: '/admin/stats',
};

// Helper functions
export const apiService = {
  // Health check
  checkHealth: () => api.get(endpoints.health),
  
  // Authentication
  login: (credentials) => api.post(endpoints.login, credentials),
  getCurrentUser: () => api.get(endpoints.me),
  
  // Predictions
  predictReadmission: (patientData) => api.post(endpoints.predict, { patient_data: patientData }),
  batchPredictReadmission: (patients) => api.post(endpoints.batchPredict, { patients }),
  explainPrediction: (patientData, topFeatures = 10) => 
    api.post(endpoints.explain, { patient_data: patientData, top_features: topFeatures }),
  
  // Monitoring
  getMetrics: () => api.get(endpoints.metrics),
  getModelInfo: () => api.get(endpoints.modelInfo),
  getRateLimitInfo: () => api.get(endpoints.rateLimitInfo),
  
  // Admin
  getUsers: () => api.get(endpoints.users),
  getAdminStats: () => api.get(endpoints.adminStats),
};

export default api;
