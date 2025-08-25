import React, { useState, useEffect } from 'react';
import { toast } from 'react-hot-toast';
import { useAuthStore } from '../stores/authStore';
import { apiService } from '../services/api';
import {
  Activity,
  Users,
  TrendingUp,
  Shield,
  Clock,
  AlertTriangle,
  Upload
} from 'lucide-react';

const Dashboard = () => {
  const [metrics, setMetrics] = useState(null);
  const [healthStatus, setHealthStatus] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const { user } = useAuthStore();

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    setIsLoading(true);
    try {
      const [metricsResponse, healthResponse] = await Promise.all([
        apiService.getMetrics(),
        apiService.checkHealth()
      ]);
      
      setMetrics(metricsResponse.data);
      setHealthStatus(healthResponse.data);
    } catch (error) {
      console.error('Error loading dashboard data:', error);
      toast.error('Failed to load dashboard data');
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

  const getStatusColor = (status) => {
    return status === 'healthy' ? 'text-green-600' : 'text-red-600';
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Welcome Header */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              Welcome back, {user?.full_name || user?.username}!
            </h1>
            <p className="text-gray-600 mt-1">
              Here's what's happening with your hospital readmission prediction system.
            </p>
          </div>
          <div className="text-right">
            <p className="text-sm text-gray-500">Last updated</p>
            <p className="text-sm font-medium text-gray-900">
              {new Date().toLocaleTimeString()}
            </p>
          </div>
        </div>
      </div>

      {/* System Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* System Health */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <Shield className="h-8 w-8 text-blue-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">System Status</p>
              <p className={`text-lg font-semibold ${getStatusColor(healthStatus?.status)}`}>
                {healthStatus?.status || 'Unknown'}
              </p>
            </div>
          </div>
          <div className="mt-4">
            <p className="text-xs text-gray-500">
              Uptime: {Math.round((healthStatus?.uptime || 0) / 3600)}h
            </p>
          </div>
        </div>

        {/* Model Performance */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <TrendingUp className="h-8 w-8 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Model Performance</p>
              <p className="text-lg font-semibold text-gray-900">
                {metrics?.performance_metrics?.auroc ? 
                  `${(metrics.performance_metrics.auroc * 100).toFixed(1)}%` : 
                  'N/A'
                }
              </p>
            </div>
          </div>
          <div className="mt-4">
            <p className="text-xs text-gray-500">AUROC Score</p>
          </div>
        </div>

        {/* Features Used */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <Activity className="h-8 w-8 text-purple-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Features</p>
              <p className="text-lg font-semibold text-gray-900">
                {metrics?.model_info?.feature_count || 'N/A'}
              </p>
            </div>
          </div>
          <div className="mt-4">
            <p className="text-xs text-gray-500">Total Features</p>
          </div>
        </div>

        {/* User Role */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <Users className="h-8 w-8 text-indigo-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Your Role</p>
              <p className="text-lg font-semibold text-gray-900 capitalize">
                {user?.role || 'Unknown'}
              </p>
            </div>
          </div>
          <div className="mt-4">
            <p className="text-xs text-gray-500">
              {user?.permissions?.length || 0} permissions
            </p>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button
            onClick={() => window.location.href = '/predictions'}
            className="flex items-center justify-center p-4 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors duration-200"
          >
            <Activity className="h-6 w-6 text-blue-600 mr-3" />
            <span className="font-medium text-gray-900">New Prediction</span>
          </button>
          
          <button
            onClick={() => window.location.href = '/batch-predictions'}
            className="flex items-center justify-center p-4 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors duration-200"
          >
            <Upload className="h-6 w-6 text-green-600 mr-3" />
            <span className="font-medium text-gray-900">Batch Predictions</span>
          </button>
          
          <button
            onClick={() => window.location.href = '/analytics'}
            className="flex items-center justify-center p-4 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors duration-200"
          >
            <TrendingUp className="h-6 w-6 text-purple-600 mr-3" />
            <span className="font-medium text-gray-900">View Analytics</span>
          </button>
        </div>
      </div>

      {/* Recent Activity & System Info */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model Information */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Model Information</h2>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-sm text-gray-500">Model Type:</span>
              <span className="text-sm font-medium text-gray-900">
                {metrics?.model_info?.best_model_type || 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-500">Features Used:</span>
              <span className="text-sm font-medium text-gray-900">
                {metrics?.model_info?.feature_count || 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-500">Last Updated:</span>
              <span className="text-sm font-medium text-gray-900">
                {metrics?.last_updated ? 
                  new Date(metrics.last_updated).toLocaleDateString() : 
                  'N/A'
                }
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-500">SHAP Explainer:</span>
              <span className="text-sm font-medium text-gray-900">
                {metrics?.model_info?.shap_explainer_available ? 'Available' : 'Not Available'}
              </span>
            </div>
          </div>
        </div>

        {/* System Health Details */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">System Health</h2>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-sm text-gray-500">Status:</span>
              <span className={`text-sm font-medium ${getStatusColor(healthStatus?.status)}`}>
                {healthStatus?.status || 'Unknown'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-500">Version:</span>
              <span className="text-sm font-medium text-gray-900">
                {healthStatus?.version || 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-500">Uptime:</span>
              <span className="text-sm font-medium text-gray-900">
                {healthStatus?.uptime ? 
                  `${Math.round(healthStatus.uptime / 3600)}h ${Math.round((healthStatus.uptime % 3600) / 60)}m` : 
                  'N/A'
                }
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-500">Models Loaded:</span>
              <span className="text-sm font-medium text-gray-900">
                {healthStatus?.model_status?.models_loaded || 'N/A'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
