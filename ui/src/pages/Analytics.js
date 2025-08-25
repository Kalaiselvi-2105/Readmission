import React, { useState, useEffect } from 'react';
import { toast } from 'react-hot-toast';
import { apiService } from '../services/api';
import { useAuthStore } from '../stores/authStore';
import { BarChart3, TrendingUp, Activity, Shield } from 'lucide-react';

const Analytics = () => {
  const [metrics, setMetrics] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const { canMonitor } = useAuthStore();

  useEffect(() => {
    if (canMonitor) {
      loadMetrics();
    }
  }, [canMonitor]);

  const loadMetrics = async () => {
    setIsLoading(true);
    try {
      const response = await apiService.getMetrics();
      setMetrics(response.data);
    } catch (error) {
      console.error('Error loading metrics:', error);
      toast.error('Failed to load analytics data');
    } finally {
      setIsLoading(false);
    }
  };

  if (!canMonitor) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <Shield className="h-12 w-12 mx-auto mb-3 text-gray-400" />
          <p className="text-gray-600">You don't have permission to view analytics</p>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center">
          <BarChart3 className="h-8 w-8 text-purple-600 mr-3" />
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Model Analytics</h1>
            <p className="text-gray-600 mt-1">Performance metrics and model insights</p>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Model Performance</h2>
        {metrics?.performance_metrics ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <p className="text-2xl font-bold text-blue-600">
                {metrics.performance_metrics.auroc ? 
                  `${(metrics.performance_metrics.auroc * 100).toFixed(1)}%` : 'N/A'}
              </p>
              <p className="text-sm text-blue-600">AUROC Score</p>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <p className="text-2xl font-bold text-green-600">
                {metrics.performance_metrics.f1_score ? 
                  `${(metrics.performance_metrics.f1_score * 100).toFixed(1)}%` : 'N/A'}
              </p>
              <p className="text-sm text-green-600">F1 Score</p>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <p className="text-2xl font-bold text-purple-600">
                {metrics.model_info?.feature_count || 'N/A'}
              </p>
              <p className="text-sm text-purple-600">Features</p>
            </div>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <p>No performance metrics available</p>
          </div>
        )}
      </div>

      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Feature Importance</h2>
        {metrics?.feature_importance && metrics.feature_importance.length > 0 ? (
          <div className="space-y-3">
            {metrics.feature_importance.map((feature, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <span className="text-sm font-medium text-gray-700">
                  {feature.feature}
                </span>
                <span className="text-sm font-medium text-gray-900">
                  {feature.importance.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <p>No feature importance data available</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Analytics;
