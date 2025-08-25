import React from 'react';
import { NavLink } from 'react-router-dom';
import { useAuthStore } from '../stores/authStore';
import {
  Home,
  Activity,
  Users,
  BarChart3,
  Settings,
  FileText,
  Upload,
  Shield
} from 'lucide-react';

const Sidebar = () => {
  const { canPredict, canExplain, canAdmin, canMonitor } = useAuthStore();

  const navigation = [
    {
      name: 'Dashboard',
      href: '/',
      icon: Home,
      current: true,
      permission: 'read'
    },
    {
      name: 'Single Prediction',
      href: '/predictions',
      icon: Activity,
      current: false,
      permission: 'predict'
    },
    {
      name: 'Batch Predictions',
      href: '/batch-predictions',
      icon: Upload,
      current: false,
      permission: 'predict'
    },
    {
      name: 'Explanations',
      href: '/explanations',
      icon: FileText,
      current: false,
      permission: 'explain'
    },
    {
      name: 'Analytics',
      href: '/analytics',
      icon: BarChart3,
      current: false,
      permission: 'monitor'
    },
    {
      name: 'Admin Panel',
      href: '/admin',
      icon: Shield,
      current: false,
      permission: 'admin'
    }
  ];

  const hasPermission = (permission) => {
    switch (permission) {
      case 'predict':
        return canPredict;
      case 'explain':
        return canExplain;
      case 'admin':
        return canAdmin;
      case 'monitor':
        return canMonitor;
      default:
        return true;
    }
  };

  const filteredNavigation = navigation.filter(item => hasPermission(item.permission));

  return (
    <div className="w-64 bg-white shadow-sm border-r border-gray-200 min-h-screen">
      <div className="flex flex-col h-full">
        {/* Navigation */}
        <nav className="flex-1 px-4 py-6 space-y-2">
          {filteredNavigation.map((item) => (
            <NavLink
              key={item.name}
              to={item.href}
              className={({ isActive }) =>
                `group flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors duration-200 ${
                  isActive
                    ? 'bg-blue-100 text-blue-700 border-r-2 border-blue-600'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                }`
              }
            >
              <item.icon
                className={`mr-3 h-5 w-5 ${
                  'text-gray-400 group-hover:text-gray-500'
                }`}
              />
              {item.name}
            </NavLink>
          ))}
        </nav>

        {/* Bottom section */}
        <div className="p-4 border-t border-gray-200">
          <div className="flex items-center space-x-3">
            <div className="h-8 w-8 bg-green-100 rounded-full flex items-center justify-center">
              <svg className="h-4 w-4 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-900 truncate">
                System Status
              </p>
              <p className="text-xs text-gray-500">
                Operational
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
