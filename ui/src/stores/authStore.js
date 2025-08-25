import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import api from '../services/api';

const useAuthStore = create(
  persist(
    (set, get) => ({
      // State
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

      // Actions
      login: async (username, password) => {
        set({ isLoading: true, error: null });
        try {
          const response = await api.post('/auth/login', { username, password });
          const { access_token, user } = response.data;
          
          // Set token in API headers
          api.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
          
          set({
            user,
            token: access_token,
            isAuthenticated: true,
            isLoading: false,
            error: null
          });
          
          return { success: true };
        } catch (error) {
          const errorMessage = error.response?.data?.detail || 'Login failed';
          set({
            isLoading: false,
            error: errorMessage
          });
          return { success: false, error: errorMessage };
        }
      },

      logout: () => {
        // Remove token from API headers
        delete api.defaults.headers.common['Authorization'];
        
        set({
          user: null,
          token: null,
          isAuthenticated: false,
          isLoading: false,
          error: null
        });
      },

      checkAuth: async () => {
        const { token } = get();
        if (!token) {
          set({ isAuthenticated: false });
          return false;
        }

        try {
          // Set token in headers
          api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
          
          // Verify token by calling /auth/me
          const response = await api.get('/auth/me');
          const user = response.data;
          
          set({
            user,
            isAuthenticated: true,
            error: null
          });
          
          return true;
        } catch (error) {
          // Token is invalid, clear auth state
          set({
            user: null,
            token: null,
            isAuthenticated: false,
            error: null
          });
          
          // Remove token from API headers
          delete api.defaults.headers.common['Authorization'];
          
          return false;
        }
      },

      clearError: () => set({ error: null }),

      // Getters
      getUserRole: () => {
        const { user } = get();
        return user?.role || null;
      },

      hasPermission: (permission) => {
        const { user } = get();
        return user?.permissions?.includes(permission) || false;
      },

      canPredict: () => get().hasPermission('predict'),
      canExplain: () => get().hasPermission('explain'),
      canAdmin: () => get().hasPermission('admin'),
      canMonitor: () => get().hasPermission('monitor'),
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({ 
        user: state.user, 
        token: state.token, 
        isAuthenticated: state.isAuthenticated 
      }),
    }
  )
);

export default useAuthStore;
