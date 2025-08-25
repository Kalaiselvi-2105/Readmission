-- Database initialization script for Hospital Readmission Predictor
-- This script creates the necessary tables and initial data

-- Create database if it doesn't exist
-- Note: This needs to be run as a superuser or the database needs to exist already

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    permissions TEXT[] NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create predictions table for logging
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    patient_id VARCHAR(100) NOT NULL,
    admission_id VARCHAR(100) NOT NULL,
    risk_score DECIMAL(5,4) NOT NULL,
    risk_category VARCHAR(50) NOT NULL,
    confidence VARCHAR(50) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    features_used INTEGER NOT NULL,
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id INTEGER REFERENCES users(id),
    patient_data JSONB
);

-- Create explanations table for logging
CREATE TABLE IF NOT EXISTS explanations (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER REFERENCES predictions(id),
    feature_contributions JSONB NOT NULL,
    explanation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create model_versions table for tracking
CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(100) UNIQUE NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    training_date TIMESTAMP NOT NULL,
    performance_metrics JSONB,
    feature_count INTEGER NOT NULL,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create audit_log table for system monitoring
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id VARCHAR(100),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_predictions_patient_id ON predictions(patient_id);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(prediction_timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp);

-- Insert default admin user (password: admin123)
INSERT INTO users (username, email, full_name, role, permissions, hashed_password, is_active)
VALUES (
    'admin.user',
    'admin@hospital.com',
    'System Administrator',
    'admin',
    ARRAY['read', 'predict', 'explain', 'admin', 'monitor'],
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/RK.s5uO.G', -- admin123
    TRUE
) ON CONFLICT (username) DO NOTHING;

-- Insert default clinician user (password: password123)
INSERT INTO users (username, email, full_name, role, permissions, hashed_password, is_active)
VALUES (
    'doctor.smith',
    'doctor.smith@hospital.com',
    'Dr. John Smith',
    'clinician',
    ARRAY['read', 'predict', 'explain'],
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/RK.s5uO.G', -- password123
    TRUE
) ON CONFLICT (username) DO NOTHING;

-- Insert default nurse user (password: password123)
INSERT INTO users (username, email, full_name, role, permissions, hashed_password, is_active)
VALUES (
    'nurse.jones',
    'nurse.jones@hospital.com',
    'Sarah Jones, RN',
    'nurse',
    ARRAY['read', 'predict'],
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/RK.s5uO.G', -- password123
    TRUE
) ON CONFLICT (username) DO NOTHING;

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for users table
CREATE TRIGGER update_users_updated_at 
    BEFORE UPDATE ON users 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions to the application user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
