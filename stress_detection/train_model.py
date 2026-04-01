import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pickle
import json
from datetime import datetime, timedelta
import random

class SocialMediaStressModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.stages = ['Minimal', 'Low', 'Moderate', 'High']

    def generate_synthetic_data(self, num_samples=10000):
        """Generate synthetic training data based on behavioral patterns"""
        np.random.seed(42)

        data = []
        for _ in range(num_samples):
            # Login Time Features (0-100%)
            night_usage = np.random.beta(2, 5) * 100  # Skewed toward low values
            morning_check = np.random.binomial(1, 0.3)  # 30% morning checkers
            late_night_days = np.random.poisson(2.5)  # Avg 2.5 late nights/week

            # Session Duration Features
            daily_usage_hours = np.random.gamma(2, 0.5)  # Skewed toward 1-2 hours
            avg_session_min = np.random.gamma(3, 5)  # Avg 15 mins
            binge_sessions = np.random.poisson(1.5)  # Avg 1.5 binge sessions/week

            # Frequency Features
            daily_checkins = np.random.poisson(15)  # Avg 15 checkins/day
            avg_interval_min = np.random.exponential(30)  # Avg 30 min intervals
            weekend_spike = np.random.uniform(0.8, 2.0)  # 0.8-2.0x weekday usage

            # Calculate stress score (based on our earlier formula)
            login_score = self._calculate_login_score(night_usage, morning_check, late_night_days)
            duration_score = self._calculate_duration_score(daily_usage_hours, avg_session_min, binge_sessions)
            frequency_score = self._calculate_frequency_score(daily_checkins, avg_interval_min, weekend_spike)

            total_score = login_score + duration_score + frequency_score

            # Determine stress stage
            if total_score <= 7:
                stage = 0  # Minimal
            elif total_score <= 15:
                stage = 1  # Low
            elif total_score <= 23:
                stage = 2  # Moderate
            else:
                stage = 3  # High

            data.append([
                night_usage, morning_check, late_night_days,
                daily_usage_hours, avg_session_min, binge_sessions,
                daily_checkins, avg_interval_min, weekend_spike,
                login_score, duration_score, frequency_score,
                stage
            ])

        columns = [
            'night_usage_pct', 'morning_check', 'late_night_days',
            'daily_usage_hours', 'avg_session_min', 'binge_sessions',
            'daily_checkins', 'avg_interval_min', 'weekend_spike',
            'login_score', 'duration_score', 'frequency_score',
            'stress_stage'
        ]

        return pd.DataFrame(data, columns=columns)

    def _calculate_login_score(self, night_usage, morning_check, late_night_days):
        """Calculate login pattern score (0-10)"""
        score = 0
        if night_usage > 30:
            score += 4
        if morning_check:
            score += 3
        if late_night_days >= 4:
            score += 3
        return min(score, 10)

    def _calculate_duration_score(self, daily_usage, avg_session, binge_sessions):
        """Calculate duration pattern score (0-10)"""
        score = 0
        if daily_usage > 3:
            score += 4
        if avg_session > 20:
            score += 3
        if binge_sessions >= 3:
            score += 3
        return min(score, 10)

    def _calculate_frequency_score(self, daily_checkins, avg_interval, weekend_spike):
        """Calculate frequency pattern score (0-10)"""
        score = 0
        if daily_checkins > 15:
            score += 4
        if avg_interval < 15:
            score += 3
        if weekend_spike > 1.5:
            score += 3
        return min(score, 10)

    def train(self, df):
        """Train the model on the dataframe"""
        # Features: all except scores and target
        feature_columns = [
            'night_usage_pct', 'morning_check', 'late_night_days',
            'daily_usage_hours', 'avg_session_min', 'binge_sessions',
            'daily_checkins', 'avg_interval_min', 'weekend_spike'
        ]

        X = df[feature_columns]
        y = df['stress_stage']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        # Get unique labels present in the test set
        unique_labels = np.unique(y_test)
        # Map unique labels to stage names for classification report
        present_target_names = [self.stages[label] for label in unique_labels]

        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, labels=unique_labels, target_names=present_target_names))

        # Feature importance
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nFeature Importance:")
        print(importance_df)

        return accuracy

    def save_model(self, model_path='stress_model.pkl', scaler_path='scaler.pkl', info_path='model_info.json'):
        """Save model, scaler, and metadata"""
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        metadata = {
            'model_type': 'GradientBoostingClassifier',
            'trained_date': datetime.now().isoformat(),
            'features': [
                'night_usage_pct',
                'morning_check',
                'late_night_days',
                'daily_usage_hours',
                'avg_session_min',
                'binge_sessions',
                'daily_checkins',
                'avg_interval_min',
                'weekend_spike'
            ],
            'stages': self.stages
        }

        with open(info_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        print(f"Metadata saved to {info_path}")

    def load_model(self, model_path='stress_model.pkl', scaler_path='scaler.pkl'):
        """Load trained model and scaler"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        print("Model loaded successfully")

    def predict_stress(self, user_data):
        """Predict stress stage for new user data"""
        # Convert user data to dataframe
        user_df = pd.DataFrame([user_data])

        # Scale features
        user_scaled = self.scaler.transform(user_df)

        # Predict
        prediction = self.model.predict(user_scaled)[0]
        probabilities = self.model.predict_proba(user_scaled)[0]

        result = {
            'stage': self.stages[prediction],
            'stage_index': int(prediction),
            'probabilities': {
                self.stages[i]: float(prob)
                for i, prob in enumerate(probabilities)
            },
            'confidence': float(probabilities[prediction])
        }

        return result

    def predict_stress_from_patterns(self, patterns):
        """Predict stress from our 3 main patterns"""
        # Convert patterns to features
        user_data = {
            'night_usage_pct': patterns.get('night_usage', 0),
            'morning_check': 1 if patterns.get('morning_check', False) else 0,
            'late_night_days': patterns.get('late_night_days', 0),
            'daily_usage_hours': patterns.get('daily_usage_hours', 0),
            'avg_session_min': patterns.get('avg_session_min', 0),
            'binge_sessions': patterns.get('binge_sessions', 0),
            'daily_checkins': patterns.get('daily_checkins', 0),
            'avg_interval_min': patterns.get('avg_interval_min', 0),
            'weekend_spike': patterns.get('weekend_spike', 1.0)
        }

        return self.predict_stress(user_data)


def main():
    """Main training pipeline"""
    print("=== Social Media Stress Model Trainer ===\n")

    # Initialize model
    stress_model = SocialMediaStressModel()

    # Generate synthetic data
    print("Generating synthetic training data...")
    df = stress_model.generate_synthetic_data(num_samples=5000)

    print(f"Dataset shape: {df.shape}")
    print("\nClass distribution:")
    print(df['stress_stage'].value_counts().sort_index())

    # Train model
    print("\nTraining model...")
    accuracy = stress_model.train(df)

    # Save model
    print("\nSaving model...")
    stress_model.save_model()

    # Test with sample data
    print("\n=== Testing with Sample Data ===")

    # Sample user data
    sample_user = {
        'night_usage_pct': 45.0,
        'morning_check': 1,
        'late_night_days': 5,
        'daily_usage_hours': 4.2,
        'avg_session_min': 28.5,
        'binge_sessions': 4,
        'daily_checkins': 22,
        'avg_interval_min': 12.3,
        'weekend_spike': 1.8
    }

    prediction = stress_model.predict_stress(sample_user)
    print(f"\nPrediction for sample user:")
    print(f"Stage: {prediction['stage']}")
    print(f"Confidence: {prediction['confidence']:.2%}")
    print(f"Probabilities: {prediction['probabilities']}")

if __name__ == "__main__":
    main()

