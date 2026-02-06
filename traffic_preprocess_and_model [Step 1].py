import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_and_validate_data(file_path):
    """Load data with proper dtype specification to handle mixed types"""
    try:
        # Specify dtype for problematic column to avoid mixed type warning
        df = pd.read_csv(file_path, dtype={'Unnamed: 0': str}, low_memory=False)
        print(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def preprocess_data(df):
    """Preprocess the traffic data"""
    print("🔄 Preprocessing data...")
    
    # Initial info
    print(f"Initial data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Handle missing values
    initial_rows = df.shape[0]
    df = df.dropna(subset=['traffic_volume', 'aqi'])
    print(f"Removed {initial_rows - df.shape[0]} rows with missing target values")
    
    # Fill missing values
    df.fillna({
        'holiday': 'None', 
        'snow_1h': 0, 
        'rain_1h': 0,
        'weather_main': 'Unknown',
        'weather_description': 'Unknown'
    }, inplace=True)
    
    # Convert date_time with error handling
    try:
        df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
        # Remove rows where date conversion failed
        df = df.dropna(subset=['date_time'])
    except Exception as e:
        print(f"Warning: Error converting dates: {e}")
    
    # Create time-based features
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    
    # Encode categorical variables
    cat_cols = ['holiday', 'weather_main', 'weather_description', 'from_city', 'to_city']
    encoders = {}
    
    for col in cat_cols:
        if col in df.columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            encoders[col] = encoder
        else:
            print(f"Warning: Column '{col}' not found in dataframe")
    
    print(f"Final data shape after preprocessing: {df.shape}")
    return df, encoders

def train_model(X_train, X_test, y_train, y_test):
    """Train and evaluate the Random Forest model"""
    print("🤖 Training Random Forest model...")
    
    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,  # Use all available cores
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n📊 Model Performance Metrics:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔝 Top 10 Feature Importances:")
    print(feature_importance.head(10).to_string(index=False))
    
    return model, feature_importance

def main():
    """Main execution function"""
    print("🚦 Traffic Volume Prediction Model Training")
    print("=" * 50)
    
    # Load data
    df = load_and_validate_data("indian_traffic_data.csv")
    if df is None:
        return
    
    # Preprocess data
    df, encoders = preprocess_data(df)
    
    # Define features and target
    features = ['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all',
                'weather_main', 'weather_description', 'hour', 'day_of_week', 
                'month', 'is_rush_hour', 'aqi', 'from_city', 'to_city', 'is_weekend']
    
    # Check if all features exist
    available_features = [f for f in features if f in df.columns]
    missing_features = set(features) - set(available_features)
    
    if missing_features:
        print(f"⚠️  Missing features: {missing_features}")
        print(f"✅ Using available features: {available_features}")
    
    X = df[available_features]
    y = df['traffic_volume']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df['hour']  # Stratify by hour for better time distribution
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    model, feature_importance = train_model(X_train, X_test, y_train, y_test)
    
    # Save model and encoders
    try:
        joblib.dump({
            'model': model,
            'encoders': encoders,
            'feature_names': available_features,
            'feature_importance': feature_importance
        }, "traffic_volume_model.pkl")
        print("✅ Model saved as 'traffic_volume_model.pkl'")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
    
    print("\n🎯 Training completed successfully!")

if __name__ == "__main__":
    main()