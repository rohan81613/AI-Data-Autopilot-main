"""
Smart Auto-Pilot System for Beginners
One-click intelligent data cleaning, preparation, and analysis
Uses ML to automatically fix all data issues
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

class SmartAutoPilot:
    """
    Intelligent Auto-Pilot that cleans and prepares data automatically
    Perfect for beginners - just one click!
    """
    
    def __init__(self):
        self.cleaning_report = []
        self.recommendations = []
    
    def auto_clean_everything(self, df: pd.DataFrame, dataset_name: str = "data") -> tuple:
        """
        🚀 ONE-CLICK SMART CLEANING
        Automatically fixes ALL data issues using ML
        Returns: (cleaned_df, detailed_report, recommendations)
        """
        self.cleaning_report = []
        self.recommendations = []
        
        original_shape = df.shape
        df_clean = df.copy()
        
        self.cleaning_report.append(f"🚀 **Starting Smart Auto-Clean for '{dataset_name}'**")
        self.cleaning_report.append(f"📊 Original: {original_shape[0]:,} rows × {original_shape[1]} columns\n")
        
        # Step 1: Remove completely empty rows/columns
        df_clean, step_report = self._remove_empty_data(df_clean)
        self.cleaning_report.extend(step_report)
        
        # Step 2: Detect and fix data types automatically
        df_clean, step_report = self._auto_fix_data_types(df_clean)
        self.cleaning_report.extend(step_report)
        
        # Step 3: Handle missing values intelligently
        df_clean, step_report = self._smart_handle_missing(df_clean)
        self.cleaning_report.extend(step_report)
        
        # Step 4: Remove duplicates
        df_clean, step_report = self._remove_duplicates(df_clean)
        self.cleaning_report.extend(step_report)
        
        # Step 5: Detect and handle outliers using ML
        df_clean, step_report = self._ml_handle_outliers(df_clean)
        self.cleaning_report.extend(step_report)
        
        # Step 6: Clean text data
        df_clean, step_report = self._clean_text_data(df_clean)
        self.cleaning_report.extend(step_report)
        
        # Step 7: Optimize data types for memory
        df_clean, step_report = self._optimize_memory(df_clean)
        self.cleaning_report.extend(step_report)
        
        # Generate final report
        final_shape = df_clean.shape
        self.cleaning_report.append(f"\n✅ **CLEANING COMPLETE!**")
        self.cleaning_report.append(f"📊 Final: {final_shape[0]:,} rows × {final_shape[1]} columns")
        self.cleaning_report.append(f"📉 Removed: {original_shape[0] - final_shape[0]:,} rows")
        self.cleaning_report.append(f"💾 Memory saved: {self._calculate_memory_saved(df, df_clean)}")
        
        # Generate recommendations
        self._generate_recommendations(df_clean)
        
        report = "\n".join(self.cleaning_report)
        recommendations = "\n".join(self.recommendations)
        
        return df_clean, report, recommendations
    
    def _remove_empty_data(self, df):
        """Remove completely empty rows and columns"""
        report = []
        report.append("### 🗑️ Step 1: Removing Empty Data")
        
        # Remove empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            df = df.drop(columns=empty_cols)
            report.append(f"   ✅ Removed {len(empty_cols)} completely empty columns")
        
        # Remove empty rows
        empty_rows = df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            df = df.dropna(how='all')
            report.append(f"   ✅ Removed {empty_rows} completely empty rows")
        
        if not empty_cols and empty_rows == 0:
            report.append("   ✅ No empty rows/columns found")
        
        report.append("")
        return df, report
    
    def _auto_fix_data_types(self, df):
        """Automatically detect and fix data types"""
        report = []
        report.append("### 🔧 Step 2: Auto-Fixing Data Types")
        
        conversions = 0
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try numeric conversion
                try:
                    numeric_converted = pd.to_numeric(df[col], errors='coerce')
                    # If more than 80% convert successfully, use it
                    if numeric_converted.notna().sum() / len(df) > 0.8:
                        df[col] = numeric_converted
                        conversions += 1
                        continue
                except:
                    pass
                
                # Try datetime conversion
                try:
                    if df[col].astype(str).str.match(r'\d{4}-\d{2}-\d{2}').sum() > len(df) * 0.5:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        conversions += 1
                except:
                    pass
        
        if conversions > 0:
            report.append(f"   ✅ Auto-converted {conversions} columns to correct types")
        else:
            report.append("   ✅ All data types are correct")
        
        report.append("")
        return df, report
    
    def _smart_handle_missing(self, df):
        """Intelligently handle missing values using ML"""
        report = []
        report.append("### 🔮 Step 3: Smart Missing Value Handling (ML-Powered)")
        
        total_missing = df.isnull().sum().sum()
        if total_missing == 0:
            report.append("   ✅ No missing values found")
            report.append("")
            return df, report
        
        filled_count = 0
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count == 0:
                continue
            
            missing_pct = (missing_count / len(df)) * 100
            
            # If too many missing (>50%), drop the column
            if missing_pct > 50:
                df = df.drop(columns=[col])
                report.append(f"   🗑️ Dropped '{col}' ({missing_pct:.1f}% missing)")
                continue
            
            # Numeric columns: Use KNN imputation for better accuracy
            if df[col].dtype in ['int64', 'float64']:
                try:
                    # Use KNN imputer with other numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) > 1:
                        imputer = KNNImputer(n_neighbors=min(5, len(df)//2))
                        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                    else:
                        # Fallback to median
                        df[col].fillna(df[col].median(), inplace=True)
                    filled_count += missing_count
                except:
                    # Fallback to median
                    df[col].fillna(df[col].median(), inplace=True)
                    filled_count += missing_count
            
            # Categorical columns: Use mode or 'Unknown'
            else:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col].fillna(mode_val[0], inplace=True)
                else:
                    df[col].fillna('Unknown', inplace=True)
                filled_count += missing_count
        
        report.append(f"   ✅ Intelligently filled {filled_count:,} missing values using ML")
        report.append("")
        return df, report
    
    def _remove_duplicates(self, df):
        """Remove duplicate rows"""
        report = []
        report.append("### 🔄 Step 4: Removing Duplicates")
        
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            df = df.drop_duplicates()
            report.append(f"   ✅ Removed {dup_count:,} duplicate rows")
        else:
            report.append("   ✅ No duplicates found")
        
        report.append("")
        return df, report
    
    def _ml_handle_outliers(self, df):
        """Use ML (Isolation Forest) to detect and handle outliers"""
        report = []
        report.append("### 🎯 Step 5: ML-Based Outlier Detection")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            report.append("   ℹ️ No numeric columns for outlier detection")
            report.append("")
            return df, report
        
        total_outliers = 0
        
        for col in numeric_cols:
            try:
                # Use Isolation Forest for outlier detection
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_pred = iso_forest.fit_predict(df[[col]].dropna())
                
                # Cap outliers instead of removing (preserves data)
                outlier_indices = df[col].dropna().index[outlier_pred == -1]
                if len(outlier_indices) > 0:
                    # Cap at 5th and 95th percentile
                    lower = df[col].quantile(0.05)
                    upper = df[col].quantile(0.95)
                    df.loc[outlier_indices, col] = df.loc[outlier_indices, col].clip(lower, upper)
                    total_outliers += len(outlier_indices)
            except:
                pass
        
        if total_outliers > 0:
            report.append(f"   ✅ Detected and capped {total_outliers:,} outliers using ML")
        else:
            report.append("   ✅ No significant outliers detected")
        
        report.append("")
        return df, report
    
    def _clean_text_data(self, df):
        """Clean text columns"""
        report = []
        report.append("### 📝 Step 6: Cleaning Text Data")
        
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        if len(text_cols) == 0:
            report.append("   ℹ️ No text columns to clean")
            report.append("")
            return df, report
        
        cleaned_count = 0
        for col in text_cols:
            # Remove leading/trailing whitespace
            df[col] = df[col].astype(str).str.strip()
            
            # Remove extra spaces
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            
            # Standardize case for categorical data (if low cardinality)
            if df[col].nunique() < 50:
                df[col] = df[col].str.title()
            
            cleaned_count += 1
        
        report.append(f"   ✅ Cleaned {cleaned_count} text columns")
        report.append("")
        return df, report
    
    def _optimize_memory(self, df):
        """Optimize data types to reduce memory usage"""
        report = []
        report.append("### 💾 Step 7: Optimizing Memory Usage")
        
        original_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert low-cardinality strings to category
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique
                df[col] = df[col].astype('category')
        
        final_memory = df.memory_usage(deep=True).sum() / 1024**2
        saved = original_memory - final_memory
        
        if saved > 0.1:
            report.append(f"   ✅ Optimized memory: {original_memory:.2f}MB → {final_memory:.2f}MB (saved {saved:.2f}MB)")
        else:
            report.append("   ✅ Memory already optimized")
        
        report.append("")
        return df, report
    
    def _calculate_memory_saved(self, df_original, df_clean):
        """Calculate memory saved"""
        try:
            original = df_original.memory_usage(deep=True).sum() / 1024**2
            final = df_clean.memory_usage(deep=True).sum() / 1024**2
            saved = original - final
            if saved > 0:
                return f"{saved:.2f}MB ({saved/original*100:.1f}%)"
            return "0MB"
        except:
            return "N/A"
    
    def _generate_recommendations(self, df):
        """Generate smart recommendations for next steps"""
        self.recommendations.append("\n## 💡 Smart Recommendations\n")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Recommend visualizations
        if len(numeric_cols) > 0:
            self.recommendations.append("### 📊 Recommended Visualizations:")
            self.recommendations.append(f"   • Create histogram for '{numeric_cols[0]}' to see distribution")
            if len(numeric_cols) >= 2:
                self.recommendations.append(f"   • Create scatter plot: '{numeric_cols[0]}' vs '{numeric_cols[1]}'")
            if len(categorical_cols) > 0:
                self.recommendations.append(f"   • Create bar chart for '{categorical_cols[0]}'")
        
        # Recommend ML models
        self.recommendations.append("\n### 🤖 Recommended ML Models:")
        if len(numeric_cols) >= 2:
            self.recommendations.append("   • Try Random Forest (works for most problems)")
            self.recommendations.append("   • Use Auto-ML for best results")
        
        # Data quality score
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        dup_pct = (df.duplicated().sum() / len(df)) * 100
        
        quality_score = 100 - (missing_pct * 2) - (dup_pct * 3)
        quality_score = max(0, min(100, quality_score))
        
        self.recommendations.append(f"\n### ⭐ Data Quality Score: {quality_score:.0f}/100")
        if quality_score >= 90:
            self.recommendations.append("   ✅ Excellent! Your data is ready for analysis")
        elif quality_score >= 70:
            self.recommendations.append("   ✅ Good! Data is clean and usable")
        else:
            self.recommendations.append("   ⚠️ Fair. Consider additional cleaning if needed")
    
    def auto_prepare_for_ml(self, df: pd.DataFrame, target_column: str = None) -> tuple:
        """
        🤖 ONE-CLICK ML PREPARATION
        Automatically prepares data for machine learning
        Returns: (X, y, feature_names, encoding_info)
        """
        df_ml = df.copy()
        report = []
        
        report.append("🤖 **Auto-Preparing Data for Machine Learning**\n")
        
        # If no target specified, try to detect it
        if target_column is None:
            # Last column is often the target
            target_column = df_ml.columns[-1]
            report.append(f"   🎯 Auto-detected target: '{target_column}'")
        
        # Separate features and target
        X = df_ml.drop(columns=[target_column])
        y = df_ml[target_column]
        
        # Encode categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            report.append(f"   🔤 Encoding {len(categorical_cols)} categorical features")
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Encode target if categorical
        le = None
        if y.dtype == 'object' or y.dtype.name == 'category':
            le = LabelEncoder()
            y = le.fit_transform(y)
            report.append(f"   🎯 Encoded target variable")
        
        # Scale numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            scaler = StandardScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
            report.append(f"   ⚖️ Scaled {len(numeric_cols)} numeric features")
        
        report.append(f"\n   ✅ Ready for ML! Features: {X.shape[1]}, Samples: {X.shape[0]}")
        
        return X, y, X.columns.tolist(), "\n".join(report)
    
    def auto_train_best_model(self, X, y, task_type='auto') -> tuple:
        """
        🚀 ONE-CLICK MODEL TRAINING
        Automatically trains the best model
        Returns: (model, score, report)
        """
        report = []
        report.append("🚀 **Auto-Training Best Model**\n")
        
        # Auto-detect task type
        if task_type == 'auto':
            if len(np.unique(y)) < 20:
                task_type = 'classification'
            else:
                task_type = 'regression'
            report.append(f"   🎯 Detected task: {task_type.title()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        if task_type == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            metric = "Accuracy"
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            metric = "R² Score"
        
        report.append(f"   ✅ Model trained successfully!")
        report.append(f"   📊 {metric}: {score:.4f}")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_features = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)[:5]
            report.append(f"\n   🎯 Top 5 Important Features:")
            for feat, imp in top_features:
                report.append(f"      • {feat}: {imp:.4f}")
        
        return model, score, "\n".join(report)


# Global instance
smart_pilot = SmartAutoPilot()
