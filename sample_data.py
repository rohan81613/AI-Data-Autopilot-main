
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

class SampleDatasets:
    def __init__(self):
        pass

    def load_sample(self, dataset_name):
        """Load a sample dataset by name"""
        if dataset_name == "iris":
            return self._create_iris_dataset()
        elif dataset_name == "titanic":
            return self._create_titanic_dataset()
        elif dataset_name == "housing":
            return self._create_housing_dataset()
        elif dataset_name == "wine quality":
            return self._create_wine_dataset()
        else:
            st.error(f"Unknown sample dataset: {dataset_name}")
            return None

    def _create_iris_dataset(self):
        """Create a sample Iris dataset"""
        np.random.seed(42)

        # Create Iris-like data
        n_samples = 150

        # Setosa
        setosa_sepal_length = np.random.normal(5.0, 0.4, 50)
        setosa_sepal_width = np.random.normal(3.4, 0.4, 50)
        setosa_petal_length = np.random.normal(1.5, 0.2, 50)
        setosa_petal_width = np.random.normal(0.3, 0.1, 50)
        setosa_species = ["setosa"] * 50

        # Versicolor
        versicolor_sepal_length = np.random.normal(6.0, 0.5, 50)
        versicolor_sepal_width = np.random.normal(2.8, 0.3, 50)
        versicolor_petal_length = np.random.normal(4.3, 0.5, 50)
        versicolor_petal_width = np.random.normal(1.3, 0.2, 50)
        versicolor_species = ["versicolor"] * 50

        # Virginica
        virginica_sepal_length = np.random.normal(6.5, 0.6, 50)
        virginica_sepal_width = np.random.normal(3.0, 0.3, 50)
        virginica_petal_length = np.random.normal(5.5, 0.6, 50)
        virginica_petal_width = np.random.normal(2.0, 0.3, 50)
        virginica_species = ["virginica"] * 50

        # Combine all data
        data = {
            'sepal_length': np.concatenate([setosa_sepal_length, versicolor_sepal_length, virginica_sepal_length]),
            'sepal_width': np.concatenate([setosa_sepal_width, versicolor_sepal_width, virginica_sepal_width]),
            'petal_length': np.concatenate([setosa_petal_length, versicolor_petal_length, virginica_petal_length]),
            'petal_width': np.concatenate([setosa_petal_width, versicolor_petal_width, virginica_petal_width]),
            'species': setosa_species + versicolor_species + virginica_species
        }

        # Introduce some missing values
        df = pd.DataFrame(data)
        missing_indices = np.random.choice(df.index, size=10, replace=False)
        df.loc[missing_indices[:5], 'sepal_width'] = np.nan
        df.loc[missing_indices[5:], 'petal_length'] = np.nan

        return df

    def _create_titanic_dataset(self):
        """Create a sample Titanic dataset"""
        np.random.seed(42)

        n_samples = 800

        # Generate synthetic Titanic-like data
        passenger_classes = np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.3, 0.5])
        ages = np.random.normal(35, 15, n_samples)
        ages = np.clip(ages, 0.5, 80)  # Clip to reasonable age range

        # Gender
        genders = np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35])

        # Fare (correlated with class)
        fares = []
        for pclass in passenger_classes:
            if pclass == 1:
                fare = np.random.normal(80, 30)
            elif pclass == 2:
                fare = np.random.normal(25, 10)
            else:
                fare = np.random.normal(10, 5)
            fares.append(max(0, fare))

        # Survival (influenced by class, gender, age)
        survival_prob = []
        for i in range(n_samples):
            prob = 0.4  # Base probability

            if genders[i] == 'female':
                prob += 0.4
            if passenger_classes[i] == 1:
                prob += 0.3
            elif passenger_classes[i] == 2:
                prob += 0.1
            if ages[i] < 16:
                prob += 0.2

            survival_prob.append(min(0.95, max(0.05, prob)))

        survived = np.random.binomial(1, survival_prob)

        # Siblings/Spouses and Parents/Children
        sibsp = np.random.poisson(0.5, n_samples)
        parch = np.random.poisson(0.4, n_samples)

        # Embarked
        embarked = np.random.choice(['C', 'Q', 'S'], n_samples, p=[0.2, 0.1, 0.7])

        data = {
            'survived': survived,
            'pclass': passenger_classes,
            'sex': genders,
            'age': ages,
            'sibsp': sibsp,
            'parch': parch,
            'fare': fares,
            'embarked': embarked
        }

        df = pd.DataFrame(data)

        # Introduce missing values
        missing_indices = np.random.choice(df.index, size=80, replace=False)
        df.loc[missing_indices[:30], 'age'] = np.nan
        df.loc[missing_indices[30:35], 'fare'] = np.nan
        df.loc[missing_indices[35:40], 'embarked'] = np.nan

        return df

    def _create_housing_dataset(self):
        """Create a sample housing dataset"""
        np.random.seed(42)

        n_samples = 500

        # Generate housing features
        bedrooms = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1])
        bathrooms = bedrooms * 0.5 + np.random.normal(0, 0.3, n_samples)
        bathrooms = np.clip(bathrooms, 1, 5)

        # Square footage (correlated with bedrooms)
        sqft = bedrooms * 500 + np.random.normal(0, 300, n_samples)
        sqft = np.clip(sqft, 500, 5000)

        # Age of house
        age = np.random.exponential(15, n_samples)
        age = np.clip(age, 0, 100)

        # Location factor
        location_score = np.random.normal(5, 2, n_samples)
        location_score = np.clip(location_score, 1, 10)

        # Garage
        has_garage = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])

        # Property type
        property_types = np.random.choice(
            ['House', 'Condo', 'Townhouse', 'Apartment'],
            n_samples,
            p=[0.6, 0.2, 0.15, 0.05]
        )

        # Price (function of other features with noise)
        base_price = (
            bedrooms * 50000 +
            bathrooms * 25000 +
            sqft * 150 +
            location_score * 20000 -
            age * 1000 +
            has_garage * 15000
        )

        # Add property type adjustment
        type_adjustment = pd.Series(property_types).map({
            'House': 0,
            'Condo': -30000,
            'Townhouse': -15000,
            'Apartment': -40000
        })

        price = base_price + type_adjustment + np.random.normal(0, 30000, n_samples)
        price = np.clip(price, 50000, 2000000)

        data = {
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'sqft': sqft,
            'age': age,
            'location_score': location_score,
            'has_garage': has_garage,
            'property_type': property_types,
            'price': price
        }

        df = pd.DataFrame(data)

        # Introduce missing values
        missing_indices = np.random.choice(df.index, size=25, replace=False)
        df.loc[missing_indices[:10], 'age'] = np.nan
        df.loc[missing_indices[10:15], 'location_score'] = np.nan
        df.loc[missing_indices[15:20], 'bathrooms'] = np.nan

        return df

    def _create_wine_dataset(self):
        """Create a sample wine quality dataset"""
        np.random.seed(42)

        n_samples = 400

        # Wine features
        fixed_acidity = np.random.normal(7.5, 1.2, n_samples)
        volatile_acidity = np.random.gamma(2, 0.3, n_samples)
        citric_acid = np.random.beta(2, 3, n_samples) * 0.8
        residual_sugar = np.random.exponential(2.5, n_samples)
        chlorides = np.random.gamma(1.5, 0.05, n_samples)
        free_sulfur_dioxide = np.random.normal(35, 15, n_samples)
        total_sulfur_dioxide = free_sulfur_dioxide + np.random.normal(100, 40, n_samples)
        density = np.random.normal(0.997, 0.002, n_samples)
        ph = np.random.normal(3.2, 0.3, n_samples)
        sulphates = np.random.normal(0.65, 0.15, n_samples)
        alcohol = np.random.normal(10.5, 1.2, n_samples)

        # Wine type
        wine_type = np.random.choice(['red', 'white'], n_samples, p=[0.6, 0.4])

        # Quality (influenced by other features)
        quality_score = (
            (alcohol - 8) * 0.3 +
            (14 - volatile_acidity * 10) * 0.2 +
            (citric_acid * 10) * 0.15 +
            (sulphates * 5) * 0.1 +
            np.random.normal(0, 1, n_samples)
        )

        # Convert to 0-10 scale
        quality = np.clip(np.round(quality_score + 6), 3, 9)

        data = {
            'fixed_acidity': fixed_acidity,
            'volatile_acidity': volatile_acidity,
            'citric_acid': citric_acid,
            'residual_sugar': residual_sugar,
            'chlorides': chlorides,
            'free_sulfur_dioxide': free_sulfur_dioxide,
            'total_sulfur_dioxide': total_sulfur_dioxide,
            'density': density,
            'ph': ph,
            'sulphates': sulphates,
            'alcohol': alcohol,
            'wine_type': wine_type,
            'quality': quality.astype(int)
        }

        df = pd.DataFrame(data)

        # Introduce missing values
        missing_indices = np.random.choice(df.index, size=20, replace=False)
        df.loc[missing_indices[:8], 'citric_acid'] = np.nan
        df.loc[missing_indices[8:12], 'residual_sugar'] = np.nan
        df.loc[missing_indices[12:16], 'ph'] = np.nan
        df.loc[missing_indices[16:20], 'sulphates'] = np.nan

        return df

    def get_available_datasets(self):
        """Get list of available sample datasets"""
        return ["iris", "titanic", "housing", "wine quality"]

    def get_dataset_description(self, dataset_name):
        """Get description of a sample dataset"""
        descriptions = {
            "iris": "Classic iris flower dataset with sepal/petal measurements and species classification (150 samples, 4 features + 1 target)",
            "titanic": "Titanic passenger survival dataset with demographics and ticket information (800 samples, 7 features + 1 target)",
            "housing": "Housing price prediction dataset with property features and location data (500 samples, 6 features + 1 target)",
            "wine quality": "Wine quality dataset with chemical properties and quality ratings (400 samples, 11 features + 2 targets)"
        }
        return descriptions.get(dataset_name.lower(), "No description available")
