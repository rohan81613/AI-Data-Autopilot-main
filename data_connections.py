# Fixed data_connections.py with improved spacing and layout

import streamlit as st
import pandas as pd
import numpy as np
import json
import tempfile
from datetime import datetime
from pipeline_history import PipelineHistory
from sample_data import SampleDatasets

# Database and API connection imports with error handling
try:
    from sqlalchemy import create_engine, text

    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

try:
    import pymongo

    HAS_PYMONGO = True
except ImportError:
    HAS_PYMONGO = False

try:
    import psycopg2

    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

import warnings

warnings.filterwarnings('ignore')


class DataConnections:
    def __init__(self):
        self.history = PipelineHistory()
        self.sample_data = SampleDatasets()

    def render_data_connection_ui(self):
        """Render the data connection interface"""

        st.subheader("🔌 Data Connection Hub")

        # Connection type selector
        connection_types = [
            "📁 Local Files (CSV, Excel, Parquet, JSON)",
            "🗄️ SQL Database (MySQL, PostgreSQL, SQL Server)",
            "🍃 MongoDB",
            "🌐 REST API",
            "📝 Raw Text/JSON Input",
            "🎯 Sample Datasets"
        ]

        selected_connection = st.selectbox("Select Data Source", connection_types)

        # Render appropriate connection interface
        if "Local Files" in selected_connection:
            self._render_file_upload()
        elif "SQL Database" in selected_connection:
            self._render_sql_connection()
        elif "MongoDB" in selected_connection:
            self._render_mongodb_connection()
        elif "REST API" in selected_connection:
            self._render_api_connection()
        elif "Raw Text" in selected_connection:
            self._render_raw_input()
        elif "Sample Datasets" in selected_connection:
            self._render_sample_datasets()

    def _render_file_upload(self):
        """Render file upload interface"""
        st.markdown("### 📁 Upload Local Files")

        # File uploader with multiple formats
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['csv', 'xlsx', 'xls', 'parquet', 'json', 'txt'],
            accept_multiple_files=True,
            help="Supported formats: CSV, Excel, Parquet, JSON, Text"
        )

        if uploaded_files:
            st.markdown("**📋 Files to Process:**")

            # Show uploaded files
            files_info = []
            for file in uploaded_files:
                files_info.append({
                    'Filename': file.name,
                    'Size (KB)': f"{file.size / 1024:.1f}",
                    'Type': file.type
                })

            files_df = pd.DataFrame(files_info)
            st.dataframe(files_df, use_container_width=True, hide_index=True)

            # Process files
            col1, col2 = st.columns(2)

            with col1:
                if st.button("📊 Load All Files", type="primary"):
                    self._process_uploaded_files(uploaded_files)

            with col2:
                # File processing options
                with st.expander("⚙️ Processing Options"):
                    encoding = st.selectbox("Text Encoding", ["utf-8", "latin-1", "cp1252"])
                    separator = st.selectbox("CSV Separator", [",", ";", "|", "\t"])
                    skip_rows = st.number_input("Skip Rows", min_value=0, value=0)

    def _process_uploaded_files(self, files):
        """Process uploaded files and load into session state"""

        with st.spinner("Processing uploaded files..."):
            success_count = 0
            error_count = 0

            for file in files:
                try:
                    # Determine file type and process
                    file_extension = file.name.split('.')[-1].lower()
                    dataset_name = file.name.replace(f'.{file_extension}', '')

                    if file_extension == 'csv':
                        df = pd.read_csv(file, encoding='utf-8')
                    elif file_extension in ['xlsx', 'xls']:
                        df = pd.read_excel(file)
                    elif file_extension == 'parquet':
                        df = pd.read_parquet(file)
                    elif file_extension == 'json':
                        df = pd.read_json(file)
                    elif file_extension == 'txt':
                        # Try to read as CSV first
                        try:
                            df = pd.read_csv(file, encoding='utf-8')
                        except:
                            # Read as text
                            content = str(file.read(), "utf-8")
                            df = pd.DataFrame({'text': [content]})
                    else:
                        st.error(f"Unsupported file type: {file_extension}")
                        continue

                    # Store dataset
                    st.session_state.datasets[dataset_name] = df
                    success_count += 1

                    # Log to history
                    self.history.log_step(
                        "File Upload",
                        f"Successfully loaded {file.name}",
                        {
                            "filename": file.name,
                            "rows": len(df),
                            "columns": len(df.columns),
                            "file_size_kb": file.size / 1024
                        },
                        "success"
                    )

                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                    error_count += 1

                    # Log error
                    self.history.log_step(
                        "File Upload",
                        f"Failed to load {file.name}",
                        {"filename": file.name, "error": str(e)},
                        "error"
                    )

            # Show results
            if success_count > 0:
                st.success(f"✅ Successfully loaded {success_count} file(s)")

                # Show data preview with improved spacing
                self._render_data_preview()

            if error_count > 0:
                st.warning(f"⚠️ Failed to load {error_count} file(s)")

    def _render_data_preview(self):
        """Render data preview with improved spacing and layout"""

        if not st.session_state.datasets:
            return

        st.markdown("---")
        st.markdown("## 📊 **Loaded Datasets Preview**")

        # Dataset selector
        dataset_names = list(st.session_state.datasets.keys())
        selected_dataset = st.selectbox(
            "**Select Dataset to Preview**",
            dataset_names,
            key="preview_dataset_selector"
        )

        if selected_dataset:
            df = st.session_state.datasets[selected_dataset]

            # Dataset summary metrics
            st.markdown("### 📋 **Dataset Summary**")

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("📏 **Rows**", f"{len(df):,}")
            with col2:
                st.metric("📊 **Columns**", len(df.columns))
            with col3:
                missing_count = df.isnull().sum().sum()
                st.metric("🕳️ **Missing Values**", f"{missing_count:,}")
            with col4:
                memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                st.metric("💾 **Memory (MB)**", f"{memory_mb:.1f}")
            with col5:
                duplicates = df.duplicated().sum()
                st.metric("🔄 **Duplicates**", f"{duplicates:,}")

            st.markdown("---")

            # Create tabs for organized display
            tab1, tab2, tab3 = st.tabs(["📄 **Data Preview**", "🔍 **Column Info**", "📈 **Quick Stats**"])

            with tab1:
                st.markdown("### 📄 **Sample Data** (First 10 rows)")

                # Data preview with full width and better formatting
                st.dataframe(
                    df.head(10),
                    use_container_width=True,
                    height=400,  # Fixed height for better display
                    column_config={
                        col: st.column_config.TextColumn(
                            width="medium" if len(str(df[col].iloc[0])) > 20 else "small"
                        ) for col in df.columns
                    }
                )

                # Show data types below the preview
                st.markdown("### 🏷️ **Data Types Overview**")

                # Create a more compact data types display
                dtype_data = []
                for col in df.columns:
                    dtype_data.append({
                        'Column': col,
                        'Type': str(df[col].dtype),
                        'Sample': str(df[col].iloc[0]) if len(df) > 0 else 'N/A'
                    })

                dtype_df = pd.DataFrame(dtype_data)
                st.dataframe(dtype_df, use_container_width=True, height=300)

            with tab2:
                st.markdown("### 🔍 **Detailed Column Information**")

                # Column analysis with better spacing
                col_info_data = []
                for col in df.columns:
                    non_null_count = df[col].count()
                    null_count = df[col].isnull().sum()
                    unique_count = df[col].nunique()

                    col_info_data.append({
                        'Column Name': col,
                        'Data Type': str(df[col].dtype),
                        'Non-Null': f"{non_null_count:,}",
                        'Null': f"{null_count:,}",
                        'Null %': f"{(null_count / len(df) * 100):.1f}%",
                        'Unique': f"{unique_count:,}",
                        'Unique %': f"{(unique_count / len(df) * 100):.1f}%"
                    })

                col_info_df = pd.DataFrame(col_info_data)

                # Display with improved formatting
                st.dataframe(
                    col_info_df,
                    use_container_width=True,
                    height=400,
                    column_config={
                        "Column Name": st.column_config.TextColumn(width="large"),
                        "Data Type": st.column_config.TextColumn(width="medium"),
                        "Non-Null": st.column_config.TextColumn(width="small"),
                        "Null": st.column_config.TextColumn(width="small"),
                        "Null %": st.column_config.ProgressColumn("Null %", min_value=0, max_value=100,
                                                                  format="%.1f%%"),
                        "Unique": st.column_config.TextColumn(width="small"),
                        "Unique %": st.column_config.ProgressColumn("Unique %", min_value=0, max_value=100,
                                                                    format="%.1f%%")
                    }
                )

            with tab3:
                st.markdown("### 📈 **Statistical Summary**")

                # Separate numeric and categorical analysis
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns

                if len(numeric_cols) > 0:
                    st.markdown("#### 🔢 **Numeric Columns**")
                    numeric_stats = df[numeric_cols].describe()
                    st.dataframe(numeric_stats, use_container_width=True, height=300)

                if len(categorical_cols) > 0:
                    st.markdown("#### 🏷️ **Categorical Columns**")

                    cat_stats = []
                    for col in categorical_cols[:10]:  # Show first 10 categorical columns
                        top_category = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A'
                        cat_stats.append({
                            'Column': col,
                            'Unique Values': df[col].nunique(),
                            'Most Frequent': str(top_category),
                            'Frequency': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0,
                            'Missing': df[col].isnull().sum()
                        })

                    cat_stats_df = pd.DataFrame(cat_stats)
                    st.dataframe(cat_stats_df, use_container_width=True, height=300)

            # Data quality indicators
            st.markdown("---")
            st.markdown("### 🎯 **Data Quality Indicators**")

            col1, col2, col3 = st.columns(3)

            with col1:
                # Completeness score
                completeness = ((df.count().sum()) / (len(df) * len(df.columns))) * 100
                if completeness >= 95:
                    st.success(f"🟢 **Completeness**: {completeness:.1f}% (Excellent)")
                elif completeness >= 80:
                    st.warning(f"🟡 **Completeness**: {completeness:.1f}% (Good)")
                else:
                    st.error(f"🔴 **Completeness**: {completeness:.1f}% (Needs Attention)")

            with col2:
                # Consistency score (no duplicates)
                duplicate_rate = (df.duplicated().sum() / len(df)) * 100
                if duplicate_rate == 0:
                    st.success(f"🟢 **Uniqueness**: 100% (No Duplicates)")
                elif duplicate_rate < 5:
                    st.warning(f"🟡 **Uniqueness**: {100 - duplicate_rate:.1f}% (Low Duplicates)")
                else:
                    st.error(f"🔴 **Uniqueness**: {100 - duplicate_rate:.1f}% (High Duplicates)")

            with col3:
                # Data type consistency
                mixed_types = 0
                for col in df.columns:
                    if df[col].dtype == 'object':
                        # Check if object column contains mixed types
                        sample = df[col].dropna().head(100)
                        types = set([type(x).__name__ for x in sample])
                        if len(types) > 1:
                            mixed_types += 1

                type_consistency = ((len(df.columns) - mixed_types) / len(df.columns)) * 100
                if type_consistency >= 95:
                    st.success(f"🟢 **Type Consistency**: {type_consistency:.1f}%")
                elif type_consistency >= 80:
                    st.warning(f"🟡 **Type Consistency**: {type_consistency:.1f}%")
                else:
                    st.error(f"🔴 **Type Consistency**: {type_consistency:.1f}%")

    def _render_sql_connection(self):
        """Render SQL database connection interface"""

        if not HAS_SQLALCHEMY:
            st.error("SQLAlchemy not available. Please install: pip install sqlalchemy")
            return

        st.markdown("### 🗄️ SQL Database Connection")

        # Database type selector
        db_types = {
            "PostgreSQL": "postgresql://username:password@host:port/database",
            "MySQL": "mysql+pymysql://username:password@host:port/database",
            "SQL Server": "mssql+pyodbc://username:password@host:port/database?driver=ODBC+Driver+17+for+SQL+Server",
            "SQLite": "sqlite:///path/to/database.db"
        }

        selected_db = st.selectbox("Database Type", list(db_types.keys()))

        # Connection string input
        connection_string = st.text_input(
            "Connection String",
            value=db_types[selected_db],
            help="Enter your database connection string"
        )

        # SQL query input
        sql_query = st.text_area(
            "SQL Query",
            value="SELECT * FROM your_table LIMIT 100;",
            height=100,
            help="Enter your SQL query to fetch data"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔍 Test Connection"):
                self._test_sql_connection(connection_string)

        with col2:
            if st.button("📊 Load Data", type="primary"):
                self._load_sql_data(connection_string, sql_query)

    def _test_sql_connection(self, connection_string):
        """Test SQL database connection"""
        try:
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                st.success("✅ Connection successful!")
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")

    def _load_sql_data(self, connection_string, query):
        """Load data from SQL database"""
        try:
            with st.spinner("Connecting to database and executing query..."):
                engine = create_engine(connection_string)
                df = pd.read_sql_query(query, engine)

                # Store dataset
                dataset_name = f"sql_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state.datasets[dataset_name] = df

                # Log success
                self.history.log_step(
                    "SQL Database Load",
                    f"Successfully loaded data from SQL query",
                    {
                        "dataset_name": dataset_name,
                        "rows": len(df),
                        "columns": len(df.columns),
                        "query_preview": query[:100] + "..." if len(query) > 100 else query
                    },
                    "success"
                )

                st.success(f"✅ Data loaded successfully! Dataset: {dataset_name}")

                # Show preview
                self._render_data_preview()

        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            self.history.log_step(
                "SQL Database Load",
                "Failed to load SQL data",
                {"error": str(e)},
                "error"
            )

    def _render_mongodb_connection(self):
        """Render MongoDB connection interface"""

        if not HAS_PYMONGO:
            st.error("PyMongo not available. Please install: pip install pymongo")
            return

        st.markdown("### 🍃 MongoDB Connection")

        # MongoDB connection details
        col1, col2 = st.columns(2)

        with col1:
            mongo_uri = st.text_input(
                "MongoDB URI",
                value="mongodb://username:password@localhost:27017/",
                help="MongoDB connection URI"
            )
            database_name = st.text_input("Database Name", value="your_database")

        with col2:
            collection_name = st.text_input("Collection Name", value="your_collection")
            limit = st.number_input("Document Limit", value=1000, min_value=1, max_value=10000)

        # MongoDB query (optional)
        mongo_query = st.text_area(
            "Query (JSON, optional)",
            value="{}",
            height=100,
            help="MongoDB query in JSON format, e.g., {'status': 'active'}"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔍 Test MongoDB Connection"):
                self._test_mongodb_connection(mongo_uri, database_name)

        with col2:
            if st.button("📊 Load MongoDB Data", type="primary"):
                self._load_mongodb_data(mongo_uri, database_name, collection_name, mongo_query, limit)

    def _test_mongodb_connection(self, uri, db_name):
        """Test MongoDB connection"""
        try:
            client = pymongo.MongoClient(uri)
            db = client[db_name]
            # Test connection
            db.list_collection_names()
            st.success("✅ MongoDB connection successful!")

            # Show collections
            collections = db.list_collection_names()
            st.write(f"Available collections: {collections}")

        except Exception as e:
            st.error(f"❌ MongoDB connection failed: {str(e)}")

    def _load_mongodb_data(self, uri, db_name, collection_name, query, limit):
        """Load data from MongoDB"""
        try:
            with st.spinner("Connecting to MongoDB and loading data..."):
                client = pymongo.MongoClient(uri)
                db = client[db_name]
                collection = db[collection_name]

                # Parse query
                query_dict = json.loads(query) if query.strip() != "{}" else {}

                # Fetch documents
                cursor = collection.find(query_dict).limit(limit)
                documents = list(cursor)

                if not documents:
                    st.warning("No documents found matching the query.")
                    return

                # Convert to DataFrame
                df = pd.DataFrame(documents)

                # Handle ObjectId and other MongoDB types
                for col in df.columns:
                    if col == '_id':
                        df[col] = df[col].astype(str)

                # Store dataset
                dataset_name = f"mongodb_{collection_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state.datasets[dataset_name] = df

                # Log success
                self.history.log_step(
                    "MongoDB Load",
                    f"Successfully loaded data from MongoDB collection",
                    {
                        "dataset_name": dataset_name,
                        "collection": collection_name,
                        "rows": len(df),
                        "columns": len(df.columns)
                    },
                    "success"
                )

                st.success(f"✅ MongoDB data loaded successfully! Dataset: {dataset_name}")

                # Show preview
                self._render_data_preview()

        except Exception as e:
            st.error(f"❌ Error loading MongoDB data: {str(e)}")
            self.history.log_step(
                "MongoDB Load",
                "Failed to load MongoDB data",
                {"error": str(e)},
                "error"
            )

    def _render_api_connection(self):
        """Render REST API connection interface"""

        if not HAS_REQUESTS:
            st.error("Requests library not available. Please install: pip install requests")
            return

        st.markdown("### 🌐 REST API Connection")

        # API connection details
        api_url = st.text_input(
            "API Base URL",
            value="https://jsonplaceholder.typicode.com/posts",
            help="Base URL for the REST API"
        )

        # Authentication
        auth_method = st.selectbox("Authentication", ["None", "API Key", "Bearer Token", "Basic Auth"])

        headers = {}
        if auth_method == "API Key":
            api_key = st.text_input("API Key", type="password")
            key_name = st.text_input("API Key Header Name", value="X-API-Key")
            if api_key:
                headers[key_name] = api_key

        elif auth_method == "Bearer Token":
            token = st.text_input("Bearer Token", type="password")
            if token:
                headers["Authorization"] = f"Bearer {token}"

        # Additional headers
        custom_headers = st.text_area(
            "Custom Headers (JSON format)",
            value='{"Content-Type": "application/json"}',
            help="Additional headers in JSON format"
        )

        try:
            if custom_headers.strip():
                additional_headers = json.loads(custom_headers)
                headers.update(additional_headers)
        except json.JSONDecodeError:
            st.warning("Invalid JSON format in custom headers")

        # JSON path for data extraction
        json_path = st.text_input(
            "JSON Path (optional)",
            value="",
            help="Path to extract data from response, e.g., 'data.results' or leave empty for root"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔍 Test API Connection"):
                self._test_api_connection(api_url, headers)

        with col2:
            if st.button("📊 Load API Data", type="primary"):
                self._load_api_data(api_url, headers, json_path)

    def _test_api_connection(self, url, headers):
        """Test API connection"""
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                st.success(f"✅ API connection successful! Status: {response.status_code}")

                # Show sample response
                try:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        st.write("Sample response (first item):")
                        st.json(data[0] if len(data) > 0 else {})
                    elif isinstance(data, dict):
                        st.write("Sample response:")
                        st.json(data)
                except:
                    st.write(f"Response preview: {response.text[:200]}...")
            else:
                st.error(f"❌ API request failed with status: {response.status_code}")

        except Exception as e:
            st.error(f"❌ API connection failed: {str(e)}")

    def _load_api_data(self, url, headers, json_path):
        """Load data from REST API"""
        try:
            with st.spinner("Fetching data from API..."):
                response = requests.get(url, headers=headers, timeout=30)

                if response.status_code != 200:
                    st.error(f"API request failed with status: {response.status_code}")
                    return

                data = response.json()

                # Extract data using JSON path if provided
                if json_path:
                    path_parts = json_path.split('.')
                    for part in path_parts:
                        if isinstance(data, dict) and part in data:
                            data = data[part]
                        else:
                            st.error(f"JSON path '{json_path}' not found in response")
                            return

                # Convert to DataFrame
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    df = pd.DataFrame([data])
                else:
                    st.error("API response is not in a format that can be converted to DataFrame")
                    return

                # Store dataset
                dataset_name = f"api_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state.datasets[dataset_name] = df

                # Log success
                self.history.log_step(
                    "API Data Load",
                    f"Successfully loaded data from API",
                    {
                        "dataset_name": dataset_name,
                        "api_url": url,
                        "rows": len(df),
                        "columns": len(df.columns)
                    },
                    "success"
                )

                st.success(f"✅ API data loaded successfully! Dataset: {dataset_name}")

                # Show preview
                self._render_data_preview()

        except Exception as e:
            st.error(f"❌ Error loading API data: {str(e)}")
            self.history.log_step(
                "API Data Load",
                "Failed to load API data",
                {"error": str(e)},
                "error"
            )

    def _render_raw_input(self):
        """Render raw text/JSON input interface"""
        st.markdown("### 📝 Raw Text/JSON Input")

        input_format = st.selectbox("Input Format", ["JSON", "CSV Text", "Plain Text"])

        raw_input = st.text_area(
            f"Paste your {input_format} data here",
            height=300,
            help=f"Paste your {input_format} data directly"
        )

        if input_format == "JSON":
            st.info("For JSON arrays, each object will become a row. For single JSON object, it will become one row.")
        elif input_format == "CSV Text":
            separator = st.selectbox("Separator", [",", ";", "|", "\t"])

        if st.button("📊 Process Input", type="primary") and raw_input.strip():
            self._process_raw_input(raw_input, input_format, locals().get('separator', ','))

    def _process_raw_input(self, raw_input, input_format, separator=','):
        """Process raw input and convert to DataFrame"""
        try:
            with st.spinner("Processing input data..."):

                if input_format == "JSON":
                    data = json.loads(raw_input)

                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    elif isinstance(data, dict):
                        df = pd.DataFrame([data])
                    else:
                        st.error("JSON must be an object or array of objects")
                        return

                elif input_format == "CSV Text":
                    from io import StringIO
                    df = pd.read_csv(StringIO(raw_input), separator=separator)

                elif input_format == "Plain Text":
                    lines = raw_input.split('\n')
                    df = pd.DataFrame({'text': lines})

                # Store dataset
                dataset_name = f"raw_input_{input_format.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state.datasets[dataset_name] = df

                # Log success
                self.history.log_step(
                    "Raw Input Processing",
                    f"Successfully processed {input_format} input",
                    {
                        "dataset_name": dataset_name,
                        "input_format": input_format,
                        "rows": len(df),
                        "columns": len(df.columns)
                    },
                    "success"
                )

                st.success(f"✅ {input_format} data processed successfully! Dataset: {dataset_name}")

                # Show preview
                self._render_data_preview()

        except Exception as e:
            st.error(f"❌ Error processing {input_format} input: {str(e)}")
            self.history.log_step(
                "Raw Input Processing",
                f"Failed to process {input_format} input",
                {"error": str(e)},
                "error"
            )

    def _render_sample_datasets(self):
        """Render sample datasets interface"""
        st.markdown("### 🎯 Sample Datasets")

        available_datasets = {
            "Iris": "Classic iris flower dataset (150 rows, 5 columns)",
            "Titanic": "Titanic passenger data (891 rows, 12 columns)",
            "Housing": "Boston housing prices (506 rows, 14 columns)",
            "Wine Quality": "Wine quality dataset (1599 rows, 12 columns)"
        }

        selected_sample = st.selectbox("Select Sample Dataset", list(available_datasets.keys()))

        if selected_sample:
            st.info(f"**{selected_sample}**: {available_datasets[selected_sample]}")

            if st.button(f"📊 Load {selected_sample} Dataset", type="primary"):
                try:
                    df = self.sample_data.load_sample(selected_sample.lower())

                    if df is not None:
                        dataset_name = f"sample_{selected_sample.lower()}"
                        st.session_state.datasets[dataset_name] = df

                        # Log success
                        self.history.log_step(
                            "Sample Dataset Load",
                            f"Successfully loaded {selected_sample} sample dataset",
                            {
                                "dataset_name": dataset_name,
                                "sample_type": selected_sample,
                                "rows": len(df),
                                "columns": len(df.columns)
                            },
                            "success"
                        )

                        st.success(f"✅ {selected_sample} dataset loaded successfully!")

                        # Show preview
                        self._render_data_preview()
                    else:
                        st.error(f"Failed to load {selected_sample} dataset")

                except Exception as e:
                    st.error(f"Error loading {selected_sample}: {str(e)}")
                    self.history.log_step(
                        "Sample Dataset Load",
                        f"Failed to load {selected_sample} sample dataset",
                        {"error": str(e)},
                        "error"
                    )