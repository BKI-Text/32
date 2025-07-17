import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Configure Streamlit
st.set_page_config(
    page_title="Beverly Knits AI Supply Chain Planner",
    page_icon="🧶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main header
st.markdown('<h1 style="text-align: center; color: #2E86AB;">🧶 Beverly Knits AI Supply Chain Planner</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.selectbox("Select Page", ["Dashboard", "Data Overview", "Quick Analysis"])

if page == "Dashboard":
    st.header("📊 Executive Dashboard")
    
    # Create some demo metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Materials", "248", "↑ 12")
    
    with col2:
        st.metric("Active Suppliers", "45", "↑ 3")
    
    with col3:
        st.metric("Data Quality Score", "87%", "↑ 5%")
    
    with col4:
        st.metric("Cost Savings", "$125K", "↑ $15K")
    
    # Demo chart
    st.subheader("📈 Planning Overview")
    
    # Create sample data
    data = {
        'Category': ['Yarn', 'Fabric', 'Thread', 'Accessories', 'Trim'],
        'Count': [120, 85, 25, 12, 6],
        'Value': [450000, 320000, 85000, 35000, 18000]
    }
    
    df = pd.DataFrame(data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.pie(df, values='Count', names='Category', title='Materials by Category')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.bar(df, x='Category', y='Value', title='Value by Category')
        st.plotly_chart(fig2, use_container_width=True)

elif page == "Data Overview":
    st.header("📋 Data Overview")
    
    # Check if data directory exists
    data_path = Path("data/live")
    
    if data_path.exists():
        st.success("✅ Live data directory found!")
        
        # List CSV files
        csv_files = list(data_path.glob("*.csv"))
        
        if csv_files:
            st.subheader("📁 Available Data Files")
            
            for file in csv_files:
                with st.expander(f"📄 {file.name}"):
                    try:
                        df = pd.read_csv(file, encoding='utf-8-sig')
                        st.write(f"**Rows:** {len(df)}, **Columns:** {len(df.columns)}")
                        st.write("**Column Names:**", list(df.columns))
                        st.dataframe(df.head(), use_container_width=True)
                    except Exception as e:
                        st.error(f"Error reading file: {e}")
        else:
            st.warning("No CSV files found in data/live directory")
    else:
        st.error("❌ Live data directory not found")
        st.info("Expected path: data/live/")

elif page == "Quick Analysis":
    st.header("⚡ Quick Analysis")
    
    st.info("🚧 Analysis features will be available once the full system is running.")
    
    # Simple data upload demo
    st.subheader("📤 Upload Data for Quick Analysis")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! {len(df)} rows, {len(df.columns)} columns")
            
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.subheader("📈 Quick Stats")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Numeric Columns:**")
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                st.write(numeric_cols)
            
            with col2:
                st.write("**Text Columns:**")
                text_cols = df.select_dtypes(include=['object']).columns.tolist()
                st.write(text_cols)
            
            if numeric_cols:
                selected_col = st.selectbox("Select column for visualization:", numeric_cols)
                if selected_col:
                    fig = px.histogram(df, x=selected_col, title=f'Distribution of {selected_col}')
                    st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Footer
st.markdown("---")
st.markdown("**Beverly Knits AI Supply Chain Planner** - *Intelligent automation for textile manufacturing excellence*")