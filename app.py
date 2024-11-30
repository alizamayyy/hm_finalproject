import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Customer Segregation", page_icon=":beer:", layout="wide", initial_sidebar_state="auto")

    # Load dataset
df = pd.read_csv('Mall_Customers.csv')
columns = ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']

#Functions
def show_csv(df):
    st.dataframe(
        df.style.background_gradient(cmap='Pastel2')
        .set_properties(**{'font-family': 'Segoe UI'})
        .hide(axis='index')
    )

def show_col_names():
    col_descriptions = {
        'CustomerID': 'Unique ID assigned to the customer',
        'Gender': 'Gender of the customer',
        'Age': 'Age of the customer',
        'Annual Income (k$)': 'Annual Income of the customer',
        'Spending Score (1-100)': 'Score assigned by the mall based on customer behavior and spending nature'
    }
    
    st.markdown("### Column Descriptions")
    for col, desc in col_descriptions.items():
        st.markdown(f"**{col}**  \n{desc}")

def clean_data(df):
    df.dropna(inplace=True) 
    df.drop_duplicates(inplace=True) 
    df.drop(columns=['CustomerID'], inplace=True)
    return df

def show_cleaned_data(df):
    st.subheader("Cleaning the Dataset")
    df = clean_data(df)
    col1, col2, col3 = st.columns(3, gap='large')
    with col1:
        st.write("__Handling Missing Values__")
        code = '''
        df.dropna(inplace=True)'''
        st.code(code, language="python")
 
        st.write(
        "The `df.dropna()` method eliminates any rows with missing values in the DataFrame, ensuring the data remains reliable. "
        "This step is crucial for conducting accurate analyses, and the use of `inplace=True` ensures that the original DataFrame is updated."
        )
        
    with col2:
        st.write("__Handling Duplicate Data__")
        code = '''
        df.drop_duplicates(inplace=True) '''
        st.code(code, language="python")
        
        st.write(
        "The `df.drop_duplicates()` function removes any duplicate rows from the DataFrame, ensuring that each record is unique. "
        "This process is essential for maintaining data integrity and avoiding skewed analysis results, and using `inplace=True` updates the DataFrame directly."
    )
        
    with col3:
        st.write("__Dropping 'CustomerID' Column__")
        code = '''
        df.drop(columns=['CustomerID'], inplace=True)'''
        st.code(code, language="python")
        
        st.write(
        "The `df.drop(columns=['CustomerID'], inplace=True)` method removes the 'CustomerID' column from the DataFrame. "
        "This is done to simplify the dataset by eliminating unnecessary information that may not contribute to the analysis."
        )
        
    st.write("\n")

def plot_histograms(df):
    st.write("### Distribution of Continuous Variables")
    
    # Create histograms for Age, Annual Income, and Spending Score
    numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    
    for col in numeric_cols:
        fig = px.histogram(
            df, 
            x=col,
            title=f'Distribution of {col}',
            template='simple_white',
            color_discrete_sequence=['#3498db']
        )
        fig.update_layout(
            xaxis_title=col,
            yaxis_title="Count",
            showlegend=False,
            title_x=0.5
        )
        st.plotly_chart(fig, use_container_width=True)

def plot_boxplots(df):
    st.write("### Box Plots of Continuous Variables")
    
    # Create box plots for Age, Annual Income, and Spending Score
    numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    
    fig = px.box(
        df,
        y=numeric_cols,
        title='Box Plots of Continuous Variables',
        template='simple_white',
        color_discrete_sequence=['#2ecc71']
    )
    fig.update_layout(
        showlegend=False,
        title_x=0.5,
        yaxis_title="Value"
    )
    st.plotly_chart(fig, use_container_width=True)

# Navigation bar in sidebar
page = st.sidebar.selectbox("Select a section:", ["Introduction", "Data Exploration and Preparation", "Analysis and Insights", "Conclusion"])

#Application
st.header("Customer Segmentation with K-Means")
st.markdown("<small>by Halimaw Magbeg</small>", unsafe_allow_html=True)


# Content based on navigation selection
if page == "Introduction":
    st.subheader("Introduction")


elif page == "Data Exploration and Preparation":
    st.subheader("Dataset Overview")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Data Exploration and Preparation", "Data Visualization"])
    
    with tab1:
        # Create two columns
        col1, col2 = st.columns([1, 1])  # 2:1 ratio for better layout
        
        # Show DataFrame in the left column
        with col1:
            show_csv(df)
        
        # Show column descriptions in the right column
        with col2:
            show_col_names()
        
        show_cleaned_data(df)
    
    with tab2:
        plot_histograms(df)
        plot_boxplots(df)

elif page == "Analysis and Insights":
    st.subheader("Analysis and Insights")
    # Add your analysis content here

elif page == "Conclusion":
    st.subheader("Conclusion and Recommendations")
    # Add your conclusion content here