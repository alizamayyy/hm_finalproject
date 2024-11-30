import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

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
    
    # Create three columns
    col1, col2, col3 = st.columns(3)
    
    # Create histograms for Age, Annual Income, and Spending Score
    numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    columns = [col1, col2, col3]  # Map columns to their respective plots
    
    # Create each histogram in its respective column
    for col_name, column in zip(numeric_cols, columns):
        with column:
            fig = px.histogram(
                df, 
                x=col_name,
                title=f'Distribution of {col_name}',
                template='simple_white',
                color_discrete_sequence=['#3498db']
            )
            fig.update_layout(
                xaxis_title=col_name,
                yaxis_title="Count",
                showlegend=False,
                title_x=0.5,
                height=400  # Fixed height for better alignment
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

def plot_donut(df):
    st.write("### Gender Distribution")
    
    # Calculate gender distribution
    gender_counts = df['Gender'].value_counts()
    
    # Create donut chart
    fig = px.pie(
        values=gender_counts.values,
        names=gender_counts.index,
        title='Distribution of Customer Gender',
        hole=0.6,  # This makes it a donut chart (0.6 = 60% hole)
        template='simple_white',
        color_discrete_sequence=['#FF69B4', '#4169E1']  # Pink and Blue colors
    )
    
    # Update layout
    fig.update_layout(
        title_x=0.5,
        annotations=[dict(text='Gender', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

def age_vs_spending_score(df):
    # Create the joint plot
    g = sns.jointplot(x="Age", y="Spending Score (1-100)", data=df, kind='reg', height=8, color='#FF69B4', space=0)
    
    # Display the plot in Streamlit
    st.pyplot(g.figure)
    
    # Clear the figure to prevent memory issues
    plt.clf()

def age_vs_annual_income(df):
    # Create the joint plot
    g = sns.jointplot(x=df["Age"], y=df["Annual Income (k$)"], kind='hex', color='#FF69B4', height=8, ratio=5, space=0)
    
    # Display the plot in Streamlit
    st.pyplot(g.figure)
    
    # Clear the figure to prevent memory issues
    plt.clf()

def spending_score_vs_annual_income(df):
    # Create the joint plot
    g = sns.JointGrid(data=df, height=8, x="Annual Income (k$)", y="Spending Score (1-100)", space=0.1)
    g.plot_joint(sns.kdeplot, fill=True, thresh=0, color='#FF69B4')
    g.plot_marginals(sns.histplot, color='#FF69B4', alpha=1, bins=20)
    
    # Display the plot in Streamlit
    st.pyplot(g.figure)
    
    # Clear the figure to prevent memory issues
    plt.clf()

def scatter_age_vs_annual_income(df):
    # Create the scatter plot with regression line grouped by gender
    fig = px.scatter(
        df,
        x="Age",
        y="Annual Income (k$)",
        color="Gender",
        trendline="ols",  # Ordinary Least Squares regression
        template="simple_white",
        color_discrete_sequence=['#FF69B4', '#4169E1']  # Pink and Blue colors
    )
    
    # Update layout
    fig.update_layout(
        title='',
        title_x=0.5,
        height=400,
        xaxis_title="Age",
        yaxis_title="Annual Income (k$)",
        legend_title="Gender"
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

def scatter_age_vs_spending_score(df):
    # Create the scatter plot with regression line grouped by gender
    fig = px.scatter(
        df,
        x="Age",
        y="Spending Score (1-100)",
        color="Gender",
        trendline="ols",  # Ordinary Least Squares regression
        template="simple_white",
        color_discrete_sequence=['#FF69B4', '#4169E1']  # Pink and Blue colors
    )
    
    # Update layout
    fig.update_layout(
        title='',
        title_x=0.5,
        height=400,
        xaxis_title="Age",
        yaxis_title="Spending Score (1-100)",
        legend_title="Gender"
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

def scatter_annual_income_vs_spending_score(df):
    # Create the scatter plot with regression line grouped by gender
    fig = px.scatter(
        df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color="Gender",
        trendline="ols",  # Ordinary Least Squares regression
        template="simple_white",
        color_discrete_sequence=['#FF69B4', '#4169E1']  # Pink and Blue colors
    )
    
    # Update layout
    fig.update_layout(
        title='',
        title_x=0.5,
        height=400,
        xaxis_title="Annual Income (k$)",
        yaxis_title="Spending Score (1-100)",
        legend_title="Gender"
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

def correlation_heatmap(df):
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Create a figure with specified size
    plt.figure(figsize=(10, 6))
    
    # Create the heatmap
    heatmap = sns.heatmap(numeric_df.corr(), vmin=-1, vmax=1, annot=True, cmap='viridis')
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 18}, pad=12)
    
    # Save heatmap as .png file
    plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
    
    # Display the plot in Streamlit
    st.pyplot(plt)
    
    # Clear the figure to prevent memory issues
    plt.clf()

def standardize_variables(df):
    col_names = ['Annual Income (k$)', 'Age', 'Spending Score (1-100)']
    sd = StandardScaler()
    features = df[col_names]
    scaler = sd.fit(features.values)
    features = scaler.transform(features.values)
    scaled_features = pd.DataFrame(features, columns=col_names)
    return scaled_features

def one_hot_encoding(df):
    # Standardize the variables first
    scaled_features = standardize_variables(df)
    
    # One-hot encoding for the 'Gender' column
    gender = df['Gender']
    newdf = scaled_features.join(gender)
    newdf = pd.get_dummies(newdf, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None)
    newdf = newdf.drop(['Gender_Male'], axis=1)
    
    # Return the one-hot encoded DataFrame
    return newdf

def pcacomponents(df):
    newdf = one_hot_encoding(df)
    pca = PCA(n_components=4)
    principalComponents = pca.fit_transform(newdf)
    features = range(pca.n_components_)
    plt.figure(figsize=(16,8))
    plt.bar(features, pca.explained_variance_ratio_, color='blue')  # Changed to solid blue
    plt.xlabel('PCA features')
    plt.ylabel('variance %')
    plt.xticks(features)

    # Create a DataFrame for PCA components
    PCA_components = pd.DataFrame(principalComponents)

    # Display the plot in Streamlit
    st.pyplot(plt)  # Display the plot
    
    return PCA_components  # Return the PCA components

def elbow_method(df):
    # Call pcacomponents to get PCA_components
    PCA_components = pcacomponents(df)

    # Check if PCA_components is None
    if PCA_components is None:
        st.error("PCA components could not be generated.")
        return

    # Calculate inertia for different values of k
    inertia = []
    k_values = range(1, 11)
    for k in k_values:
        model = KMeans(n_clusters=k, init='k-means++', random_state=42)
        model.fit(PCA_components.iloc[:, :2])
        inertia.append(model.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertia, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.xticks(k_values)
    plt.grid()
    
    # Display the plot in Streamlit
    st.pyplot(plt)  # Display the plot


def silhouette_coefficient_metric(df):
    PCA_components = pcacomponents(df)

    # Check if PCA_components is None
    if PCA_components is None:
        st.error("PCA components could not be generated.")
        return

    # Calculate silhouette scores for different values of k
    silhouette_scores = []
    k_values = range(2, 11)  # Start from 2 because silhouette score requires at least 2 clusters
    for k in k_values:
        model = KMeans(n_clusters=k, init='k-means++', random_state=42)
        clusters = model.fit_predict(PCA_components.iloc[:, :2])
        score = silhouette_score(PCA_components.iloc[:, :2], clusters)
        silhouette_scores.append(score)

    # Plot the silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.title('Silhouette Coefficient for Different k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.xticks(k_values)
    plt.grid()
    
    # Display the plot in Streamlit
    st.pyplot(plt)  # Display the plot

def cluster_visualization(df):
    newdf = one_hot_encoding(df)
    PCA_components = pcacomponents(df)
    model = KMeans(n_clusters=4, init='k-means++', random_state=42)
    clusters = model.fit_predict(PCA_components.iloc[:,:2])
    newdf["label"] = clusters
 
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(newdf.Age[newdf["label"] == 0], newdf["Annual Income (k$)"][newdf["label"] == 0], newdf["Spending Score (1-100)"][newdf["label"] == 0], c='blue', s=80)
    ax.scatter(newdf.Age[newdf["label"] == 1], newdf["Annual Income (k$)"][newdf["label"] == 1], newdf["Spending Score (1-100)"][newdf["label"] == 1], c='red', s=80)
    ax.scatter(newdf.Age[newdf["label"] == 2], newdf["Annual Income (k$)"][newdf["label"] == 2], newdf["Spending Score (1-100)"][newdf["label"] == 2], c='green', s=80)
    ax.scatter(newdf.Age[newdf["label"] == 3], newdf["Annual Income (k$)"][newdf["label"] == 3], newdf["Spending Score (1-100)"][newdf["label"] == 3], c='purple', s=80)  # Changed color for clarity
    ax.set_xlabel("Age")
    ax.set_ylabel("Annual Income (k$)")
    ax.set_zlabel("Spending Score (1-100)")
    
    # Display the plot in Streamlit
    st.pyplot(fig)  # Use st.pyplot instead of plt.show()
    

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
    tab1, tab2, tab3 = st.tabs(["Data Exploration and Preparation", "Data Visualization", "Relationships and Patterns"])
    
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
        col1, col2 = st.columns(2)
        with col1:
            plot_boxplots(df)
        with col2:
            plot_donut(df)

    with tab3:
        st.subheader("Relationship between Features")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("### Age vs. Spending Score")
            age_vs_spending_score(df)
        with col2:
            st.write("### Age vs. Annual Income")
            age_vs_annual_income(df)
        with col3:
            st.write("### Spending Score vs. Annual Income")
            spending_score_vs_annual_income(df)
        
        st.subheader("Relationship with Linear Regression Line of Best Fit")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("### Age vs. Annual Income")
            scatter_age_vs_annual_income(df)
        with col2:
            st.write("### Age vs. Spending Score")
            scatter_age_vs_spending_score(df)
        with col3:
            st.write("### Annual Income vs. Annual Income")
            scatter_annual_income_vs_spending_score(df)

        st.subheader("Correlation between Features")
        col1, col2, col3 = st.columns(3)
        with col2:
            st.write("### Correlation Heatmap")
            correlation_heatmap(df)
        

elif page == "Analysis and Insights":
    st.subheader("Analysis and Insights")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Data Pre-processing", "K-Means Clustering"])
    with tab1:
        st.markdown("<p style='font-size: 15px;'>Since gender is a categorical variable, it needs to be encoded and converted into numeric. All other variables will be scaled to follow a normal distribution before being fed into the model. We will standardize these variables with a mean of 0 and a standard deviation of 1.</p>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Standardizing variables")
            st.markdown("<p style='font-size: 15px;'>First, let's standardize all variables in the dataset to get them around the same scale.</p>", unsafe_allow_html=True)
            standardized_df = standardize_variables(df)  # Call the function
            # Apply styling when displaying
            st.dataframe(standardized_df.style.background_gradient(cmap='plasma').set_properties(**{'font-family': 'Segoe UI'}))  # Display the styled DataFrame
        with col2:
            st.write("### One-Hot Encoding")
            st.markdown("<p style='font-size: 15px;'>Second, (to be added)</p>", unsafe_allow_html=True)
            # Call the one_hot_encoding function and store the result
            one_hot_encoded_df = one_hot_encoding(df)
                    
            # Display the one-hot encoded DataFrame without styling
            st.dataframe(one_hot_encoded_df.style.background_gradient(cmap='plasma').set_properties(**{'font-family': 'Segoe UI'}))
    
    with tab2:
        st.markdown("<p style='font-size: 15px;'>(K-Means clustering intro)</p>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Implementation of PCA")
            st.markdown("<p style='font-size: 15px;'>PCA is a technique that helps us reduce the dimension of a dataset. When we run PCA on a data frame, new components are created. These components explain the maximum variance in the model. We can select a subset of these variables and include them into the K-means model.</p>", unsafe_allow_html=True)
            pcacomponents(df)
            st.markdown("<p style='font-size: 15px;'>This chart shows us each PCA component, along with its variance. Based on this visualization, we can see that the first two PCA components explain more than 70% of the dataset variance. We can feed these two components into the model.</p>", unsafe_allow_html=True)
        with col2:
            st.markdown("### Elbow Method for the Clustering Model")
            elbow_method(df)
            st.markdown("<p style='font-size: 15px;'>Using the elbow method We can observe that the optimal number of clusters is k=4. Now we can run a K-Means using as n_clusters the number 4.</p>", unsafe_allow_html=True)
        st.markdown("### Visualization of clusters built by the model")
        cluster_visualization(df)

elif page == "Conclusion":
    st.subheader("Conclusion and Recommendations")
    # Add your conclusion content here