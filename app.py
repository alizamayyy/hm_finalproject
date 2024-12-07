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
        .hide(axis='index'),
        use_container_width=True  # Makes the dataframe span the column's width
    )
    
def show_col_names():
    col_descriptions = {
        'CustomerID': 'Unique ID assigned to the customer',
        'Gender': 'Gender of the customer',
        'Age': 'Age of the customer',
        'Annual Income (k$)': 'Annual Income of the customer',
        'Spending Score (1-100)': 'Score assigned by the mall based on customer behavior and spending nature'
    }
    
    st.markdown("#### Column Descriptions")
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

import plotly.express as px
import streamlit as st

def plot_histograms(df):
    st.write("#### Distribution of Variables")
    
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
                title_x=0,  # Left-aligned title
                title_y=0.98,  # Position the title close to the top
                height=400,  # Fixed height for better alignment
                margin=dict(t=50, b=50, l=50, r=50)  # Optional: Adjust margins for better spacing
            )
            st.plotly_chart(fig, use_container_width=True)

def plot_boxplots(df):
    # st.write("#### Box Plots of Continuous Variables")
    
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
        title_x=0,  # Left-aligned title
        title_y=0.98,  # Position the title close to the top
        yaxis_title="Value",
        height=400,  # Optional: Set fixed height for better alignment
        margin=dict(t=50, b=50, l=50, r=50)  # Optional: Adjust margins for better spacing
    )
    st.plotly_chart(fig, use_container_width=True)
    
def plot_donut(df):
    # st.write("### Gender Distribution")
    
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
        title_x=0,  # Left-aligned title
        title_y=0.98,  # Position the title close to the top
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

import plotly.express as px
import streamlit as st

def scatter_age_vs_annual_income(df):
    # st.markdown("##### Age vs. Annual Income")
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
    
    # Update layout with fixed size
    fig.update_layout(
        title='Age vs. Annual Income',  # Set title
        title_x=0,  # Left-aligned title
        title_y=0.98,  # Position the title close to the top
        width=600,  # Fixed width
        height=400,  # Fixed height
        xaxis_title="Age",
        yaxis_title="Annual Income (k$)",
        legend_title=None,  # Remove legend title
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",  # Anchor legend at the bottom
            y=-0.4,  # Position legend below the plot
            xanchor="center",  # Center the legend horizontally
            x=0.5  # Center the legend horizontally
        )
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)


import plotly.express as px
import streamlit as st

def scatter_age_vs_spending_score(df):
    # st.write("### Age vs. Spending Score")
    
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
    
    # Update layout with fixed size
    fig.update_layout(
        title='Age vs. Spending Score',  # Set title
        title_x=0,  # Left-aligned title
        title_y=0.98,  # Position the title close to the top
        width=600,  # Fixed width
        height=400,  # Fixed height
        xaxis_title="Age",
        yaxis_title="Spending Score (1-100)",
        legend_title=None,  # Remove legend title
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",  # Anchor legend at the bottom
            y=-0.4,  # Position legend below the plot
            xanchor="center",  # Center the legend horizontally
            x=0.5  # Center the legend horizontally
        )
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)



import plotly.express as px
import streamlit as st

def scatter_annual_income_vs_spending_score(df):
    # st.write("### Annual Income vs. Spending Score")
     
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
    
    # Update layout with fixed size
    fig.update_layout(
        title='Annual Income vs. Spending Score',  # Set title
        title_x=0,  # Left-aligned title
        title_y=0.98,  # Position the title close to the top
        width=600,  # Fixed width
        height=400,  # Fixed height
        xaxis_title="Annual Income (k$)",
        yaxis_title="Spending Score (1-100)",
        legend_title=None,  # Remove legend title
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",  # Anchor legend at the bottom
            y=-0.4,  # Position legend below the plot
            xanchor="center",  # Center the legend horizontally
            x=0.5  # Center the legend horizontally
        )
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

def pcacomponents(df, show=True):
    # Assuming one_hot_encoding function is already defined elsewhere
    newdf = one_hot_encoding(df)  
    
    # Perform PCA with 4 components
    pca = PCA(n_components=4)
    principalComponents = pca.fit_transform(newdf)
    
    # Create a DataFrame for PCA components
    PCA_components = pd.DataFrame(principalComponents)

    # If show is True, display the PCA chart
    if show:
        features = range(pca.n_components_)
        plt.figure(figsize=(16,8))
        plt.bar(features, pca.explained_variance_ratio_, color='blue')  # Changed to solid blue
        plt.xlabel('PCA features')
        plt.ylabel('Variance (%)')
        plt.xticks(features)

        # Display the plot in Streamlit
        st.pyplot(plt)  # Display the plot
    
    # Return the PCA components DataFrame
    return PCA_components

def elbow_method(df):
    # Call pcacomponents to get PCA_components
    PCA_components = pcacomponents(df, show=False)

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

from yellowbrick.cluster import SilhouetteVisualizer

def show_silhouette(df, n_clusters=4):
    """
    Displays the silhouette graph for the PCA components.
    Calls pcacomponents function internally to get PCA components.
    """
    
    # Call the pcacomponents function to get PCA components
    PCA_components = pcacomponents(df, show=False)


    # Perform KMeans clustering
    model1 = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    
    # Create silhouette visualizer
    visualizer = SilhouetteVisualizer(model1, size=(1080, 500))
    
    # Fit and show silhouette plot for the first two PCA components
    visualizer.fit(PCA_components.iloc[:, :2])  # Use only the first two components for visualization
    visualizer.show()  # Finalize and render the figure

    # Display the visualizer plot in Streamlit
    st.pyplot(plt)  # Display the plot explicitly in Streamlit

def cluster_visualization(df):
    newdf = one_hot_encoding(df)
    PCA_components = pcacomponents(df, show=False)
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
    
    st.write("_Small introduction to our prokject._")
    st.write("")
    st.write("")
    
    st.markdown("##### Customer Segmentation")
    st.write("_On Customer Segmentation._")
    st.write("")
    
    st.markdown("##### K-Means Clustering")
    st.write("_On K-Means Clustering._")
    st.write("")
    
    st.markdown("##### PCA Components")
    st.write("_On PCA_")
    st.write("")
    
    st.markdown("##### Elbow Method")
    st.write("_On Elbow Method_")
    st.write("")
    
    st.markdown("##### Silhouette Plot")
    st.write("_On Silhouette Plot")
    st.write("")

elif page == "Data Exploration and Preparation":
    st.subheader("Data Exploration and Preparation")
    
    tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Data Visualization", "Relationships and Patterns"])
    
    with tab1:
        st.subheader("About the dataset")
        st.write("_Dataset Description_")
        col1, col2 = st.columns([1.5, 1], gap='medium') 
        
        with col1:
            show_csv(df)
        
        with col2:
            show_col_names()

        st.write("")
        show_cleaned_data(df)
    
    with tab2:
        plot_histograms(df)
        st.write("Most customers are between the ages of 25 and 35, with an average age of 39, suggesting a younger customer base. The average annual income ranges from 60,000 USD to 80,000 USD, indicating that many customers have lower incomes. Additionally, spending scores are mostly between 40 and 60, and the distribution appears nearly symmetric, suggesting that the spending scores are evenly distributed across the customer base.")
        st.write("")
        
        col1, col2 = st.columns(2, gap='large')
        with col1:
            plot_boxplots(df)
            st.write("The box plot reveals that spending scores are relatively evenly distributed, with a median around 50. Annual income, however, is skewed right, indicating a majority of lower-income individuals with a few high-income outliers. Age, while fairly evenly distributed, is slightly skewed right, indicating a slightly higher proportion of younger customers. The outlier in annual income suggests the presence of a high-net-worth individual, potentially influencing the overall distribution.")
              
        with col2:
            plot_donut(df)
            st.write("As we can see, 56% of customers are female, while 44% are male. The donut plot suggests a slight majority of female customers, reflecting a somewhat balanced yet slightly skewed gender distribution. ")

    with tab3:
        st.markdown("#### Relationship with Linear Regression Line of Best Fit")
        col1, col2, col3 = st.columns(3)
        with col1:
            scatter_age_vs_annual_income(df)
            st.write("The scatter plot shows the relationship between age and annual income, with separate trendlines for each gender. The trendlines appear to have a negative correlation, indicating that, as age increases, annual income tends to decrease within each gender group. This suggests an inverse relationship between age and annual income for both males and females.")
        with col2:      
            scatter_age_vs_spending_score(df)
            st.write("The scatter plot depicts the relationship between age and spending score, with separate trendlines for males and females. Both trendlines show a downward slope, indicating that as age increases, the spending score tends to decrease for both genders. This suggests a negative correlation between age and spending score for both males and females.")
        with col3:
            scatter_annual_income_vs_spending_score(df)
            st.write("The scatter plot illustrates the relationship between annual income and spending score, with separate trendlines for males and females. Both trendlines show almost no slope, indicating that there is little to no correlation between annual income and spending score for both genders. This suggests that changes in annual income have little impact on spending score for either males or females.")

        st.write("")
        
        st.markdown("#### Correlation between Features")
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            correlation_heatmap(df)
        st.write("The correlation heatmap reveals a moderate negative correlation between age and spending score, suggesting that younger customers tend to spend more. However, there is a weak positive correlation between annual income and spending score, indicating that income alone may not be a strong predictor of spending behavior. Age and annual income are almost unrelated. These insights can guide marketing strategies by targeting younger demographics and considering factors beyond income to influence spending habits.")
        
        

elif page == "Analysis and Insights":
    st.subheader("Analysis and Insights")
    
    tab1, tab2 = st.tabs(["Data Pre-processing", "K-Means Clustering"])
    
    with tab1:
        st.write("Since gender is a categorical variable, it needs to be encoded and converted into numeric. All other variables will be scaled to follow a normal distribution before being fed into the model. We will standardize these variables with a mean of 0 and a standard deviation of 1.")
        col1, col2 = st.columns(2, gap='medium')
        with col1:
            st.write("#### Standardizing variables")
            st.write("First, let's standardize all variables in the dataset to get them around the same scale.", )
            standardized_df = standardize_variables(df) 
            
           
            st.dataframe(standardized_df.style.background_gradient(cmap='plasma').set_properties(**{'font-family': 'Segoe UI'}), use_container_width=True)  # Display the styled DataFrame
        with col2:
            st.write("#### One-Hot Encoding")
            st.write("Second, (to be added)")
            # Call the one_hot_encoding function and store the result
            one_hot_encoded_df = one_hot_encoding(df)
                    
            # Display the one-hot encoded DataFrame without styling
            st.dataframe(one_hot_encoded_df.style.background_gradient(cmap='plasma').set_properties(**{'font-family': 'Segoe UI'}), use_container_width=True)
    
    with tab2:
        st.markdown("<p style='font-size: 15px;'>(K-Means clustering intro)</p>", unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap='large')
        with col1:
            st.markdown("#### Implementation of PCA")
            pcacomponents(df, show=True)
            st.write("PCA (Principal Component Analysis) was run to reduce the dimensionality of the dataset while preserving as much variance as possible. The chart displays the variance explained by each PCA component. It shows that the first two PCA components together account for more than 70% of the dataset's variance, indicating that these two components are sufficient for feeding into the model.")
        with col2:
            st.markdown("#### Elbow Method for the Clustering Model")
            elbow_method(df)
            st.write("The elbow method plot shows a sharp decrease in inertia from 1 to 4 clusters, then a more gradual decline. The elbow point, where the rate of decrease changes significantly, is around 4 clusters. This suggests that increasing the number of clusters beyond 4 may not significantly improve the clustering quality.")
            
        
        st.write("")
        st.write("")
        
        st.markdown("#### Silhouette Coefficient for Optimal k")
        col1, col2, col3 = st.columns([1, 4, 1], gap='large')

        with col2:
            show_silhouette(df, n_clusters=4)
            
        st.write("The silhouette plot provides a visual representation of the clustering quality. In this case, the overall silhouette score is moderate, suggesting that the clustering solution is reasonable but might not be optimal. Clusters 1 and 2 appear to be more cohesive, with data points that are well-matched to their own clusters. Clusters 0 and 3 have more variability in their data points, with some points being well-matched and others less so.")
        col1, col2, col3, col4 = st.columns(4, gap='medium')
        with col1:
            st.markdown("##### Cluster 0")
            st.write("Cluster 0 has a wide range of silhouette coefficients, indicating that some data points are well-clustered, while others might be misclassified or on the boundary. The average silhouette coefficient for this cluster is lower compared to other clusters, suggesting that it might not be as well-defined as the others.")
        
        with col2:
            st.markdown("##### Cluster 1")
            st.write("Cluster 1 has a narrower range of silhouette coefficients, suggesting that its data points are more consistently well-clustered. The average silhouette coefficient for this cluster is higher than cluster 0, indicating that, on average, the data points in this cluster are better matched to their own cluster.")
          
        with col3:
            st.markdown("##### Cluster 2")
            st.write("Cluster 2 has a similar pattern to cluster 1, with a narrow range of silhouette coefficients and a relatively high average silhouette coefficient. This suggests that the data points in this cluster are consistently well-matched to their own cluster and poorly matched to other clusters.")  
          
        with col4:
            st.markdown("##### Cluster 3")
            st.write("Cluster 3 has a wide range of silhouette coefficients, similar to cluster 0. However, the average silhouette coefficient for this cluster is higher, indicating that it might be a more cohesive cluster overall. Despite the outliers, the overall average is higher than cluster 0, suggesting that, on average, the data points in this cluster are better matched to their own cluster.")
        
        st.write("")
        st.markdown("#### Visualization of clusters built by the model")  
        cluster_visualization(df)

elif page == "Conclusion":
    st.subheader("Conclusion and Recommendations")
    # Add your conclusion content here