import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import base64

st.set_page_config(page_title="Customer Segregation", page_icon="👥", layout="wide", initial_sidebar_state="auto")

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
        st.markdown("**Handling Missing Values**")
        code = '''
        df.dropna(inplace=True)'''
        st.code(code, language="python")
        
        st.markdown(
            """
            <div style="text-align: justify;">
                The `df.dropna()` method eliminates any rows with missing values in the DataFrame, ensuring the data remains reliable. 
                This step is crucial for conducting accurate analyses, and the use of `inplace=True` ensures that the original DataFrame is updated.
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    with col2:
        st.markdown("**Handling Duplicate Data**")
        code = '''
        df.drop_duplicates(inplace=True) '''
        st.code(code, language="python")
        
        st.markdown(
            """
            <div style="text-align: justify;">
                The `df.drop_duplicates()` function removes any duplicate rows from the DataFrame, ensuring that each record is unique. 
                This process is essential for maintaining data integrity and avoiding skewed analysis results, and using `inplace=True` updates the DataFrame directly.
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    with col3:
        st.markdown("**Dropping 'CustomerID' Column**")
        code = '''
        df.drop(columns=['CustomerID'], inplace=True)'''
        st.code(code, language="python")
        
        st.markdown(
            """
            <div style="text-align: justify;">
                The `df.drop(columns=['CustomerID'], inplace=True)` method removes the 'CustomerID' column from the DataFrame. 
                This is done to simplify the dataset by eliminating unnecessary information that may not contribute to the analysis.
            </div>
            """,
            unsafe_allow_html=True,
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
    
    # Convert 'Gender' to 0 or 1 (0 for Male, 1 for Female)
    df['Gender_Female'] = df['Gender'].apply(lambda x: 1 if x == 'Female' else 0)
    
    # Drop the original 'Gender' column
    df = df.drop(columns=['Gender'])
    
    # Join the standardized features and the modified 'Gender_Female' column
    newdf = scaled_features.join(df['Gender_Female'])
    
    # Return the modified DataFrame
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
        custom_colors = ["blue", "green", "red", "purple"] 
        features = range(pca.n_components_)
        plt.figure(figsize=(16,8))
        for i, feature in enumerate(features):
            bar_color = custom_colors[i % len(custom_colors)]  # Cycle through cluster colors
            plt.bar(i, pca.explained_variance_ratio_[i], color=bar_color) # Changed to solid blue
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

def show_silhouette(df, n_clusters=4):

    # Assuming you have a PCA function called pcacomponents
    PCA_components = pcacomponents(df, show=False)

    # Perform KMeans clustering
    model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    clusters = model.fit_predict(PCA_components.iloc[:, :2])  # Use first two PCA components

    # Calculate silhouette score for the current clustering
    silhouette_avg = silhouette_score(PCA_components.iloc[:, :2], clusters)

    # Define custom colors for clusters
    custom_colors = ["blue", "green", "red", "purple"]  # Add more colors if needed for additional clusters
    # Plot the silhouette graph
    plt.figure(figsize=(10, 6))
    plt.title(f"Silhouette Plot for {n_clusters} Clusters (Average Score: {silhouette_avg:.2f})")

    # Plot silhouette for each sample
    sample_silhouette_values = silhouette_samples(PCA_components.iloc[:, :2], clusters)
    y_lower = 10
    for i in range(n_clusters):
        cluster_silhouette_values = sample_silhouette_values[clusters == i]
        cluster_silhouette_values.sort()
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = custom_colors[i % len(custom_colors)]  
        plt.fill_betweenx(range(y_lower, y_upper), 0, cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
        y_lower = y_upper + 10  # Leave some space between clusters

    # Draw the vertical line for the average silhouette score
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")

    # Labels and formatting
    plt.yticks([])
    plt.xlabel("Silhouette Coefficient Values")
    plt.ylabel("Cluster Label")
    plt.grid()

    # Show the plot in Streamlit
    st.pyplot(plt)

def cluster_visualization(df):
    newdf = one_hot_encoding(df)
    PCA_components = pcacomponents(df, show=False)
    model = KMeans(n_clusters=4, init='k-means++', random_state=42)
    clusters = model.fit_predict(PCA_components.iloc[:,:2])
    newdf["label"] = clusters

    # Create 3D scatter plot using Plotly
    trace0 = go.Scatter3d(
        x=newdf.Age[newdf["label"] == 0],
        y=newdf["Annual Income (k$)"][newdf["label"] == 0],
        z=newdf["Spending Score (1-100)"][newdf["label"] == 0],
        mode='markers',
        marker=dict(
            size=8,
            color='blue',
            opacity=0.8
        ),
        name='Cluster 0',
        showlegend=False
    )

    trace1 = go.Scatter3d(
        x=newdf.Age[newdf["label"] == 1],
        y=newdf["Annual Income (k$)"][newdf["label"] == 1],
        z=newdf["Spending Score (1-100)"][newdf["label"] == 1],
        mode='markers',
        marker=dict(
            size=8,
            color='red',
            opacity=0.8
        ),
        name='Cluster 1',
        showlegend=False
    )

    trace2 = go.Scatter3d(
        x=newdf.Age[newdf["label"] == 2],
        y=newdf["Annual Income (k$)"][newdf["label"] == 2],
        z=newdf["Spending Score (1-100)"][newdf["label"] == 2],
        mode='markers',
        marker=dict(
            size=8,
            color='green',
            opacity=0.8
        ),
        name='Cluster 2',
        showlegend=False
    )

    trace3 = go.Scatter3d(
        x=newdf.Age[newdf["label"] == 3],
        y=newdf["Annual Income (k$)"][newdf["label"] == 3],
        z=newdf["Spending Score (1-100)"][newdf["label"] == 3],
        mode='markers',
        marker=dict(
            size=8,
            color='purple',
            opacity=0.8
        ),
        name='Cluster 3',
        showlegend=False
    )

    # Combine all the traces for the plot
    data = [trace0, trace1, trace2, trace3]

    # Layout of the plot
    layout = go.Layout(
        scene=dict(
            xaxis_title='Age',
            yaxis_title='Annual Income (k$)',
            zaxis_title='Spending Score (1-100)',
        ),
        height=800,
    )

    # Create the figure
    fig = go.Figure(data=data, layout=layout)

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)  # Use st.pyplot instead of plt.show()

    
# def map_clusters(df):
#     PCA_components = pcacomponents(df, show=False)
#     model = KMeans(n_clusters=4, init='k-means++', random_state=42)
    
#     # Fit the model to the PCA components
#     model.fit(PCA_components.iloc[:, :2])  # Fit the model first
    
#     # Now you can predict
#     pred = model.predict(PCA_components.iloc[:, :2])
#     df['cluster'] = pred  # Add the cluster column to the dataframe

#     # Display the DataFrame in Streamlit without hiding the index
#     styled_frame = df.head().style.background_gradient(cmap='plasma').set_properties(**{'font-family': 'Segoe UI'})
#     st.dataframe(styled_frame, use_container_width=True)  # Use the styled DataFrame
    
#     # Return the updated dataframe
#     return df


# def compute_cluster_averages(df):
#     """
#     Compute average values for Age, Spending Score, and Annual Income by cluster.
#     """
#     # Ensure columns are numeric
#     df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
#     df['Spending Score (1-100)'] = pd.to_numeric(df['Spending Score (1-100)'], errors='coerce')
#     df['Annual Income (k$)'] = pd.to_numeric(df['Annual Income (k$)'], errors='coerce')

#     # Drop rows with missing or non-numeric data
#     df = df.dropna(subset=['Age', 'Spending Score (1-100)', 'Annual Income (k$)'])

#     # Group by cluster and compute averages
#     avg_df = df.groupby('cluster').mean()[['Age', 'Spending Score (1-100)', 'Annual Income (k$)']].reset_index()
#     return avg_df

def show_spending_score_vs_annual_income_vs_age(df):
    df = df.drop(['CustomerID'], axis=1)

    # Map back clusters to dataframe
    PCA_components = pcacomponents(df, show=False)
    model = KMeans(n_clusters=4, init='k-means++', random_state=42)
    model.fit(PCA_components.iloc[:, :2])  # Fit the model first

    pred = model.predict(PCA_components.iloc[:, :2])
    frame = pd.DataFrame(df)
    frame['cluster'] = pred

    # Ensure only numeric columns are used in the groupby mean
    numeric_columns = ['Age', 'Spending Score (1-100)', 'Annual Income (k$)']
    avg_df = frame.groupby('cluster')[numeric_columns].mean().reset_index()

    # Plotting the bar plots
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    sns.barplot(x='cluster', y='Age', data=avg_df, ax=ax[0])
    sns.barplot(x='cluster', y='Spending Score (1-100)', data=avg_df, ax=ax[1])
    sns.barplot(x='cluster', y='Annual Income (k$)', data=avg_df, ax=ax[2])
    plt.suptitle('Spending Score vs Annual Income vs Age', fontsize=20)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.8, wspace=0.4, hspace=0.4)

    st.pyplot(fig)

def cluster_analysis(df):
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.markdown("From the 3d scatter plot we can observe the following things:")
    # First Row
    col1, col2 = st.columns(2, gap='large')
    with col1:
        st.markdown(
        """
        <div style="background-color: #d9eaf7; color: #000; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; margin-bottom: 10px">
             Cluster 0 (<span style="color: blue;">blue</span>)
        </div>
        """,
        unsafe_allow_html=True,
    )
        st.markdown(
            """
            <div style="text-align: justify; margin-bottom: 20px">
                High average annual income, low average spending capacity. 
                Mean age is around 40 and gender is predominantly male.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
        """
        <div style="background-color: #d7f7da; color: #000; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; margin-bottom: 10px">
             Cluster 1 (<span style="color: green;">green</span>)
        </div>
        """,
        unsafe_allow_html=True,
    )
        st.markdown(
            """
            <div style="text-align: justify;">
                High average income, high spending score. 
                Mean age is around 30 and gender is predominantly female.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Second Row
    col3, col4 = st.columns(2, gap='large')
    with col3:
        st.markdown(
        """
        <div style="background-color: #f7d9d9; color: #000; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; margin-bottom: 10px">
             Cluster 2 (<span style="color: red;">red</span>)
        </div>
        """,
        unsafe_allow_html=True,
    )
        st.markdown(
            """
            <div style="text-align: justify;">
                Low average income, high spending score. 
                Mean age is around 25 and gender is predominantly female.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
        """
        <div style="background-color: #e8d9f7; color: #000; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; margin-bottom: 10px">
             Cluster 3 (<span style="color: purple;">purple</span>)
        </div>
        """,
        unsafe_allow_html=True,
    )
        st.markdown(
            """
            <div style="text-align: justify; margin-bottom: 30px">
                Low to mid average income, average spending capacity. 
                Mean age is around 50 and gender is predominantly female.
            </div>
            """,
            unsafe_allow_html=True,
        )


def plot_cluster_analysis(df):
    # Perform PCA for dimensionality reduction
    pca = PCA(n_components=2)
    PCA_components = pd.DataFrame(
        pca.fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]), 
        columns=['PCA1', 'PCA2']
    )

    # Perform clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster'] = kmeans.fit_predict(PCA_components)

    # Convert 'cluster' column to string type for proper color handling
    df['cluster'] = df['cluster'].astype(str)

    # Calculate averages by cluster
    avg_df = df.groupby('cluster', as_index=False).mean()

    # Swap values between Cluster 0 and Cluster 3
    cluster_0 = avg_df[avg_df['cluster'] == '0']
    cluster_3 = avg_df[avg_df['cluster'] == '3']

    # Swap rows for Cluster 0 and Cluster 3
    avg_df.loc[avg_df['cluster'] == '0', ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']] = cluster_3[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
    avg_df.loc[avg_df['cluster'] == '3', ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']] = cluster_0[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

    # Define the custom color mapping
    color_map = {
        '0': 'blue',     
        '1': 'green',   
        '2': 'red',      
        '3': 'purple'   
    }

    col1, col2, col3 = st.columns(3)

    # Average Age by Cluster
    with col1:
        fig_age = px.bar(
            avg_df,
            x="cluster",
            y="Age",
            title="Average Age by Cluster",
            labels={"cluster": "Cluster", "Age": "Average Age"},
            color="cluster",
            color_discrete_map=color_map, 
        )
        st.plotly_chart(fig_age, use_container_width=True)

    # Spending Score by Cluster
    with col2:
        fig_spending = px.bar(
            avg_df,
            x="cluster",
            y="Spending Score (1-100)",
            title="Spending Score by Cluster",
            labels={"cluster": "Cluster", "Spending Score (1-100)": "Spending Score"},
            color="cluster",
            color_discrete_map=color_map,  
        )
        st.plotly_chart(fig_spending, use_container_width=True)

    # Annual Income by Cluster
    with col3:
        fig_income = px.bar(
            avg_df,
            x="cluster",
            y="Annual Income (k$)",
            title="Annual Income by Cluster",
            labels={"cluster": "Cluster", "Annual Income (k$)": "Annual Income (k$)"},
            color="cluster",
            color_discrete_map=color_map, 
            
        )
        st.plotly_chart(fig_income, use_container_width=True)

        print(df.groupby('cluster').mean())
        # Manually adjust cluster labels based on the correct annual income range.
        cluster_map = {0: 3, 3: 0}  # Swap cluster 0 and cluster 3
        df['cluster'] = df['cluster'].map(cluster_map).fillna(df['cluster']).astype(int)
        print(df[['cluster', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].head())

    
# Navigation bar in sidebar
page = st.sidebar.selectbox("Select a section:", ["Introduction", "Data Exploration and Preparation", "Analysis and Insights", "Conclusion and Recommendations"])

#Application
st.header("Customer Segmentation with K-Means 👥")
st.markdown("<small>by Halimaw Magbeg</small>", unsafe_allow_html=True)


# Content based on navigation selection
if page == "Introduction":

    col1, col2 = st.columns([1, 1])  # Adjust column widths as needed

# Add the image to the left column
    with col1:
        st.image("images/5.png", width=500)
    with col2:
    # Why this dataset and why study it?
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("**In this project, we focus on customer segmentation using machine learning techniques.**")
        st.write("The primary goal is to understand customer behavior and identify distinct customer groups, which can help businesses make data-driven decisions.")
        st.write("We chose this dataset because it contains valuable features like age, annual income, and spending score, which are key indicators of consumer behavior. By analyzing these features, we can uncover insights that allow businesses to effectively target different customer segments.")
        st.write("This dataset provides an opportunity to apply clustering techniques, such as K-Means, to explore hidden patterns and better understand how customers differ in their spending and income behavior.")
        st.write("**This dataset is taken from [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python).**")
        st.write("")

    # Customer Segmentation explained
    st.markdown("### Customer Segmentation 📊")
    st.write("Customer segmentation is the process of dividing customers into distinct categories based on shared attributes. This practice helps organizations create more focused strategies, foster deeper customer relationships, and achieve better business outcomes.")
    st.write("By grouping customers based on common characteristics, businesses can design targeted marketing campaigns that resonate with each segment, improving engagement and customer satisfaction.")
    st.write("")

    # Customer Segmentation explained
    st.markdown("#### Why is Customer Segmentation important?")
    st.write(" Businesses need to understand their customers to offer personalized services, design targeted marketing campaigns, and enhance customer satisfaction. By analyzing patterns in customer data, we can unlock insights that lead to better decision-making and improved business outcomes.")
    st.write("Customer segmentation is an effective tool for businesses to closely align their strategy and tactics with, and better target, their current and future customers. Every customer is different and every customer journey is different, so a single approach often isn't going to work for all.")

    st.markdown(
    """
    <div style='text-align: center; margin: 20px 0;'>
        <div style='display: inline-block; height: 4px; width: 100%; background: linear-gradient(to right, #84ffc9, #aab2ff,#eca0ff);'></div>
    </div>
    """,
    unsafe_allow_html=True
)
    st.write("### The Team ✨")
    st.write("Meet the team members who made the exploration possible.")
    # Function to encode the image
    def img_to_base64(img_path):
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    # Encode the image
    bataluna = img_to_base64("images/bataluna.jpg")
    mier = img_to_base64("images/mier.png")
    alegam = img_to_base64("images/alegam.png")
    madaya = img_to_base64("images/madaya.jpg")
    cabo = img_to_base64("images/cabo.jpg")

    # Create a 3x2 grid of divs with rounded corners, drop shadows, and hover effects       
    grid_html = """
    <style>
        .grid-container {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 20px;
            margin-top: 10px;
        }
        .grid-item {
            background-color: #0e1117;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            display: flex;
            justify-content: center;
            align-items: center;
            width: 250px;  /* Set a fixed width */
            height: 250px;
            font-size: 24px;
            font-weight: bold;
            transition: background-color 0.3s ease;
            cursor: pointer;
            flex-direction: column;
            padding-top: 30px;
            
        }
        .grid-item:hover {
            background-color: #8b0026;
        }
        .grid-item img {
            width: 150px;  /* Set a fixed width */
            height: 150px; /* Set a fixed height */
            object-fit: cover; 
            padding-bottom: 10px;
            
            
            border-radius: 100px;  // {{ edit_1 }} Added border-radius for rounded corners
        }
    </style>
    <div class="grid-container">
    """

    # Add items to the grid (5 items)
    grid_items = [
        (bataluna, "Aliza May Bataluna"),
        (mier, "France Gieb Mier"),
        (alegam, "Cielo Alegam"),
        (madaya, "Angela Madaya"),
        (cabo, "Kerch Cabo"),
    ]

    for img, label in grid_items:
        grid_html += f'<div class="grid-item"><img src="data:image/png;base64,{img}" alt="{label}"><p>{label}</p></div>'

    grid_html += "</div>"

    st.markdown(grid_html, unsafe_allow_html=True)





elif page == "Data Exploration and Preparation":
    st.subheader("Data Exploration and Preparation")
    
    tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Data Visualization", "Relationships and Patterns"])
    
    with tab1:
        st.subheader("About the dataset")
        st.write("The dataset consists of customer information collected from a mall. This is what the dataset looks like:")
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
            st.markdown(
                """
                <div style="text-align: justify;">
                    The scatter plot shows the relationship between age and annual income, with separate trendlines for each gender. 
                    The trendlines appear to have a negative correlation, indicating that, as age increases, annual income tends to decrease within each gender group. 
                    This suggests an inverse relationship between age and annual income for both males and females.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:      
            scatter_age_vs_spending_score(df)
            st.markdown(
                """
                <div style="text-align: justify;">
                    The scatter plot depicts the relationship between age and spending score, with separate trendlines for males and females. 
                    Both trendlines show a downward slope, indicating that as age increases, the spending score tends to decrease for both genders. 
                    This suggests a negative correlation between age and spending score for both males and females.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            scatter_annual_income_vs_spending_score(df)
            st.markdown(
                """
                <div style="text-align: justify;">
                    The scatter plot illustrates the relationship between annual income and spending score, with separate trendlines for males and females. 
                    Both trendlines show almost no slope, indicating that there is little to no correlation between annual income and spending score for both genders. 
                    This suggests that changes in annual income have little impact on spending score for either males or females.
                </div>
                """,
                unsafe_allow_html=True,
            )


        st.write("")
        st.markdown(
            """
            <div style="text-align: center; font-size: 30px; font-weight: bold; margin-bottom: 10px">
                Correlation between Features
            </div>
            """,
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            correlation_heatmap(df)
        st.markdown(
        """
        <div style="text-align: justify;">
            The correlation heatmap reveals a moderate negative correlation between age and spending score, suggesting that younger customers tend to spend more. 
            However, there is a weak positive correlation between annual income and spending score, indicating that income alone may not be a strong predictor of spending behavior. 
            Age and annual income are almost unrelated. These insights can guide marketing strategies by targeting younger demographics and considering factors beyond income to influence spending habits.
        </div>
        """,
        unsafe_allow_html=True,
)

        

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
            st.write("Second, let's convert categorical variables into numerical ones using one-hot encoding.")
            # Call the one_hot_encoding function and store the result
            one_hot_encoded_df = one_hot_encoding(df)
                    
            # Display the one-hot encoded DataFrame without styling
            st.dataframe(one_hot_encoded_df.style.background_gradient(cmap='plasma').set_properties(**{'font-family': 'Segoe UI'}), use_container_width=True)
    
    with tab2:
        st.markdown("<p style='font-size: 15px;'>K-means clustering is an unsupervised learning technique to classify unlabeled data by grouping them by features, rather than pre-defined categories. The variable K represents the number of groups or categories created. The goal is to split the data into K different clusters and report the location of the center of mass for each cluster. Then, a new data point can be assigned a cluster (class) based on the closed center of mass.</p>", unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap='large')
        with col1:
            st.markdown("#### Implementation of PCA")
            pcacomponents(df, show=True)
            st.markdown(
                """
                <div style="text-align: justify;">
                    PCA (Principal Component Analysis) was run to reduce the dimensionality of the dataset while preserving as much variance as possible. 
                    The chart displays the variance explained by each PCA component. It shows that the first two PCA components together account for 
                    more than 70% of the dataset's variance, indicating that these two components are sufficient for feeding into the model.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown("#### Elbow Method for the Clustering Model")
            elbow_method(df)
            st.markdown(
                """
                <div style="text-align: justify;">
                    The elbow method plot shows a sharp decrease in inertia from 1 to 4 clusters, then a more gradual decline. 
                    The elbow point, where the rate of decrease changes significantly, is around 4 clusters. This suggests that increasing 
                    the number of clusters beyond 4 may not significantly improve the clustering quality.
                </div>
                """,
                unsafe_allow_html=True,
            )

        
        st.write("")
        st.write("")
        
        st.markdown("#### Silhouette Coefficient for Optimal k")
        col1, col2, col3 = st.columns([1, 4, 1], gap='large')

        with col2:
            show_silhouette(df, n_clusters=4)
            
        st.markdown(
            """
            <div style="text-align: justify; margin-bottom: 30px">
                The silhouette plot provides a visual representation of the clustering quality. In this case, the overall silhouette score 
                is moderate, suggesting that the clustering solution is reasonable but might not be optimal. Clusters 1 and 2 appear to be 
                more cohesive, with data points that are well-matched to their own clusters. Clusters 0 and 3 have more variability in their 
                data points, with some points being well-matched and others less so.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4 = st.columns(4, gap='medium')
        with col1:
            st.markdown(
        """
        <div style="background-color: #d9eaf7; color: #000; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; margin-bottom: 10px">
            Cluster 0 (<span style="color: blue;">blue</span>)
        </div>
        """,
        unsafe_allow_html=True,
    )
            st.markdown(
                """
                <div style="text-align: justify;">
                    Cluster 0 has a wide range of silhouette coefficients, indicating that some data points are well-clustered, while others might be misclassified or on the boundary. The average silhouette coefficient for this cluster is lower compared to other clusters, suggesting that it might not be as well-defined as the others.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
                <div style="background-color: #d7f7da; color: #000; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; margin-bottom: 10px">
                    Cluster 1 (<span style="color: green;">green</span>)
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="text-align: justify;">
                    Cluster 1 has a narrower range of silhouette coefficients, suggesting that its data points are more consistently well-clustered. The average silhouette coefficient for this cluster is higher than cluster 0, indicating that, on average, the data points in this cluster are better matched to their own cluster.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                """
                <div style="background-color: #f7d9d9; color: #000; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; margin-bottom: 10px">
                    Cluster 2 (<span style="color: red;">red</span>)
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="text-align: justify;">
                    Cluster 2 has a similar pattern to cluster 1, with a narrow range of silhouette coefficients and a relatively high average silhouette coefficient. This suggests that the data points in this cluster are consistently well-matched to their own cluster and poorly matched to other clusters.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                """
                <div style="background-color: #e8d9f7; color: #000; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; margin-bottom: 10px">
                    Cluster 3 (<span style="color: purple;">purple</span>)
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="text-align: justify;">
                    Cluster 3 has a wide range of silhouette coefficients, similar to cluster 0. However, the average silhouette coefficient for this cluster is higher, indicating that it might be a more cohesive cluster overall. Despite the outliers, the overall average is higher than cluster 0, suggesting that, on average, the data points in this cluster are better matched to their own cluster.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")
        st.markdown("#### Visualization of clusters built by the model")  
        col1, col2 = st.columns(2, gap='large')
        with col1:
            cluster_visualization(df)
        with col2:
            cluster_analysis(df)
            st.markdown(
            """
            <div style="text-align: justify;">
                The 3D scatter plot visualizes the clusters created by the K-Means model. 
                Each cluster is represented by a different color, with data points grouped 
                by age, annual income, and spending score. The clusters are well-separated, 
                indicating that the model has successfully segmented the customers based on these features.
            </div>
            """,
            unsafe_allow_html=True,
        )
        df = pd.read_csv('Mall_Customers.csv')
        df = df.drop(['CustomerID'], axis=1)
        plot_cluster_analysis(df)

elif page == "Conclusion and Recommendations":
    
    col1, col2 = st.columns([1, 1])  # Adjust column widths as needed

# Add the image to the left column
    with col1:
        st.image("images/7.png", width=500)
    with col2:
        st.subheader("Conclusion 🎯")
        st.write("The analysis of customer segmentation using K-Means clustering, along with Principal Component Analysis (PCA) and the Elbow Method, has provided valuable insights into the dataset. By clustering customers based on features like age, annual income, and spending score, we were able to identify distinct segments with different behaviors and characteristics. The silhouette plot helped evaluate the quality of the clustering, suggesting that the model's segmentation is fairly accurate but could potentially be further optimized.")
        st.write("The PCA revealed that the first two components explained over 70% of the variance in the dataset, indicating that a reduced dimensionality approach is effective in capturing key patterns. The Elbow Method suggested that four clusters were the optimal number for this dataset, striking a balance between simplicity and detailed segmentation.")
        st.write("The data analysis shows that younger customers tend to spend more, while income has a weaker impact on spending. These insights are essential for targeted marketing strategies.")
        

    st.write("Additionally, the clustering results reveal that customers with high annual income but low spending scores belong to a separate group, indicating that income alone may not fully explain customer spending behavior. These customers may not be engaging with products or services as much as expected, suggesting a potential opportunity to increase customer engagement through personalized offers or loyalty programs.")
    st.write("On the other hand, customers with lower income but higher spending scores represent another distinct group. These individuals are likely to value products or services based on factors other than price, such as brand loyalty or specific product features. Businesses could target this group with premium offerings or exclusive memberships.")
    st.write("The clustering also identified a group of customers with mid-range income and average spending scores, which could be the largest group. This segment may represent a balanced demographic that is open to a wide variety of products and services, making it a key focus for most marketing campaigns.")
    st.write("In summary, while income plays a role in spending behavior, other factors like age, preferences, and product engagement seem to have a more direct influence. These findings highlight the importance of using a multi-dimensional approach to customer segmentation for more effective targeting and campaign optimization.")
    

    col1, col2 = st.columns([1, 1])  # Adjust column widths as needed

# Add the image to the left column
    with col1:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.subheader("Recommendations 💡")
        st.write("1. **Target Younger Demographics**: Given the negative correlation between age and spending score, marketing efforts could focus more on younger customers, who are more likely to spend.")
        st.write("2. **Customized Marketing Campaigns**: Tailor campaigns to each cluster’s unique characteristics. For example, clusters with higher income may respond better to premium products, while younger clusters may be more interested in discounts or value-based products.")
        st.write("3. **Data-Driven Decision Making**: Continuously collect and analyze customer data to refine these segments. This can ensure that businesses stay aligned with evolving customer preferences and behaviors.")
        st.write("4. **Explore Other Features**: Further analysis can be conducted by incorporating additional features such as customer location or purchasing history, which may provide deeper insights into segmentation and consumer behavior.")
        st.write("5. **Model Improvement**: Fine-tuning the clustering model or exploring alternative clustering techniques (e.g., DBSCAN or hierarchical clustering) might yield even more accurate customer segments.")
    with col2:    
        st.image("images/6.png", width=700)

    # Overall summary and future recommendations
    st.write("Overall, the results of this analysis underscore the need for businesses to move beyond traditional demographic variables like income when segmenting their customers. By incorporating factors such as age, engagement, and specific spending patterns, companies can gain deeper insights into their customer base and design more effective marketing strategies.")
    st.write("In the future, further analysis could be conducted to explore other potential variables affecting customer behavior, such as customer location, purchase history, and brand loyalty. Machine learning techniques like deep learning could also be applied for more complex segmentation models, providing even finer granularity and better-targeted marketing efforts.")