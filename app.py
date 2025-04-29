import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


kmeans = joblib.load('kmeans_model.pkl')


st.markdown("""
    <style>
    .main {
        background-color: #F5F5F5;
    }
    h1 {
        color: #2F4F4F;
        text-align: center;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .stDownloadButton>button {
        background-color: #008CBA;
        color: white;
    }
    .css-1vq4p4l {
        padding: 2rem;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("# 📈 Customer Segmentation Dashboard")
st.markdown("**Utilisez le machine learning pour segmenter vos clients**")
st.markdown("**Vous pouvez utiliser ce dataset pour faire un test: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python**")

with st.container():
    st.markdown("### 📤 Importation des données")
    uploaded_file = st.file_uploader("", type=["csv"], help="Format requis : CSV avec colonnes [Gender, Age, Annual Income, Spending Score]")

if uploaded_file is not None:
    
    with st.spinner('Analyse des données en cours...'):
        df = pd.read_csv(uploaded_file)
        
        
        with st.expander("👀 Aperçu des données brutes", expanded=True):
            st.dataframe(
                df.head(),
                use_container_width=True,
                height=200
            )
        
        
        df = pd.get_dummies(df, columns=['Gender'], drop_first=True)
        X = df[['Gender_Male', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
        
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        
        df['Cluster'] = kmeans.predict(X_scaled)

    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🎨 Visualisation des clusters")
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        
        custom_palette = sns.color_palette("husl", 4)
        
        scatter = sns.scatterplot(
            data=df, 
            x='Annual Income (k$)', 
            y='Spending Score (1-100)', 
            hue='Cluster',
            palette=custom_palette,
            s=80,
            alpha=0.9
        )
        
        
        plt.title('Distribution des clients par cluster', fontsize=14)
        plt.xlabel('Revenu annuel (k$)', fontsize=12)
        plt.ylabel('Score de dépense (1-100)', fontsize=12)
        handles, labels = scatter.get_legend_handles_labels()
        scatter.legend(
            handles=handles,
            labels=['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'],
            title='Clusters',
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            frameon=True
        )
        st.pyplot(plt, use_container_width=True)

    with col2:
        st.markdown("### 📊 Statistiques clés")
        st.metric("Clients analysés", value=f"{len(df)}")
        st.metric("Nombre de clusters", value=5)
        st.metric("Revenu moyen", f"{df['Annual Income (k$)'].mean():.1f}k$")
        st.metric("Score de dépense moyen", f"{df['Spending Score (1-100)'].mean():.1f}")

    
    st.markdown("### 📋 Analyse par cluster")
    cluster_stats = df.groupby('Cluster').mean().reset_index()
    st.dataframe(
        cluster_stats.style.format({
            'Annual Income (k$)': '{:.1f}', 
            'Spending Score (1-100)': '{:.1f}',
            'Age': '{:.1f}'
        }),
        use_container_width=True
    )

    
    st.markdown("### 💾 Télécharger les résultats")
    result_csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Exporter les données analysées",
        data=result_csv,
        file_name='clustered_customers.csv',
        mime='text/csv',
        help="Cliquez pour télécharger les résultats au format CSV"
    )

else:
    st.info("ℹ️ Veuillez télécharger un fichier CSV pour commencer l'analyse")
    