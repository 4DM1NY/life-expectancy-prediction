#!/usr/bin/env python
# coding: utf-8

# In[43]:


####### Projet IA - Analyse Démographique et Prédiction de l’Espérance de Vie ######

# École : HESTIM  &  INSA Hauts-de-France
# Nom et Prénom : YLEWO Arrolle Içaire  
# Unité d'enseignement :  Intelligence Artificielle  
# Année académique : 2024-2025  
# Classe : Master 2 Cyberdéfense & Sécurité de l'Information (CDSI)


# In[44]:


# 1. Introduction & Objectifs

## Présentation du sujet;
# L'espérance de vie est un indicateur clé du développement humain, influencé par de nombreux facteurs socio-économiques, 
# sanitaires et environnementaux. À travers ce projet, nous analysons un ensemble de données démographiques provenant de 
# différents pays afin de mieux comprendre les variables qui influencent l'espérance de vie.

## Objectifs du projet:
# - Exploration des données : Réaliser une analyse exploratoire pour identifier les tendances et corrélations importantes.
# - Clustering : Regrouper les pays selon leurs caractéristiques démographiques et sanitaires afin d'identifier des profils types.
# - Régression : Construire un modèle prédictif capable d'estimer l'espérance de vie d'un pays à partir de ses indicateurs économiques et sanitaires.

## Justification de l'intérêt du sujet:
# La compréhension des facteurs déterminants de l'espérance de vie est cruciale pour orienter les politiques publiques 
# en matière de santé, d'éducation et de développement. Ce projet permet non seulement de mettre en pratique des techniques 
# d'analyse de données et d'intelligence artificielle, mais aussi de contribuer à une meilleure compréhension des enjeux 
# globaux liés à la qualité de vie des populations.


# In[45]:


# 2. Importation des Bibliothèques


# In[46]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


# In[47]:


# 3. & Chargement des Données & Description du Dataset

# source : https://www.kaggle.com/datasets/nelgiriyewithana/countries-of-the-world-2023/data
# Ce dataset mondial contient des informations variées sur chaque pays : 
# indicateurs démographiques, économiques, sanitaires, éducatifs, etc.


# In[48]:


#importation du dataset
df = pd.read_csv ("world-data-2023.csv")
print("######  chargement corect du Dataset")

# Aperçu des premières lignes du dataset
print("######  Aperçu du dataset :")
display(df.head())

# Dimensions du dataset
print(f"######  Dimensions : {df.shape[0]} lignes et {df.shape[1]} colonnes.\n")

# Types de variables
print("######  Types de données :")
print(df.dtypes)

# Valeurs manquantes (en % par colonne)
print("\n❗ Pourcentage de valeurs manquantes :")
missing_values = df.isnull().mean() * 100
print(missing_values[missing_values > 0].sort_values(ascending=False))

# Statistiques descriptives
print("\n######  Statistiques descriptives :")
display(df.describe())

# Visualisation : Histogrammes des principales variables
variables_a_plot = ['Life expectancy', 'GDP', 'Fertility Rate', 'Infant mortality', 'Physicians per thousand']
df[variables_a_plot].hist(bins=30, figsize=(15,10), color='skyblue', edgecolor='black')
plt.suptitle("Distribution des variables principales")
plt.tight_layout()
plt.show()

# Visualisation : Boxplots pour détecter les outliers
plt.figure(figsize=(15,8))
for i, var in enumerate(variables_a_plot, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x=df[var], color='lightcoral')
    plt.title(f'Boxplot de {var}')
plt.tight_layout()
plt.show()


# In[49]:


# 4. Prétraitement des Données
#  Suppression ou imputation des valeurs manquantes
#  Encodage des variables catégorielles (si besoin)
#  Normalisation / Standardisation
#  Sauvegarde du dataset nettoyé (optionnel)


# In[50]:


## Nettoyage des valeurs 

def clean_numeric_column(col):
    col = col.astype(str)

    # Corrige les notations scientifiques malformées comme "1.23-04" => "1.23e-04"
    col = col.str.replace(r'(?<=\d)-(?=\d{2,}$)', r'e-', regex=True)

    # Supprime tous les caractères sauf chiffres, points, "e", "-" (pour les exposants négatifs)
    col = col.str.replace(r'[^0-9eE\.-]', '', regex=True)

    # Remplace les chaînes vides par NaN
    col = col.replace('', np.nan)

    return pd.to_numeric(col, errors='coerce')


# Colonnes à nettoyer (extraites de ton exemple, à ajuster selon le dataset complet)
cols_to_clean = [
    'Agricultural Land( %)', 'Land Area(Km2)', 'Armed Forces size', 'Birth Rate', 'Co2-Emissions',
    'CPI', 'CPI Change (%)', 'Fertility Rate', 'Forested Area (%)', 'Gasoline Price', 'GDP',
    'Gross primary education enrollment (%)', 'Gross tertiary education enrollment (%)', 
    'Infant mortality', 'Life expectancy', 'Maternal mortality ratio', 'Minimum wage', 
    'Out of pocket health expenditure', 'Physicians per thousand', 'Population',
    'Population: Labor force participation (%)', 'Tax revenue (%)', 'Total tax rate',
    'Unemployment rate', 'Urban_population', 'Latitude', 'Longitude']

for col in cols_to_clean:
    if col in df.columns:
        df[col] = clean_numeric_column(df[col])


## Traitement des valeurs manquantes

# Imputation moyenne (pour les colonnes numériques)
num_cols = df.select_dtypes(include=np.number).columns.tolist()
imputer = SimpleImputer(strategy='mean')
df[num_cols] = imputer.fit_transform(df[num_cols])


## Normalisation (StandardScaler)
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Maintenant, df est prêt pour l’analyse ou le machine learning
print(df.head())


# Affichage des colonnes avec valeurs manquantes
missing_values = df.isnull().sum()
print("Valeurs manquantes :\n", missing_values[missing_values > 0])


# In[51]:


# 5. Analyse Exploratoire des Données (EDA)
# Heatmap de corrélation
# Scatterplots : Life Expectancy vs autres variables
# GroupBy et boxplots par catégories (continent, revenus, etc.)
# Analyse multivariée pour identifier les facteurs influents


# Heatmap : Visualise les corrélations entre les variables numériques.
# Scatterplots : Montre la relation entre l'espérance de vie et les variables les plus corrélées.
# Boxplots : Compare l'espérance de vie par catégories (continent, groupe de revenus, etc.).
# Pairplot : Explore les relations multivariées entre l'espérance de vie et les variables influentes.
# Régression linéaire multiple : Prédit l'espérance de vie en fonction des variables les plus corrélées.


# In[52]:


# Heatmap de corrélation

# Calcul de la matrice de corrélation
corr_matrix = df.corr(numeric_only=True)

# Heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, 
            cbar_kws={"shrink": .8}, linewidths=0.5)
plt.title('Heatmap de corrélation')
plt.tight_layout()
plt.show()


# Scatterplots : Life Expectancy vs autres variables clés
# Top 6 variables les plus corrélées à Life expectancy (positives et négatives)
target = 'Life expectancy'
correlations = corr_matrix[target].drop(target).sort_values(key=abs, ascending=False).head(6)

# Scatterplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, var in enumerate(correlations.index):
    sns.scatterplot(data=df, x=var, y=target, ax=axes[i])
    axes[i].set_title(f'{target} vs {var}')
plt.tight_layout()
plt.show()

# Scatter plot pour la densité de population (Density(P/Km2))
if 'Density\n(P/Km2)' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='Density\n(P/Km2)', y='Life expectancy')
    plt.title("Espérance de vie en fonction de la densité de population")
    plt.xlabel("Densité de population (P/Km²)")
    plt.ylabel("Espérance de vie")
    plt.tight_layout()
    plt.show()

# Scatter plot pour la proportion de terres agricoles
if 'Agricultural Land( %)' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='Agricultural Land( %)', y='Life expectancy')
    plt.title("Espérance de vie en fonction de la proportion de terres agricoles")
    plt.xlabel("Proportion de terres agricoles (%)")
    plt.ylabel("Espérance de vie")
    plt.tight_layout()
    plt.show()

# Scatter plot pour la taille des forces armées (Armed Forces size)
if 'Armed Forces size' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='Armed Forces size', y='Life expectancy')
    plt.title("Espérance de vie en fonction de la taille des forces armées")
    plt.xlabel("Taille des forces armées")
    plt.ylabel("Espérance de vie")
    plt.tight_layout()
    plt.show()

# Scatter plot pour les dépenses de santé à la charge des ménages (Out of pocket health expenditure)
if 'Out of pocket health expenditure' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='Out of pocket health expenditure', y='Life expectancy')
    plt.title("Espérance de vie en fonction des dépenses de santé à la charge des ménages")
    plt.xlabel("Dépenses de santé à la charge des ménages")
    plt.ylabel("Espérance de vie")
    plt.tight_layout()
    plt.show()


# # Analyse multivariée (Pairplot + Regression Lineaire Multiple)
# # airplot avec les variables les plus influentes :
# from seaborn import pairplot

# selected_vars = [target] + list(correlations.index)
# sns.pairplot(df[selected_vars], diag_kind='kde')
# plt.suptitle("Analyse multivariée des variables liées à Life expectancy", y=1.02)
# plt.show()


#Régression linéaire multiple (simple aperçu avant la modélisation complète) :

import statsmodels.api as sm

# Variables explicatives
X = df[correlations.index]
y = df[target]

# Ajout de l'intercept
X = sm.add_constant(X)

# Modèle OLS
model = sm.OLS(y, X).fit()

# Résumé
print(model.summary())



# In[53]:


# 6. Clustering (Non Supervisé)
# Standardisation des données
# Application du PCA (visualisation 2D)
# K-Means : choix du nombre optimal de clusters (méthode du coude)
# Hierarchical Clustering + Dendrogramme
# Interprétation des clusters obtenus


# In[54]:


# ========= Normalisation (StandardScaler) =========
scaler = StandardScaler()
df_std = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols, index=df.index)

# ========= PCA (visualisation 2D) =========
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_std)

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'], index=df.index)

plt.figure(figsize=(8,6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7)
plt.title('Projection PCA des pays (2 composantes principales)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)
plt.show()

# ========= K-Means – méthode du coude =========
inertia = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    km.fit(df_std)
    inertia.append(km.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K_range, inertia, 'o-')
plt.xlabel('Nombre de clusters (k)')
plt.ylabel("Inertie intra-cluster")
plt.title("Méthode du Coude")
plt.grid(True)
plt.show()

# ========= K-Means final =========
k_optimal = 4  # adapte en fonction du coude
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(df_std)

df['Cluster_KMeans'] = clusters
pca_df['Cluster_KMeans'] = clusters  # ajouter aussi dans pca_df pour l'affichage

# Visualisation PCA + Clustering
plt.figure(figsize=(8,6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster_KMeans', palette='Set2')
plt.title(f'K-Means Clustering (k={k_optimal}) - PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# ========= Clustering Hiérarchique + Dendrogramme =========
linkage_matrix = linkage(df_std, method='ward')

plt.figure(figsize=(12, 5))
dendrogram(linkage_matrix, truncate_mode='lastp', p=20, leaf_rotation=45, leaf_font_size=12, show_contracted=True)
plt.title("Dendrogramme – Clustering Hiérarchique")
plt.xlabel("Pays ou groupes")
plt.ylabel("Distance")
plt.grid(True)
plt.show()

n_clusters = 4
labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
df['Cluster_Hierarchical'] = labels

# ========= Interprétation des clusters =========
cluster_summary = df.groupby('Cluster_KMeans').mean(numeric_only=True)
print("Résumé par Cluster KMeans :\n", cluster_summary)

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='Cluster_KMeans', hue='Cluster_KMeans', y='Life expectancy', palette='Set2')
plt.title("Espérance de vie par Cluster KMeans")
plt.xlabel("Cluster")
plt.ylabel("Life Expectancy")
plt.grid(True)
plt.show()


# In[55]:


# 7. Modélisation Prédictive (Régression)
# Définir X (features) et y (target = Life Expectancy)
# Séparation train/test
# Modèles :
# Régression Linéaire
# Random Forest Regressor
# Évaluation des modèles :
# RMSE, MAE, R²
# Graphique : prédictions vs valeurs réelles


# In[56]:


# Sélectionner les variables explicatives (features) et la variable cible (target)
X = df[correlations.index]  # Les variables les plus corrélées à 'Life expectancy'
y = df[target]  # La variable cible : Life expectancy

# Séparation train/test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Régression Linéaire
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Prédictions
y_pred_lin = lin_reg.predict(X_test)

# 2. Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

# Prédictions
y_pred_rf = rf_reg.predict(X_test)

# 3. Évaluation des Modèles
# RMSE, MAE, R² pour les deux modèles

# Évaluation pour Régression Linéaire
rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
mae_lin = mean_absolute_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)

# Évaluation pour Random Forest
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Affichage des résultats
print("📊 Évaluation des Modèles:")
print("\nRégression Linéaire:")
print(f"RMSE: {rmse_lin:.2f}")
print(f"MAE: {mae_lin:.2f}")
print(f"R²: {r2_lin:.2f}")

print("\nRandom Forest Regressor:")
print(f"RMSE: {rmse_rf:.2f}")
print(f"MAE: {mae_rf:.2f}")
print(f"R²: {r2_rf:.2f}")

# 4. Graphique : Prédictions vs Valeurs Réelles

plt.figure(figsize=(12, 6))

# Graphique pour la régression linéaire
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lin, alpha=0.6, color='skyblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k')  # Redundant 'color' removed
plt.title('Régression Linéaire: Prédictions vs Valeurs Réelles')
plt.xlabel('Valeurs Réelles')
plt.ylabel('Prédictions')

# Graphique pour Random Forest
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_rf, alpha=0.6, color='lightgreen')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k')  # Redundant 'color' removed
plt.title('Random Forest: Prédictions vs Valeurs Réelles')
plt.xlabel('Valeurs Réelles')
plt.ylabel('Prédictions')

plt.tight_layout()
plt.show()


# In[57]:


# Conclusion :

# Ce projet a permis de prédire l'espérance de vie par pays en utilisant des données démographiques,
# économiques et de santé. Nous avons appliqué des techniques de prétraitement, telles que le nettoyage 
# des données et la normalisation, et utilisé des modèles de régression pour effectuer les prédictions. 
# Les résultats montrent que notre modèle offre une précision raisonnable, mais des améliorations peuvent 
# être apportées en utilisant des modèles plus complexes. Ce travail démontre l'application des techniques 
# d'intelligence artificielle à un problème concret et met en évidence l'importance d'une bonne préparation 
# des données pour obtenir des résultats fiables.


# In[ ]:




