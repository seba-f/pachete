import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import geopandas as gpd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Required packages installation:
# pip install scikit-learn  # instead of sklearn
# pip install statsmodels
# pip install geopandas
# pip install streamlit
# pip install pandas numpy matplotlib seaborn

# Define constants for column names
NUMERIC_COLUMNS = ['normalized-losses', 'bore', 'stroke', 'horsepower', 'peak-rpm', 'price']
CORRELATION_COLUMNS = ['wheel-base', 'length', 'width', 'height', 'curb-weight',
                      'engine-size', 'compression-ratio', 'horsepower', 'city-mpg', 'highway-mpg', 'price']

# Load CSS styles
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap');
    
    .title {
        font-family: 'Montserrat', sans-serif;
        color: #FFD700 !important;
        font-size: 50px;
        text-align: center;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 3px 3px 10px rgba(0, 0, 0, 0.3);
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #ADD8E6;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ADD8E6, #FFD700) !important;
        border-radius: 0 10px 10px 0;
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stSidebar"] label {
        font-size: 200px;
        font-weight: bold;
    }
    
    .stRadio > div {
        padding: 10px;
        background-color: rgba(255, 215, 0, 0.1);
        border-radius: 8px;
    }
    
    h1, h2, h3 {
        color: #1E3A8A !important;
        font-family: 'Montserrat', sans-serif;
        margin-top: 20px;
        margin-bottom: 15px;
        padding-bottom: 5px;
        border-bottom: 2px solid #FFD700;
    }
    
    .stDataFrame {
        border: 2px solid #FFD700;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    p, li {
        font-family: 'Roboto', sans-serif;
        font-size: 16px;
        line-height: 1.6;
        text-shadow: 3px 3px 10px rgba(0, 0, 0, 0.3);
    }
    
    a {
        color: #1E3A8A !important;
        text-decoration: none;
        border-bottom: 1px dotted #1E3A8A;
        transition: all 0.3s ease;
    }
    
    a:hover {
        color: #FFD700 !important;
        border-bottom: 1px solid #FFD700;
    }
    
    code {
        color: white !important;  
        background-color: #FF5733 !important;  
        padding: 3px 6px;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .car-emoji {
        display: inline-block;
        transition: transform 0.5s ease;
    }
    
    .car-emoji:hover {
        cursor:default;
        transform: translateX(20px) rotate(5deg);
        animation: carMove 1s infinite alternate;
    }
    
    @keyframes carMove {
        0% {
            transform: translateX(0) rotate(0deg);
        }
        50% {
            transform: translateX(15px) rotate(5deg);
        }
        100% {
            transform: translateX(0) rotate(0deg);
        }
    }
    
    .highlight-box {
        background-color: rgba(255, 215, 0, 0.15);
        border-left: 4px solid #FFD700;
        padding: 15px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
    }
    
    [data-testid="stHeader"] {
        background-color: rgba(173, 216, 230, 0.8);
        backdrop-filter: blur(10px);
    }
    
    .st-emotion-cache-kgpedg {height:1%}
    .st-emotion-cache-gi0tri {display:none}
    h1:hover{cursor:default}
    </style>
    """, unsafe_allow_html=True)

# Data loading and preprocessing functions
def load_data():
    return pd.read_csv(filepath_or_buffer="Automobile_data.csv")

def handle_missing_values(df):
    df = df.replace('?', np.nan)
    numeric_cols = ['normalized-losses', 'bore', 'stroke', 'horsepower', 'peak-rpm', 'price']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].mean(), inplace=True)
    
    categorical_cols = [col for col in df.columns if col not in numeric_cols]
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def encode_categorical_data(df):
    # Convert num-of-doors
    door_mapping = {'two': 2, 'four': 4}
    df['num-of-doors'] = df['num-of-doors'].map(door_mapping)
    
    # Convert num-of-cylinders
    cylinder_mapping = {
        'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'eight': 8, 'twelve': 12
    }
    df['num-of-cylinders'] = df['num-of-cylinders'].map(cylinder_mapping)
    return df

def scale_data(df):
    scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    
    scaled_data = {
        'standardized': pd.DataFrame(
            scaler.fit_transform(df[NUMERIC_COLUMNS]),
            columns=NUMERIC_COLUMNS
        ),
        'normalized': pd.DataFrame(
            minmax_scaler.fit_transform(df[NUMERIC_COLUMNS]),
            columns=NUMERIC_COLUMNS
        )
    }
    return scaled_data

# Visualization functions
def plot_boxplot(data, x, y, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, x=x, y=y, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    st.image(buf)

# Main application
def main():
    load_css()
    st.markdown('<h1 class="title">Analiză și modelare a datelor despre automobile <span class="car-emoji">🏎️</span></h1>',
                unsafe_allow_html=True)

    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = load_data()
        st.session_state.data_processing_steps = []
        st.session_state.scaled_data = None

    # Sidebar navigation
    with st.sidebar:
        st.title("Pachete software - Proiect")
        section = st.radio('', ["Date", "Descriere date", "Prelucrare date", "Analiza Exploratorie", "Modelare"])

    # Main content sections
    if section == "Date":
        show_raw_data()
    elif section == "Descriere date":
        show_data_description()
    elif section == "Prelucrare date":
        show_data_processing()
    elif section == "Analiza Exploratorie":
        show_exploratory_analysis()
    elif section == "Modelare":
        show_modeling()

# Section functions
def show_raw_data():
    st.header("Date inițiale")
    st.markdown("""
        <div class="highlight-box">
        <h3>Sursa datelor</h3>
        Acest set de date a fost preluat de pe platforma Kaggle. Poate fi accesat la următorul link:
        <a href="https://www.kaggle.com/datasets/toramky/automobile-dataset/data" target="_blank">Automobile Dataset - Kaggle</a>
        </div>
        """, unsafe_allow_html=True)
    st.dataframe(st.session_state.processed_data)

def show_data_description():
    st.header("Descriere date")
    st.markdown("""
            <h3>Descrierea setului de date "Automobile Data"</h3>

            <p>Setul de date conține <strong>205 înregistrări</strong> despre automobile, cu <strong>26 de coloane</strong>, fiecare reprezentând o caracteristică a unei mașini.</p>

            <h4>Structura datelor:</h4>
            <div class="highlight-box">
            <strong>Caracteristici generale:</strong>
            <ul>
              <li><code>symboling</code> – Ratingul de risc al mașinii (de la -2 la 3)</li>
              <li><code>make</code> – Marca mașinii (ex: Audi, BMW, Toyota)</li>
              <li><code>fuel-type</code> – Tipul de combustibil (ex: benzină, diesel)</li>
              <li><code>aspiration</code> – Tipul de admisie (standard sau turbo)</li>
              <li><code>num-of-doors</code> – Numărul de uși (ex: două, patru)</li>
              <li><code>body-style</code> – Tipul caroseriei (ex: sedan, hatchback, convertible)</li>
              <li><code>drive-wheels</code> – Tipul de tracțiune (FWD - tracțiune față, RWD - tracțiune spate, 4WD - tracțiune integrală)</li>
              <li><code>engine-location</code> – Poziția motorului (ex: față, spate)</li>
              <li><code>country-of-origin</code> – Țara de origine a mașinii (ex: Germany, Japan, USA, Sweden, UK, France, Italy)</li>
            </ul>
            </div>

            <div class="highlight-box">
            <strong>Dimensiuni și greutate:</strong>
            <ul>
              <li><code>wheel-base</code> – Distanța dintre punți (în inch)</li>
              <li><code>length</code> – Lungimea mașinii (în inch)</li>
              <li><code>width</code> – Lățimea mașinii (în inch)</li>
              <li><code>height</code> – Înălțimea mașinii (în inch)</li>
              <li><code>curb-weight</code> – Greutatea proprie a mașinii (în kg)</li>
            </ul>
            </div>

            <div class="highlight-box">
            <strong>Motor și performanță:</strong>
            <ul>
              <li><code>engine-type</code> – Tipul motorului (ex: DOHC, OHV, OHC)</li>
              <li><code>num-of-cylinders</code> – Numărul de cilindri (ex: patru, șase, opt)</li>
              <li><code>engine-size</code> – Dimensiunea motorului (în cm³)</li>
              <li><code>fuel-system</code> – Tipul de alimentare (ex: MPFI - injecție multipunct, carburetor)</li>
              <li><code>compression-ratio</code> – Raportul de compresie al motorului</li>
              <li><code>horsepower</code> – Puterea motorului (în cai putere)</li>
              <li><code>peak-rpm</code> – Numărul maxim de rotații pe minut</li>
            </ul>
            </div>

            <div class="highlight-box">
            <strong>Eficiență și consum:</strong>
            <ul>
              <li><code>city-mpg</code> – Consum de combustibil în oraș (mile per gallon)</li>
              <li><code>highway-mpg</code> – Consum de combustibil pe autostradă (mile per gallon)</li>
            </ul>
            </div>

            <div class="highlight-box">
            <strong>Preț și costuri adiționale:</strong>
            <ul>
              <li><code>price</code> – Prețul mașinii (în dolari)</li>
              <li><code>normalized-losses</code> – Pierderi normalizate din asigurări (indicator al costurilor de reparație)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

def show_data_processing():
    st.header("Prelucrare date")
    
    with st.sidebar:
        subsection = st.radio(
            "Selectează subsecțiunea",
            ["Tratare valori lipsă", "Codificarea datelor", "Metode de scalare", "Statistici descriptive"]
        )
    
    if subsection == "Tratare valori lipsă":
        if "Tratare valori lipsă" not in st.session_state.data_processing_steps:
            st.session_state.processed_data = handle_missing_values(st.session_state.processed_data)
            st.session_state.data_processing_steps.append("Tratare valori lipsă")
        show_missing_values_processing()
    
    elif subsection == "Codificarea datelor":
        if "Codificarea datelor" not in st.session_state.data_processing_steps:
            st.session_state.processed_data = encode_categorical_data(st.session_state.processed_data)
            st.session_state.data_processing_steps.append("Codificarea datelor")
        show_encoding_results()
    
    elif subsection == "Metode de scalare":
        if "Metode de scalare" not in st.session_state.data_processing_steps:
            st.session_state.scaled_data = scale_data(st.session_state.processed_data)
            st.session_state.data_processing_steps.append("Metode de scalare")
        show_scaling_results()
    
    elif subsection == "Statistici descriptive":
        show_descriptive_statistics()

def show_exploratory_analysis():
    st.header("Analiza Exploratorie")
    
    with st.sidebar:
        subsection = st.radio(
            "Selectează subsecțiunea",
            ["Analiza prețurilor", "Analiza puterii motorului", "Analiza consumului", 
             "Analiza caroseriei", "Corelații și boxplot-uri", "Distribuția geografică"]
        )
    
    if subsection == "Analiza prețurilor":
        show_price_analysis()
    elif subsection == "Analiza puterii motorului":
        show_power_analysis()
    elif subsection == "Analiza consumului":
        show_consumption_analysis()
    elif subsection == "Analiza caroseriei":
        show_body_style_analysis()
    elif subsection == "Corelații și boxplot-uri":
        show_correlations_and_boxplots()
    elif subsection == "Distribuția geografică":
        show_geographic_distribution()

def show_modeling():
    st.header("Modelare")
    
    with st.sidebar:
        subsection = st.radio(
            "Selectează subsecțiunea",
            ["Clusterizare", "Regresie Logistică", "Regresie Multiplă"]
        )
    
    if subsection == "Clusterizare":
        show_clustering()
    elif subsection == "Regresie Logistică":
        show_logistic_regression()
    elif subsection == "Regresie Multiplă":
        show_multiple_regression()

# Processing section functions
def show_missing_values_processing():
    st.markdown("""
    <div class="highlight-box">
    <h3>Tratarea valorilor lipsă</h3>
    <p>În setul de date original există valori lipsă marcate cu simbolul <code>?</code>. Pentru a realiza o analiză statistică corectă, 
    am tratat aceste valori lipsă folosind următoarele metode:</p>
    
    <ol>
        <li><strong>Identificarea valorilor lipsă</strong> - Am identificat toate valorile marcate cu <code>?</code> și le-am înlocuit cu <code>NaN</code> (Not a Number) pentru a fi recunoscute de pandas ca valori lipsă.</li>
        <li><strong>Înlocuirea valorilor numerice lipsă</strong> - Pentru coloanele numerice, am înlocuit valorile lipsă cu media valorilor existente din coloana respectivă.</li>
        <li><strong>Înlocuirea valorilor categorice lipsă</strong> - Pentru coloanele categorice, am înlocuit valorile lipsă cu cea mai frecventă valoare (modul) din coloana respectivă.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3>Tabel fără valori lipsă</h3>", unsafe_allow_html=True)
    st.dataframe(st.session_state.processed_data)

def show_encoding_results():
    st.markdown("<h3>Codificarea datelor</h3>", unsafe_allow_html=True)
    st.markdown("""
        <div class="highlight-box">
        <p>Pentru a putea efectua o analiză corectă, unele coloane categorice trebuie convertite în format numeric:</p>
        <ul>
            <li><strong>num-of-doors</strong> - Convertim direct în număr (two → 2, four → 4)</li>
            <li><strong>num-of-cylinders</strong> - Convertim textul în numărul corespunzător de cilindri</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h4>Date codificate:</h4>", unsafe_allow_html=True)
    date_codificate = pd.DataFrame({
        'num-of-doors': st.session_state.processed_data['num-of-doors'],
        'num-of-cylinders': st.session_state.processed_data['num-of-cylinders']
    })
    st.dataframe(date_codificate)

def show_scaling_results():
    st.markdown("<h3>Metode de scalare</h3>", unsafe_allow_html=True)
    st.markdown("""
        <div class="highlight-box">
        <p>Pentru a putea compara mai ușor variabilele numerice între ele, putem aplica diferite metode de scalare:</p>
        <ul>
            <li><strong>Standardizare (Z-score)</strong> - Transformă datele pentru a avea medie 0 și deviație standard 1</li>
            <li><strong>Normalizare Min-Max</strong> - Scalează datele în intervalul [0,1]</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h4>Date originale:</h4>", unsafe_allow_html=True)
    st.dataframe(st.session_state.processed_data[NUMERIC_COLUMNS].head())
    
    st.markdown("<h4>Date standardizate (Z-score):</h4>", unsafe_allow_html=True)
    st.dataframe(st.session_state.scaled_data['standardized'].head())
    
    st.markdown("<h4>Date normalizate (Min-Max):</h4>", unsafe_allow_html=True)
    st.dataframe(st.session_state.scaled_data['normalized'].head())

def show_descriptive_statistics():
    st.markdown("<h3>Statistici descriptive</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight-box">
    <p>Mai jos sunt prezentate statisticile descriptive pentru coloanele numerice din setul de date. 
    Acestea includ: numărul de valori, media, deviația standard, valoarea minimă, cuartilele și valoarea maximă.</p>
    </div>
    """, unsafe_allow_html=True)
    
    statistici = st.session_state.processed_data.describe()
    st.dataframe(statistici)
    
# Analysis section functions
def show_price_analysis():
    st.markdown("<h3>Analiza prețurilor</h3>", unsafe_allow_html=True)
    tabel_procesat = st.session_state.processed_data.copy()
    
    fig_pret = pd.DataFrame({
        'Preț (USD)': tabel_procesat['price']
    })
    st.bar_chart(fig_pret)
    
    # price distribution for each make
    marci_frecvente = tabel_procesat['make'].value_counts()
    marci_frecvente = marci_frecvente[marci_frecvente >= 10].index.tolist()
    tabel_marci_frecvente = tabel_procesat[tabel_procesat['make'].isin(marci_frecvente)]
    
    pret_marca_stats = tabel_marci_frecvente.groupby('make')['price'].agg(['mean', 'median', 'min', 'max']).reset_index()
    pret_marca_stats.columns = ['Marca', 'Preț mediu', 'Preț median', 'Preț minim', 'Preț maxim']
    
    st.write("Acest tabel arată statisticile prețurilor pentru mărcile cele mai frecvente din setul de date.")
    st.dataframe(pret_marca_stats)
    
    fig_pret_marca_bar = pd.DataFrame({
        'Marca': pret_marca_stats['Marca'],
        'Preț mediu': pret_marca_stats['Preț mediu']
    }).set_index('Marca')
    
    st.bar_chart(fig_pret_marca_bar)

def show_power_analysis():
    st.markdown("<h3>Analiza puterii motorului</h3>", unsafe_allow_html=True)
    tabel_procesat = st.session_state.processed_data.copy()
    
    # power-price relationship
    fig_putere_pret = pd.DataFrame({
        'Putere (CP)': tabel_procesat['horsepower'],
        'Preț (USD)': tabel_procesat['price']
    })
    st.scatter_chart(fig_putere_pret, x='Putere (CP)', y='Preț (USD)')

    # stats for each fuel type
    putere_combustibil_stats = tabel_procesat.groupby('fuel-type')['horsepower'].agg(['mean', 'median', 'min', 'max']).reset_index()
    putere_combustibil_stats.columns = ['Tip combustibil', 'Medie', 'Mediană', 'Minim', 'Maxim']

    st.write("Acest tabel arată statisticile puterii motorului pentru fiecare tip de combustibil.")
    st.dataframe(putere_combustibil_stats)

    # histogram
    for tip in tabel_procesat['fuel-type'].unique():
        st.write(f"Distribuția puterii pentru combustibil: {tip}")
        subset = tabel_procesat[tabel_procesat['fuel-type'] == tip]
        fig_hist = pd.DataFrame({
            'Putere (CP)': subset['horsepower']
        })
        st.bar_chart(fig_hist)

def show_consumption_analysis():
    st.markdown("<h3>Analiza consumului</h3>", unsafe_allow_html=True)
    tabel_procesat = st.session_state.processed_data.copy()

    # engine size vs urban consumption
    fig_motor_consum = pd.DataFrame({
        'Dimensiune motor (cm³)': tabel_procesat['engine-size'],
        'Consum oraș (mpg)': tabel_procesat['city-mpg']
    })

    st.write("Acest grafic arată relația inversă între dimensiunea motorului și eficiența consumului în oraș.")
    st.scatter_chart(fig_motor_consum, x='Dimensiune motor (cm³)', y='Consum oraș (mpg)')

    # urban vs highway consumption
    fig_consum = pd.DataFrame({
        'Consum oraș (mpg)': tabel_procesat['city-mpg'],
        'Consum autostradă (mpg)': tabel_procesat['highway-mpg']
    })

    st.write("Acest grafic arată corelația dintre consumul în oraș și consumul pe autostradă.")
    st.scatter_chart(fig_consum, x='Consum oraș (mpg)', y='Consum autostradă (mpg)')

def show_body_style_analysis():
    st.markdown("<h3>Analiza caroseriei</h3>", unsafe_allow_html=True)
    tabel_procesat = st.session_state.processed_data.copy()

    # chassis distribution
    fig_caroserie = pd.DataFrame(tabel_procesat['body-style'].value_counts()).reset_index()
    fig_caroserie.columns = ['Tip caroserie', 'Număr']

    st.write("Acest grafic arată distribuția tipurilor de caroserie în setul de date.")
    st.bar_chart(fig_caroserie, x='Tip caroserie', y='Număr')

    # weight distribution by chassis
    greutate_caroserie_stats = tabel_procesat.groupby('body-style')['curb-weight'].agg(['mean', 'median', 'min', 'max']).reset_index()
    greutate_caroserie_stats.columns = ['Tip caroserie', 'Greutate medie', 'Greutate mediană', 'Greutate minimă', 'Greutate maximă']

    st.write("Acest tabel arată statisticile greutății pentru fiecare tip de caroserie.")
    st.dataframe(greutate_caroserie_stats)

    fig_greutate_caroserie_bar = pd.DataFrame({
        'Tip caroserie': greutate_caroserie_stats['Tip caroserie'],
        'Greutate medie': greutate_caroserie_stats['Greutate medie']
    }).set_index('Tip caroserie')

    st.bar_chart(fig_greutate_caroserie_bar)

def show_correlations_and_boxplots():
    st.markdown("<h3>Corelații și boxplot-uri</h3>", unsafe_allow_html=True)
    tabel_procesat = st.session_state.processed_data.copy()
    
    # correlation matrix
    matrice_corelatie = tabel_procesat[CORRELATION_COLUMNS].corr()

    st.write("Această matrice de corelație arată relațiile liniare între variabilele numerice din setul de date.")
    st.dataframe(matrice_corelatie.style.background_gradient(cmap='coolwarm'))

    st.markdown("<h4>Boxplot-uri pentru analiza distribuțiilor</h4>", unsafe_allow_html=True)

    # power by fuel type
    st.markdown("<h4>Boxplot: Puterea motorului în funcție de tipul de combustibil</h4>", unsafe_allow_html=True)
    plot_boxplot(
        tabel_procesat, 
        'fuel-type', 
        'horsepower', 
        'Distribuția puterii motorului pentru fiecare tip de combustibil',
        'Tip combustibil',
        'Putere (CP)'
    )

    # price by make
    marci_frecvente = tabel_procesat['make'].value_counts()
    marci_frecvente = marci_frecvente[marci_frecvente >= 10].index.tolist()
    tabel_marci_frecvente = tabel_procesat[tabel_procesat['make'].isin(marci_frecvente)]
    
    st.markdown("<h4>Boxplot: Prețurile în funcție de marca mașinii</h4>", unsafe_allow_html=True)
    plot_boxplot(
        tabel_marci_frecvente, 
        'make', 
        'price', 
        'Distribuția prețurilor pentru mărcile cele mai frecvente',
        'Marca',
        'Preț (USD)'
    )

    # weight by chassis
    st.markdown("<h4>Boxplot: Greutatea în funcție de tipul de caroserie</h4>", unsafe_allow_html=True)
    plot_boxplot(
        tabel_procesat, 
        'body-style', 
        'curb-weight', 
        'Distribuția greutății pentru fiecare tip de caroserie',
        'Tip caroserie',
        'Greutate (kg)'
    )

def show_geographic_distribution():
    st.markdown("<h3>Distribuția geografică a automobilelor</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight-box">
    <p>Această secțiune prezintă distribuția geografică a automobilelor din setul de date, 
    oferind o perspectivă vizuală asupra originii mașinilor și a caracteristicilor lor.</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Download and load world map data
        import urllib.request
        import os
        
        # Create a data directory if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')
            
        # Download the natural earth data if it doesn't exist
        if not os.path.exists('data/ne_110m_admin_0_countries.shp'):
            # Using a different URL for the Natural Earth data
            url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
            
            try:
                urllib.request.urlretrieve(url, "data/ne_110m_admin_0_countries.geojson")
                # Convert GeoJSON to Shapefile using geopandas
                world = gpd.read_file("data/ne_110m_admin_0_countries.geojson")
                world.to_file("data/ne_110m_admin_0_countries.shp")
                # Clean up the GeoJSON file
                os.remove("data/ne_110m_admin_0_countries.geojson")
            except Exception as e:
                st.error(f"Eroare la descărcarea datelor: {str(e)}")
                st.info("Vă rugăm să verificați conexiunea la internet și să încercați din nou.")
                return
        
        # Load the shapefile
        world = gpd.read_file("data/ne_110m_admin_0_countries.shp")
        
        # Prepare data for visualization
        df = st.session_state.processed_data.copy()
        
        # Count cars by country
        country_counts = df['country-of-origin'].value_counts().reset_index()
        country_counts.columns = ['ADMIN', 'count']  # Changed from 'name' to 'ADMIN' to match the shapefile
        
        # Map country names to match natural earth data
        country_mapping = {
            'Germany': 'Germany',
            'Japan': 'Japan',
            'USA': 'United States of America',  # Updated to match Natural Earth data
            'Sweden': 'Sweden',
            'UK': 'United Kingdom of Great Britain and Northern Ireland',  # Updated to match Natural Earth data
            'France': 'France',
            'Italy': 'Italy'
        }
        
        country_counts['ADMIN'] = country_counts['ADMIN'].map(country_mapping)
        
        # Merge with world map data
        merged = world.merge(country_counts, on='ADMIN', how='left')
        merged['count'] = merged['count'].fillna(0)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        
        # Plot the map
        merged.plot(column='count', 
                    ax=ax,
                    legend=True,
                    legend_kwds={'label': 'Număr de automobile'},
                    missing_kwds={'color': 'lightgrey'},
                    cmap='YlOrRd')
        
        # Customize the map
        ax.set_title('Distribuția automobilelor pe țări', fontsize=16, pad=20)
        ax.axis('off')
        
        # Add country labels
        for idx, row in merged.iterrows():
            if row['count'] > 0:
                ax.annotate(text=f"{row['ADMIN']}\n{int(row['count'])}", 
                           xy=row.geometry.centroid.coords[0],
                           ha='center',
                           va='center',
                           fontsize=8)
        
        # Display the map in Streamlit
        st.pyplot(fig)
        
        # Display statistics
        st.markdown("<h4>Statistici pe țări</h4>", unsafe_allow_html=True)
        stats_df = df.groupby('country-of-origin').agg({
            'price': ['mean', 'count'],
            'horsepower': 'mean',
            'city-mpg': 'mean'
        }).round(2)
        
        stats_df.columns = ['Număr de automobile', 'Preț mediu (USD)', 'Putere medie (CP)', 'Consum mediu oraș (mpg)']
        stats_df = stats_df.reset_index()
        stats_df.columns = ['Țară'] + list(stats_df.columns[1:])
        
        st.dataframe(stats_df)
        
        # Additional insights
        st.markdown("""
        <div class="highlight-box">
        <h4>Observații:</h4>
        <ul>
            <li>Majoritatea automobilelor din setul de date provin din țări cu o tradiție puternică în industria auto</li>
            <li>Fiecare țară are propriile caracteristici distinctive în ceea ce privește prețul, puterea și eficiența</li>
            <li>Distribuția geografică reflectă concentrarea producătorilor auto în anumite regiuni ale lumii</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"A apărut o eroare la încărcarea datelor geografice: {str(e)}")
        st.info("Vă rugăm să verificați conexiunea la internet și să încercați din nou.")

# Modeling section functions
def show_clustering():
    st.markdown("<h3>Clusterizare K-means</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight-box">
    <p>Clusterizarea K-means este o metodă de învățare nesupervizată care grupează datele în k clustere. 
    Vom analiza automobilele folosind caracteristici numerice relevante pentru a identifica grupuri naturale de mașini cu caracteristici similare.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Select features for clustering
    features_for_clustering = ['engine-size', 'horsepower', 'price', 'curb-weight']
    
    # Prepare data - handle missing values first
    X = st.session_state.processed_data[features_for_clustering].copy()
    
    # Replace '?' with NaN
    X = X.replace('?', np.nan)
    
    # Convert to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill missing values with median
    for col in X.columns:
        X[col].fillna(X[col].median(), inplace=True)
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Select number of clusters
    n_clusters = st.slider("Selectati numarul de clustere", 2, 6, 3)
    
    # Apply K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add clusters to original data
    df_with_clusters = st.session_state.processed_data.copy()
    df_with_clusters['Cluster'] = clusters
    
    # Display results
    st.markdown("<h4>Distributia automobilelor pe clustere</h4>", unsafe_allow_html=True)
    
    # Calculate statistics for each cluster
    cluster_stats = pd.DataFrame()
    for cluster in range(n_clusters):
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster][features_for_clustering]
        # Convert to numeric and handle missing values
        for col in cluster_data.columns:
            cluster_data[col] = pd.to_numeric(cluster_data[col].replace('?', np.nan), errors='coerce')
            cluster_data[col].fillna(cluster_data[col].median(), inplace=True)
        
        # Calculate mean and std for each feature
        cluster_mean = cluster_data.mean()
        cluster_std = cluster_data.std()
        
        # Add to stats DataFrame
        cluster_stats[f'Cluster_{cluster}_mean'] = cluster_mean
        cluster_stats[f'Cluster_{cluster}_std'] = cluster_std
    
    # Display statistics
    st.dataframe(cluster_stats.round(2))
    
    # Interpret cluster characteristics
    st.markdown("""
    <div class="highlight-box">
    <h4>Interpretarea clusterelor:</h4>
    <ul>
        <li>Fiecare cluster reprezinta un grup distinct de masini cu caracteristici similare</li>
        <li>Valorile medii si deviatiile standard ne ajuta sa intelegem distributia caracteristicilor in fiecare cluster</li>
        <li>Clusterul cu pret mediu mai mare poate fi asociat cu masini de lux</li>
        <li>Clusterul cu putere si dimensiune motor mai mari poate fi asociat cu masini sport</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Visualize clusters
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')
    ax.set_xlabel('Engine Size (standardizat)')
    ax.set_ylabel('Horsepower (standardizat)')
    ax.set_title('Vizualizare clustere K-means')
    plt.colorbar(scatter, label='Cluster')
    st.pyplot(fig)
    
    # Display examples from each cluster
    st.markdown("<h4>Exemple de masini din fiecare cluster</h4>", unsafe_allow_html=True)
    for cluster in range(n_clusters):
        st.markdown(f"**Cluster {cluster}:**")
        # Afisam doar coloanele existente
        display_columns = ['make'] + features_for_clustering
        cluster_cars = df_with_clusters[df_with_clusters['Cluster'] == cluster][display_columns].head()
        # Convert numeric columns for display
        for col in features_for_clustering:
            cluster_cars[col] = pd.to_numeric(cluster_cars[col].replace('?', np.nan), errors='coerce')
            cluster_cars[col].fillna(cluster_cars[col].median(), inplace=True)
        st.dataframe(cluster_cars.round(2))

def show_logistic_regression():
    st.markdown("<h3>Regresie Logistica</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight-box">
    <p>Regresia logistica va fi folosita pentru a prezice daca o masina este de lux sau nu, 
    bazandu-ne pe caracteristicile sale. O masina este considerata de lux daca pretul sau este peste media + o deviatie standard.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Prepare data
    df = st.session_state.processed_data.copy()
    
    # Convert price to numeric and handle missing values
    df['price'] = pd.to_numeric(df['price'].replace('?', np.nan), errors='coerce')
    df['price'].fillna(df['price'].median(), inplace=True)
    
    # Define luxury cars
    price_mean = df['price'].mean()
    price_std = df['price'].std()
    luxury_threshold = price_mean + price_std
    
    # Create target variable
    y = (df['price'] > luxury_threshold).astype(int)
    
    # Select features for prediction
    features = ['engine-size', 'horsepower', 'curb-weight', 'wheel-base', 'length']
    
    # Convert features to numeric and handle missing values
    X = df[features].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col].replace('?', np.nan), errors='coerce')
        X[col].fillna(X[col].median(), inplace=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    st.markdown("<h4>Performanta modelului</h4>", unsafe_allow_html=True)
    st.write(f"Precizia pe setul de antrenare: {train_score:.2%}")
    st.write(f"Precizia pe setul de test: {test_score:.2%}")
    
    # Display feature importance
    st.markdown("<h4>Importanta caracteristicilor</h4>", unsafe_allow_html=True)
    feature_importance = pd.DataFrame({
        'Caracteristica': features,
        'Coeficient': model.coef_[0]
    })
    feature_importance = feature_importance.sort_values('Coeficient', key=abs, ascending=False)
    st.dataframe(feature_importance)
    
    # Visualize feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='Coeficient', y='Caracteristica', ax=ax)
    ax.set_title('Importanta caracteristicilor in predictia masinilor de lux')
    st.pyplot(fig)
    
    # Add prediction section
    st.markdown("<h4>Predictie pentru o masina noua</h4>", unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight-box">
    <p>Introduceti caracteristicile unei masini pentru a prezice daca este de lux sau nu.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create input fields for each feature
    col1, col2 = st.columns(2)
    with col1:
        engine_size = st.number_input('Dimensiune motor (cm³)', min_value=0.0, value=float(X['engine-size'].mean()))
        horsepower = st.number_input('Putere (CP)', min_value=0.0, value=float(X['horsepower'].mean()))
        curb_weight = st.number_input('Greutate (kg)', min_value=0.0, value=float(X['curb-weight'].mean()))
    
    with col2:
        wheel_base = st.number_input('Distanta intre punți (inch)', min_value=0.0, value=float(X['wheel-base'].mean()))
        length = st.number_input('Lungime (inch)', min_value=0.0, value=float(X['length'].mean()))
    
    # Create prediction button
    if st.button('Prezice daca masina este de lux'):
        # Prepare input data
        input_data = pd.DataFrame({
            'engine-size': [engine_size],
            'horsepower': [horsepower],
            'curb-weight': [curb_weight],
            'wheel-base': [wheel_base],
            'length': [length]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        # Display results
        if prediction == 1:
            st.success(f"Aceasta masina este prezisa ca fiind de lux (probabilitate: {probability:.2%})")
        else:
            st.info(f"Aceasta masina nu este prezisa ca fiind de lux (probabilitate: {probability:.2%})")
    
    # Add interpretation
    st.markdown("""
    <div class="highlight-box">
    <h4>Interpretarea rezultatelor:</h4>
    <ul>
        <li>Coeficientii pozitivi indica o relatie directa intre caracteristica si probabilitatea ca masina sa fie de lux</li>
        <li>Coeficientii negativi indica o relatie inversa</li>
        <li>Cu cat valoarea absoluta a coeficientului este mai mare, cu atat caracteristica are o influenta mai puternica</li>
        <li>Precizia modelului ne indica cat de bine putem prezice daca o masina este de lux sau nu</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def show_multiple_regression():
    st.markdown("<h3>Regresie Multipla</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight-box">
    <p>Regresia multipla va fi folosita pentru a prezice pretul masinii bazandu-ne pe multiple caracteristici.
    Aceasta analiza ne permite sa intelegem care caracteristici au cel mai mare impact asupra pretului.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Prepare data
    df = st.session_state.processed_data.copy()
    
    # Select features for prediction
    features = ['engine-size', 'horsepower', 'curb-weight', 'wheel-base', 'length', 'width', 'height']
    
    # Convert features to numeric and handle missing values
    X = df[features].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col].replace('?', np.nan), errors='coerce')
        X[col].fillna(X[col].median(), inplace=True)
    
    # Convert target variable to numeric and handle missing values
    y = pd.to_numeric(df['price'].replace('?', np.nan), errors='coerce')
    y.fillna(y.median(), inplace=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Add constant to X for regression
    X_train_with_const = sm.add_constant(X_train)
    X_test_with_const = sm.add_constant(X_test)
    
    # Train model
    model = sm.OLS(y_train, X_train_with_const).fit()
    
    # Display results
    st.markdown("<h4>Rezultatele regresiei</h4>", unsafe_allow_html=True)
    st.write(model.summary().tables[0].as_html(), unsafe_allow_html=True)
    
    # Evaluate predictions
    y_pred = model.predict(X_test_with_const)
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    r2 = model.rsquared
    
    st.markdown("<h4>Metrici de performanta</h4>", unsafe_allow_html=True)
    st.write(f"R-squared: {r2:.4f}")
    st.write(f"RMSE: {rmse:.2f} USD")
    
    # Visualize predictions vs actual values
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Pret real (USD)')
    ax.set_ylabel('Pret prezis (USD)')
    ax.set_title('Predictii vs Valori reale')
    st.pyplot(fig)
    
    # Display coefficients and p-values
    st.markdown("<h4>Coefficienti si semnificatie statistica</h4>", unsafe_allow_html=True)
    coef_df = pd.DataFrame({
        'Caracteristica': ['Intercept'] + features,
        'Coeficient': model.params,
        'P-value': model.pvalues
    })
    coef_df = coef_df.sort_values('P-value')
    st.dataframe(coef_df)
    
    # Visualize coefficients (excluding intercept)
    fig, ax = plt.subplots(figsize=(10, 6))
    coef_df_no_intercept = coef_df[coef_df['Caracteristica'] != 'Intercept']
    sns.barplot(data=coef_df_no_intercept, x='Coeficient', y='Caracteristica', ax=ax)
    ax.set_title('Influenta caracteristicilor asupra pretului')
    st.pyplot(fig)
    
    # Add prediction section
    st.markdown("<h4>Predictie pret masina</h4>", unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight-box">
    <p>Introduceti caracteristicile unei masini pentru a prezice pretul sau.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create input fields for each feature
    col1, col2 = st.columns(2)
    with col1:
        engine_size = st.number_input('Dimensiune motor (cm³)', min_value=0.0, value=float(X['engine-size'].mean()))
        horsepower = st.number_input('Putere (CP)', min_value=0.0, value=float(X['horsepower'].mean()))
        curb_weight = st.number_input('Greutate (kg)', min_value=0.0, value=float(X['curb-weight'].mean()))
        wheel_base = st.number_input('Distanta intre punți (inch)', min_value=0.0, value=float(X['wheel-base'].mean()))
    
    with col2:
        length = st.number_input('Lungime (inch)', min_value=0.0, value=float(X['length'].mean()))
        width = st.number_input('Latime (inch)', min_value=0.0, value=float(X['width'].mean()))
        height = st.number_input('Inaltime (inch)', min_value=0.0, value=float(X['height'].mean()))
    
    # Create prediction button
    if st.button('Prezice pretul masinii'):
        # Prepare input data
        input_data = pd.DataFrame({
            'const': [1.0],  # Add constant term first
            'engine-size': [engine_size],
            'horsepower': [horsepower],
            'curb-weight': [curb_weight],
            'wheel-base': [wheel_base],
            'length': [length],
            'width': [width],
            'height': [height]
        })
        
        # Make prediction
        predicted_price = model.predict(input_data)[0]
        
        # Display results
        st.success(f"Pretul estimat al masinii este: ${predicted_price:,.2f}")
        
        # Show confidence interval
        prediction_interval = model.get_prediction(input_data).conf_int(alpha=0.05)
        st.info(f"Interval de incredere 95%: ${prediction_interval[0][0]:,.2f} - ${prediction_interval[0][1]:,.2f}")
    
    # Add interpretation
    st.markdown("""
    <div class="highlight-box">
    <h4>Interpretarea rezultatelor:</h4>
    <ul>
        <li>R-squared ne indica proportia variatiei in pret care poate fi explicata de caracteristicile selectate</li>
        <li>RMSE ne indica eroarea medie in predictia pretului in dolari</li>
        <li>P-values mai mici de 0.05 indica o relatie statistica semnificativa intre caracteristica si pret</li>
        <li>Coeficientii pozitivi indica o relatie directa intre caracteristica si pret</li>
        <li>Coeficientii negativi indica o relatie inversa</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()