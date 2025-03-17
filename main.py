import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap');
    
    
    .title {stre
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
        display:none;
        
    }
    
    
    .st-emotion-cache-kgpedg {height:1%}
    .st-emotion-cache-gi0tri {display:none}
    h1:hover{cursor:default}
    
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="title">Analiză și modelare a datelor despre automobile <span class="car-emoji">🏎️</span></h1>',
            unsafe_allow_html = True)


with st.sidebar:
    st.title("Pachete software - Proiect")
    section = st.radio('',["Date", "Descriere date","Statistici"])

if section == "Date":
    st.header("Date inițiale")
    st.markdown("""
        <div class="highlight-box">
        <h3>Sursa datelor</h3>
        Acest set de date a fost preluat de pe platforma Kaggle. Poate fi accesat la următorul link:
        <a href="https://www.kaggle.com/datasets/toramky/automobile-dataset/data" target="_blank">Automobile Dataset - Kaggle</a>
        </div>
        """, unsafe_allow_html=True)
    tabel=pd.read_csv(filepath_or_buffer="Automobile_data.csv")
    st.dataframe(tabel)
elif section== "Descriere date":
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
elif section == "Statistici":
    st.header("Statistici")
    
    # loading data
    tabel_original = pd.read_csv(filepath_or_buffer="Automobile_data.csv")
    
    # missing values handling
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

    tabel_procesat = tabel_original.copy()
    

    tabel_procesat = tabel_procesat.replace('?', np.nan)
    
    #number of na values
    valori_lipsa_initial = tabel_procesat.isna().sum()
    
    #converting numeric values to float
    coloane_numerice = ['normalized-losses', 'bore', 'stroke', 'horsepower', 'peak-rpm', 'price']
    for coloana in coloane_numerice:
        tabel_procesat[coloana] = pd.to_numeric(tabel_procesat[coloana], errors='coerce')
    

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4>Valori lipsă înainte de procesare</h4>", unsafe_allow_html=True)
        st.dataframe(valori_lipsa_initial[valori_lipsa_initial > 0])
    
    #replacing missing values with mean
    for coloana in coloane_numerice:
        tabel_procesat[coloana].fillna(tabel_procesat[coloana].mean(), inplace=True)
    
    #replacing missing values with mode for non-numeric
    coloane_categorice = [col for col in tabel_procesat.columns if col not in coloane_numerice]
    for coloana in coloane_categorice:
        tabel_procesat[coloana].fillna(tabel_procesat[coloana].mode()[0], inplace=True)
    

    valori_lipsa_final = tabel_procesat.isna().sum()

    #processed data
    st.markdown("<h3>Tabel fără valori lipsă</h3>", unsafe_allow_html=True)
    st.dataframe(tabel_procesat)
    
    # desccriptive statistics
    st.markdown("<h3>Statistici descriptive</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight-box">
    <p>Mai jos sunt prezentate statisticile descriptive pentru coloanele numerice din setul de date. 
    Acestea includ: numărul de valori, media, deviația standard, valoarea minimă, cuartilele și valoarea maximă.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # descriptive statistics for numeric variables
    statistici = tabel_procesat.describe()
    st.dataframe(statistici)
    
    # visualisations
    st.markdown("<h3>Distribuția prețurilor</h3>", unsafe_allow_html=True)
    
    fig_pret = pd.DataFrame({
        'Preț (USD)': tabel_procesat['price']
    })
    
    st.bar_chart(fig_pret)
    
    # power-price relationship
    st.markdown("<h3>Relația între putere și preț</h3>", unsafe_allow_html=True)
    
    fig_putere_pret = pd.DataFrame({
        'Putere (CP)': tabel_procesat['horsepower'],
        'Preț (USD)': tabel_procesat['price']
    })
    
    st.scatter_chart(fig_putere_pret)

    st.markdown("<h3>Distribuția puterii motorului în funcție de tipul de combustibil</h3>", unsafe_allow_html=True)

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

    # price distribution for each make
    st.markdown("<h3>Distribuția prețurilor în funcție de marca mașinii</h3>", unsafe_allow_html=True)

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

    # engine size vs urban consumption
    st.markdown("<h3>Relația între dimensiunea motorului și consumul în oraș</h3>", unsafe_allow_html=True)

    fig_motor_consum = pd.DataFrame({
        'Dimensiune motor (cm³)': tabel_procesat['engine-size'],
        'Consum oraș (mpg)': tabel_procesat['city-mpg']
    })

    st.write("Acest grafic arată relația inversă între dimensiunea motorului și eficiența consumului în oraș.")
    st.scatter_chart(fig_motor_consum)

    # urban vs highway consumption
    st.markdown("<h3>Comparație între consumul în oraș și pe autostradă</h3>", unsafe_allow_html=True)

    fig_consum = pd.DataFrame({
        'Consum oraș (mpg)': tabel_procesat['city-mpg'],
        'Consum autostradă (mpg)': tabel_procesat['highway-mpg']
    })

    st.write("Acest grafic arată corelația dintre consumul în oraș și consumul pe autostradă.")
    st.scatter_chart(fig_consum)

    # chassis distribution
    st.markdown("<h3>Distribuția tipurilor de caroserie</h3>", unsafe_allow_html=True)

    fig_caroserie = pd.DataFrame(tabel_procesat['body-style'].value_counts()).reset_index()
    fig_caroserie.columns = ['Tip caroserie', 'Număr']

    st.write("Acest grafic arată distribuția tipurilor de caroserie în setul de date.")
    st.bar_chart(fig_caroserie, x='Tip caroserie', y='Număr')

    # weight distribution by chassis
    st.markdown("<h3>Distribuția greutății în funcție de tipul de caroserie</h3>", unsafe_allow_html=True)

    # stats for each chassis
    greutate_caroserie_stats = tabel_procesat.groupby('body-style')['curb-weight'].agg(['mean', 'median', 'min', 'max']).reset_index()
    greutate_caroserie_stats.columns = ['Tip caroserie', 'Greutate medie', 'Greutate mediană', 'Greutate minimă', 'Greutate maximă']

    st.write("Acest tabel arată statisticile greutății pentru fiecare tip de caroserie.")
    st.dataframe(greutate_caroserie_stats)


    fig_greutate_caroserie_bar = pd.DataFrame({
        'Tip caroserie': greutate_caroserie_stats['Tip caroserie'],
        'Greutate medie': greutate_caroserie_stats['Greutate medie']
    }).set_index('Tip caroserie')

    st.bar_chart(fig_greutate_caroserie_bar)

    # compression ratio vs power
    st.markdown("<h3>Relația între raportul de compresie și putere</h3>", unsafe_allow_html=True)

    fig_compresie_putere = pd.DataFrame({
        'Raport compresie': tabel_procesat['compression-ratio'],
        'Putere (CP)': tabel_procesat['horsepower']
    })

    st.write("Acest grafic arată relația dintre raportul de compresie al motorului și puterea dezvoltată.")
    st.scatter_chart(fig_compresie_putere)

    st.markdown("<h3>Matricea de corelație pentru variabilele numerice</h3>", unsafe_allow_html=True)


    coloane_pentru_corelare = ['wheel-base', 'length', 'width', 'height', 'curb-weight',
                              'engine-size', 'compression-ratio', 'horsepower', 'city-mpg', 'highway-mpg', 'price']
    matrice_corelatie = tabel_procesat[coloane_pentru_corelare].corr()

    st.write("Această matrice de corelație arată relațiile liniare între variabilele numerice din setul de date.")
    st.dataframe(matrice_corelatie.style.background_gradient(cmap='coolwarm'))




    # boxplot helper
    def afiseaza_boxplot(data, x, y, titlu, xlabel, ylabel):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=data, x=x, y=y, ax=ax)
        ax.set_title(titlu)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.image(buf)

    # power by fuel type
    st.markdown("<h3>Boxplot-uri pentru analiza distribuțiilor</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight-box">
    <p>Boxplot-urile de mai jos oferă o reprezentare vizuală a distribuției datelor, arătând mediana, cuartilele și valorile extreme.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h4>Boxplot: Puterea motorului în funcție de tipul de combustibil</h4>", unsafe_allow_html=True)
    afiseaza_boxplot(
        tabel_procesat, 
        'fuel-type', 
        'horsepower', 
        'Distribuția puterii motorului pentru fiecare tip de combustibil',
        'Tip combustibil',
        'Putere (CP)'
    )

    # price by make
    st.markdown("<h4>Boxplot: Prețurile în funcție de marca mașinii</h4>", unsafe_allow_html=True)
    afiseaza_boxplot(
        tabel_marci_frecvente, 
        'make', 
        'price', 
        'Distribuția prețurilor pentru mărcile cele mai frecvente',
        'Marca',
        'Preț (USD)'
    )

    # weight by chassis
    st.markdown("<h4>Boxplot: Greutatea în funcție de tipul de caroserie</h4>", unsafe_allow_html=True)
    afiseaza_boxplot(
        tabel_procesat, 
        'body-style', 
        'curb-weight', 
        'Distribuția greutății pentru fiecare tip de caroserie',
        'Tip caroserie',
        'Greutate (kg)'
    )

    # urban consumption by no of cylinders
    st.markdown("<h4>Boxplot: Consumul în oraș în funcție de numărul de cilindri</h4>", unsafe_allow_html=True)
    afiseaza_boxplot(
        tabel_procesat, 
        'num-of-cylinders', 
        'city-mpg', 
        'Distribuția consumului în oraș pentru fiecare număr de cilindri',
        'Număr cilindri',
        'Consum oraș (mpg)'
    )

    # price by chassis
    st.markdown("<h4>Boxplot: Prețurile în funcție de tipul de caroserie</h4>", unsafe_allow_html=True)
    afiseaza_boxplot(
        tabel_procesat, 
        'body-style', 
        'price', 
        'Distribuția prețurilor pentru fiecare tip de caroserie',
        'Tip caroserie',
        'Preț (USD)'
    )

    # observations
    st.markdown("<h3>Concluzii și observații finale</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight-box">
    <h4>Observații din analiza graficelor</h4>
    <ul>
        <li>Există o corelație puternică pozitivă între prețul mașinii și puterea motorului, după cum se poate observa în graficul de dispersie și boxplot-uri.</li>
        <li>Mașinile pe benzină tind să aibă o putere mai mare decât cele diesel, aspect evidențiat clar în boxplot-ul puterii în funcție de tipul de combustibil.</li>
        <li>Există o corelație negativă între dimensiunea motorului și eficiența consumului - motoarele mai mari consumă mai mult combustibil.</li>
        <li>Consumul pe autostradă este întotdeauna mai eficient decât consumul în oraș, dar cele două valori sunt puternic corelate.</li>
        <li>Tipurile de caroserie influențează semnificativ greutatea mașinii, cu wagon și sedan fiind în general mai grele decât hatchback, după cum se poate observa în boxplot-ul greutății în funcție de tipul de caroserie.</li>
        <li>Raportul de compresie nu pare să aibă o corelație liniară puternică cu puterea motorului.</li>
        <li>Numărul de cilindri are un impact direct asupra consumului de combustibil - mașinile cu mai mulți cilindri tind să consume mai mult, aspect vizibil în boxplot-ul consumului în funcție de numărul de cilindri.</li>
        <li>Prețurile variază semnificativ în funcție de marca mașinii, cu mărci precum BMW și Mercedes-Benz având prețuri medii mai ridicate.</li>
        <li>Tipul de caroserie influențează și prețul mașinii, cu convertible și hardtop având în general prețuri mai mari decât hatchback și sedan.</li>
    </ul>

    <h4>Concluzii generale</h4>
    <p>Analiza setului de date auto ne-a permis să identificăm mai multe relații importante între caracteristicile mașinilor. Puterea motorului, tipul de caroserie și marca sunt factori determinanți pentru preț. De asemenea, există compromisuri clare între performanță (putere) și eficiență (consum de combustibil). Aceste informații pot fi valoroase pentru consumatori în procesul de selecție a unei mașini care să răspundă cel mai bine nevoilor lor, ținând cont de buget și preferințe.</p>
    </div>
    """, unsafe_allow_html=True)


