import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./METABRIC_RNA_Mutation.csv")

print(df.head())
print(df.describe())
print(df.info())

#print('Baseline accuracy:' )
#print(df["overall_survival"].value_counts()/df["overall_survival"].count())

#Da 0 a 31 ci sono i dati clinici
#da 31 a 520 capire cosa c'è 
#da 520 alla fine ci sono le mutazioni


#Bisogna fare preprocessing dei dati.
#Mi concentro sui dati clinici presenti

total = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([total, percent], axis = 1, keys = ["Total_NaN", "Percent_NaN"])
missing_data.head(10)

#Come si può vedere, i dati clinici hanno alcune categorie in cui sono presenti dei dati mancanti, 
#come per esempio il campo "tumor_stage" con il 26% di valori mancanti.

#A) Relazione tra dati clinici e risultati.

#Estraggo dal dataset solo i dati relativi ai dati clinici
clinical_features_to_drop = df.columns[31:] #mi segno tutte le colonne da eliminare
clinical_df = df.drop(clinical_features_to_drop, axis = 1) #Con axis =1 indico che devo eliminare delle colonne
#con .drop ottengo un nuovo dataframe (senza modificare quello originale) con tutte le colonne (perchè ho usato axis=1 per indicare le colonne) di prima tranne quelle passate come parametro
#se avessi usato axis=0 o default immagino, avrei eliminato delle righe e non delle colonne.
clinical_df.head()

#Creo una funzione per normalizzare tutti i valori numerici presenti nel dataset clinical_df
def to_standard (df):
    df_numeric = df[df.select_dtypes(include = np.number).columns.to_list()] #creo un nuovo dataframe che contiene solo le colonne numeriche che vado a normalizzare con StandardScaler 
    #per ogni feature i-esima calcolo la media e la deviazione standard  e faccio x' = (x -mean i)/stddev i
    ss = StandardScaler()
    std = ss.fit_transform(df_numeric) #viene applicata la trasformazione di standardizzazione ai dati nel DataFrame num_df, 
                                   #ottenendo come risultato una matrice numpy contenente i dati standardizzati. 
                                   # Questa matrice non ha più l'informazione sugli indici e le colonne del DataFrame originale num_df.
    std_df = pd.DataFrame(std, index = df_numeric.index, columns = df_numeric.columns)
    return std_df                            

print(to_standard(clinical_df))

#Ora vediamo dei grafici che connettono alcune colonne all'output finale.

grafico = plt.figure(figsize = (15, 20))
#crea una nuova figura vuota utilizzando la libreria Matplotlib.
#L'argomento figsize specifica le dimensioni della figura in pollici (larghezza, altezza). Nell'esempio fornito, la figura viene impostata su una larghezza di 20 pollici e un'altezza di 25 pollici.
#Assegnando il risultato della chiamata plt.figure() alla variabile grafico, è possibile fare riferimento alla figura in seguito per aggiungere subplot o personalizzare la sua formattazione.
#Una volta creata la figura, è possibile aggiungere subplot, grafici o visualizzazioni utilizzando i metodi e le funzioni forniti da Matplotlib o altre librerie come Seaborn. 

j = 0
num_clinical_columns = ['age_at_diagnosis', 'lymph_nodes_examined_positive','mutation_count','nottingham_prognostic_index', 'overall_survival_months', 'tumor_size' ]
for i in clinical_df[num_clinical_columns].columns:
    #In sintesi, clinical_df[num_clinical_columns] restituisce un DataFrame con le colonne specificate, 
    # mentre clinical_df[num_clinical_columns].columns restituisce un oggetto Index contenente i nomi 
    # delle colonne del DataFrame risultante. Poteo usare direttamente num_clinical_columns.
    plt.subplot(6, 4, j+1) #perchè la numerazione inizia da 1
    # Una griglia di subplot è un layout organizzato in una griglia di celle, 
    # ognuna delle quali può contenere un grafico o una visualizzazione separata.
    # Ad esempio, una griglia di subplot può essere composta da 2 righe e 2 colonne, 
    # formando una griglia di 4 celle. Ogni cella può essere considerata come un subplot 
    # indipendente in cui è possibile disegnare un grafico separato. Nel nostro caso 6 righe e 4 colonne,
    # 6 x 4 = 24 subplot indipendenti. j (terzo parametro) indica la posizione all'interno della griglia. Con il
    # e aumentando j praticamente ogni nuovo grafico si troverà a fianco del precedente, nella nuova posizione.
    j += 1
    sns.distplot(clinical_df[i] [clinical_df['overall_survival']==1], color = 'g', label = 'survived')
    # crea un grafico a distribuzione della variabile clinical_df[i] per i casi in cui la colonna 'overall_survival' 
    # del DataFrame clinical_df è uguale a 1 (indicando i casi di sopravvivenza).
    sns.distplot(clinical_df[i] [clinical_df['overall_survival']==0], color='r', label = 'died')
    #Idem ma per i casi di decesso
    grafico.suptitle('Clinical Data Analysis')

grafico.tight_layout() # viene utilizzata per migliorare la disposizione dei subplot all'interno di una figura, in modo che siano distribuiti in modo uniforme e che non si sovrappongano.

grafico.subplots_adjust(top=0.95) #viene utilizzata per regolare i margini superiori della figura, consentendo di spostare verso il basso l'area di disegno dei subplot.

plt.show()

#Il collegamento tra fig o 'grafico'(nel mio caso) e i subplot avviene in modo implicito attraverso la sequenza di comandi che
#  utilizzano la libreria Matplotlib. Quando si crea un nuovo subplot con plt.subplot(), 
# il subplot viene automaticamente aggiunto alla figura 'variabile' grafico in questo caso.
# Inoltre, tutte le operazioni di tracciamento dei grafici, come sns.distplot(), 
# vengono eseguite all'interno del contesto della figura 'grafico'.
# In sostanza, la figura 'grafico' o fig rappresenta il contenitore principale dei subplot 
# creati con plt.subplot(), anche se non viene fatto un collegamento esplicito
#  nel codice. Questo è il comportamento predefinito di Matplotlib quando si 
# lavora con una singola figura.

#QUALI INFORMAZIONI RIESCO A RICAVARE DA QUESTO DATASET?

#Feature selection:
corr_matrix = clinical_df.corr()
plt.figure(figsize=(8, 8)) #crea una nuova figura
sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm')
plt.title('Correlation Matrix')
plt.show() # È una funzione fornita dal modulo matplotlib.pyplot che mostra tutti i grafici che sono stati creati nel contesto corrente.

#Non vi è una grande correlazione tra i vari dati clinici presenti.


#La differenza principale tra "plt.subplots" e "plt.subplot" è che "plt.subplots" crea una griglia di subplot 
# in una singola chiamata, restituendo sia la figura che gli assi dei subplot, 
# mentre "plt.subplot" crea un singolo subplot in una chiamata, restituendo solo gli assi del subplot.
# Utilizzando "plt.subplots", è possibile accedere agli assi dei subplot individuandoli tramite l'indice nell'elenco "ax". 
# Ad esempio, "ax[0]" rappresenta il primo subplot nella griglia e "ax[1]" rappresenta il secondo subplot.

fig, ax = plt.subplots(ncols = 2, figsize = (15,3), sharey = True)
#crea una figura con due assi, disposti in una griglia con due colonne (ncols=2) e una riga. 
# La dimensione totale della figura è impostata su (15, 3) utilizzando il parametro figsize, 
#che specifica la larghezza e l'altezza della figura in pollici.
# Inoltre, il parametro sharey=True viene utilizzato per condividere lo stesso asse y tra i due assi, 
# il che significa che entrambi gli assi avranno la stessa scala sull'asse y.Il risultato di questo codice 
# è una figura con due assi, pronti per essere utilizzati per disegnare grafici.
#Qui utilizzo il primo asse ovvero ax[0]
two_colors = ["#FF0000", "#0000FF"]
sns.boxplot(x = 'overall_survival_months', y = 'overall_survival', orient = 'h', data = clinical_df, ax = ax[0], palette = two_colors)
#Qui utilizzo il secondo asse ovvero ax[1]
sns.boxplot(x = 'age_at_diagnosis', y = 'overall_survival', orient = 'h', data = clinical_df, ax = ax[1], palette = two_colors)

fig.suptitle('The Distribution of Survival time in months and age with Target Attribute', fontsize = 18)

ax[0].set_xlabel('Total survival time in Months')
ax[0].set_ylabel('survival')
ax[1].set_xlabel('Age at diagnosis')
ax[1].set_ylabel('')

plt.show()
#Per confrontare le due classi di pazienti che sono sopravvissuti e i pazienti che non lo hanno fatto, 
# possiamo vedere la differenza tra le due distribuzioni nella colonna 'age_at_diagnosis', in quanto i pazienti 
# che erano più giovani al momento della diagnosi di cancro al seno avevano maggiori probabilità di sopravvivere(DA CAMBIARE)

fig, ax = plt.subplots(ncols = 2, figsize = (15,3), sharey = True)
fig.suptitle('The Distribution of Survival and Recurrence with Target Attribute', fontsize = 18)
three_colors = ["#00FF00", "#FF0000", "#0000FF"]
sns.boxplot(x = 'overall_survival_months', y = 'death_from_cancer', orient = 'h', data = clinical_df, ax = ax[0], palette = three_colors)
sns.boxplot(x = 'age_at_diagnosis', y = 'death_from_cancer', orient = 'h', data = clinical_df, ax = ax[1], palette = three_colors)

ax[0].set_xlabel('Total survival time in Months')
ax[0].set_ylabel('survival')
ax[1].set_xlabel('Age at diagnosis')
ax[1].set_ylabel('')

plt.show()

#Qui possiamo notare come, chi muore per la malattia di solito muore relativamente subito e non dopo vari mesi.
# Infatti poi dopo vari mesi muoiono persone ma non per la malattia ma per altre cause.

print(clinical_df.columns)

#Posso confrontare ancora : Tumor_size e tumor stage
#Sopravvivenza o no in base all'età, diametro tumore, numero di linfonodi positivi e di questi quanti morti per tumore e quanti per altre cause o ancora vivi
#Confronto tra chi ha fatto la chemio, la terapia ormonale e chi radio terapia ed è vivo o morto (e chi ha fatto più cose insieme)

#Poi vedere meglio la correlazione tra le features



