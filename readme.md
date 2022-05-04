Eron Fraud Identification
===

Python
---

---

**Author :** Matheus Willian Machado  
**Date :** Aug 10, 2018

---

Project Overview
---

>Banque o detetive e coloque suas habilidades de aprendizado de máquina em uso através da construção de um algoritmo para identificar funcionários da Enron que possam ter cometido fraude. Sua base será um conjunto de dados financeiros e de e-mail público da Enron.
> 
> (Udacity).

---

## Introduction

> Em 2000, Enron era uma das maiores empresas dos Estados Unidos. Já em 2002, ela colapsou e quebrou devido a uma fraude que envolveu grande parte da corporação. Resultando em uma investigação federal, muitos dados que são normalmente confidenciais, se tornaram públicos, incluindo dezenas de milhares de e-mails e detalhes financeiros para os executivos dos mais altos níveis da empresa. Neste projeto, você irá bancar o detetive, e colocar suas habilidades na construção de um modelo preditivo que visará determinar se um funcionário é ou não um funcionário de interesse (POI). Um funcionário de interesse é um funcionário que participou do escândalo da empresa Enron. Para te auxiliar neste trabalho de detetive, nós combinamos os dados financeiros e sobre e-mails dos funcionários investigados neste caso de fraude, o que significa que eles foram indiciados, fecharam acordos com o governo, ou testemunharam em troca de imunidade no processo.
> 
> (Udacity).

---

## Libraries


```python
import warnings
warnings.filterwarnings("ignore")
```


```python
import pickle
import pandas as pd
import numpy as np

from time import time

from sklearn.preprocessing import Imputer, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier, dump_classifier_and_data
```

---

## Data Exploration


```python
with open('final_project_dataset.pkl', 'rb') as f:
    dic = pickle.load(f)
```

Inicialmente o dataset disponibilizado, e serializado pela biblioteca _pickle_, foi carregado em uma variável. Udacity já havia informado que o arquivo tratava-se de uma combinação dos dados de e-mail e financeiros da base "Eron email and financial", estruturados em forma de dicionário. Onde cada chave representava o nome da pessoa e cada valor continha um outro dicionário, no qual estavam presentes os nomes dos atributos e seus respectivos valores para aquele indivíduo.


```python
data = pd.DataFrame.from_dict(dic, orient='index')
data.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>to_messages</th>
      <th>deferral_payments</th>
      <th>total_payments</th>
      <th>loan_advances</th>
      <th>bonus</th>
      <th>email_address</th>
      <th>restricted_stock_deferred</th>
      <th>deferred_income</th>
      <th>total_stock_value</th>
      <th>...</th>
      <th>from_poi_to_this_person</th>
      <th>exercised_stock_options</th>
      <th>from_messages</th>
      <th>other</th>
      <th>from_this_person_to_poi</th>
      <th>poi</th>
      <th>long_term_incentive</th>
      <th>shared_receipt_with_poi</th>
      <th>restricted_stock</th>
      <th>director_fees</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ALLEN PHILLIP K</th>
      <td>201955</td>
      <td>2902</td>
      <td>2869717</td>
      <td>4484442</td>
      <td>NaN</td>
      <td>4175000</td>
      <td>phillip.allen@enron.com</td>
      <td>-126027</td>
      <td>-3081055</td>
      <td>1729541</td>
      <td>...</td>
      <td>47</td>
      <td>1729541</td>
      <td>2195</td>
      <td>152</td>
      <td>65</td>
      <td>False</td>
      <td>304805</td>
      <td>1407</td>
      <td>126027</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BADUM JAMES P</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>178980</td>
      <td>182466</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>257817</td>
      <td>...</td>
      <td>NaN</td>
      <td>257817</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BANNANTINE JAMES M</th>
      <td>477</td>
      <td>566</td>
      <td>NaN</td>
      <td>916197</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>james.bannantine@enron.com</td>
      <td>-560222</td>
      <td>-5104</td>
      <td>5243487</td>
      <td>...</td>
      <td>39</td>
      <td>4046157</td>
      <td>29</td>
      <td>864523</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>465</td>
      <td>1757552</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BAXTER JOHN C</th>
      <td>267102</td>
      <td>NaN</td>
      <td>1295738</td>
      <td>5634343</td>
      <td>NaN</td>
      <td>1200000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1386055</td>
      <td>10623258</td>
      <td>...</td>
      <td>NaN</td>
      <td>6680544</td>
      <td>NaN</td>
      <td>2660303</td>
      <td>NaN</td>
      <td>False</td>
      <td>1586055</td>
      <td>NaN</td>
      <td>3942714</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BAY FRANKLIN R</th>
      <td>239671</td>
      <td>NaN</td>
      <td>260455</td>
      <td>827696</td>
      <td>NaN</td>
      <td>400000</td>
      <td>frank.bay@enron.com</td>
      <td>-82782</td>
      <td>-201641</td>
      <td>63014</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>69</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>145796</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



Utilizando a biblioteca pandas foi possível transformar o dicionário em um _data frame_ (estrutura tabelada e semelhante a uma matrix). Nele, cada atributo representava uma coluna, cada indivíduo uma linha, e seus nomes foram utilizados como índices.
As 5 primeiras linhas da estrutura foram apresentadas e, com base nelas, notou-se uma grande quantidade de valores "NaN", ou seja, sem informação.


```python
data.replace('NaN', np.nan, inplace=True)
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 146 entries, ALLEN PHILLIP K to YEAP SOON
    Data columns (total 21 columns):
    salary                       95 non-null float64
    to_messages                  86 non-null float64
    deferral_payments            39 non-null float64
    total_payments               125 non-null float64
    loan_advances                4 non-null float64
    bonus                        82 non-null float64
    email_address                111 non-null object
    restricted_stock_deferred    18 non-null float64
    deferred_income              49 non-null float64
    total_stock_value            126 non-null float64
    expenses                     95 non-null float64
    from_poi_to_this_person      86 non-null float64
    exercised_stock_options      102 non-null float64
    from_messages                86 non-null float64
    other                        93 non-null float64
    from_this_person_to_poi      86 non-null float64
    poi                          146 non-null bool
    long_term_incentive          66 non-null float64
    shared_receipt_with_poi      86 non-null float64
    restricted_stock             110 non-null float64
    director_fees                17 non-null float64
    dtypes: bool(1), float64(19), object(1)
    memory usage: 24.1+ KB


Os valores de texto "NaN" foram substituidos pela representação da biblioteca Numpy, facilitando assim sua contagem. Além disso, utilizou-se a função "info" a fim de obter não somente uma visão da quantidade de informações faltantes, como também do tipo de dado de cada atributo, além das dimensões do _dataframe_ (número de linhas e colunas).
É importante ressaltar que a grande maioria dos atributos são do tipo _float_, a exceção de "email_address" e "poi".


```python
label = 'poi'
data[label].value_counts()
```




    False    128
    True      18
    Name: poi, dtype: int64



Para o campo "poi", do tipo booleano, foi realizada uma contagem dos representantes de cada classes ("True" e "False"). O processo revelou um grande desbalanceamento entre elas, já que a grande maioria dos dados estão concentrados na classe "False" enquanto que um pouco mais de 10% encontravam-se na outra.


```python
data['email_address'].nunique()

```




    111




```python
del data['email_address']
```

Quanto ao campo "email_address", do tipo objeto, foi realizado uma contagem de valores únicos, para dimensionar a quantidade de classes presentes no mesmo. Como resultado, haviam tantas quanto células preenchidas (111 itens), o que representaria um atributo candidato a índice. No entanto, sabendo que os nomes dos indivíduos já estavam realizando essa função, optou-se remover aquele campo.


```python
s = data[data[label] == 1].isnull().sum()
s
```




    salary                        1
    to_messages                   4
    deferral_payments            13
    total_payments                0
    loan_advances                17
    bonus                         2
    restricted_stock_deferred    18
    deferred_income               7
    total_stock_value             0
    expenses                      0
    from_poi_to_this_person       4
    exercised_stock_options       6
    from_messages                 4
    other                         0
    from_this_person_to_poi       4
    poi                           0
    long_term_incentive           6
    shared_receipt_with_poi       4
    restricted_stock              1
    director_fees                18
    dtype: int64




```python
limit = data[label].value_counts()[1]/3
few_poi_values = s[s > limit].index.tolist()
few_poi_values
```




    ['deferral_payments',
     'loan_advances',
     'restricted_stock_deferred',
     'deferred_income',
     'director_fees']



Devido a preocupação com o grande desbalanceamento entre os rótulos, somados à noção da quantidade da células não preenchidas, foi estudado a parcela de valores faltante, em todos os campos, para os registros pertencentes à classe "True" (pessoas de interesse). Visando evitar que a falta de informação fosse reconhecida como sendo uma caracteristica desse grupo, foram listados os campos que continha menos de 2/3 de completude, para remoção futura.


```python
data[data.drop(label, axis=1).isnull().all(1)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>to_messages</th>
      <th>deferral_payments</th>
      <th>total_payments</th>
      <th>loan_advances</th>
      <th>bonus</th>
      <th>restricted_stock_deferred</th>
      <th>deferred_income</th>
      <th>total_stock_value</th>
      <th>expenses</th>
      <th>from_poi_to_this_person</th>
      <th>exercised_stock_options</th>
      <th>from_messages</th>
      <th>other</th>
      <th>from_this_person_to_poi</th>
      <th>poi</th>
      <th>long_term_incentive</th>
      <th>shared_receipt_with_poi</th>
      <th>restricted_stock</th>
      <th>director_fees</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LOCKHART EUGENE E</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Linhas que continham apenas valores nulos também foram procuradas.


```python
payments = ['salary',
            'deferral_payments',
            'loan_advances',
            'bonus',
            'deferred_income',
            'expenses',
            'long_term_incentive',
            'other',
            'director_fees',
            'total_payments']
```


```python
data[payments] = data[payments].fillna(0)
```

A fim de verificar a validade da coluna "total_payments" separou-se os atributos de pagamento para soma dos valores e batimento dos resultados. Por se tratar de dados financeiros, optou-se por preencher os dados sem informação com valor 0, tal como está no pdf disponibilizado como insumo, permitindo assim a realização das operações elencadas.


```python
data[data[payments[:-1]].sum(axis=1) != data.total_payments][payments]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>deferral_payments</th>
      <th>loan_advances</th>
      <th>bonus</th>
      <th>deferred_income</th>
      <th>expenses</th>
      <th>long_term_incentive</th>
      <th>other</th>
      <th>director_fees</th>
      <th>total_payments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BELFER ROBERT</th>
      <td>0.0</td>
      <td>-102500.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3285.0</td>
      <td>102500.0</td>
    </tr>
    <tr>
      <th>BHATNAGAR SANJAY</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>137864.0</td>
      <td>137864.0</td>
      <td>15456290.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
correct = ['deferred_income','deferral_payments', 'expenses', 'director_fees', 'total_payments']
data.loc['BELFER ROBERT',correct] = np.array([-102500, 0, 3285, 102500, 3285])

```


```python
correct = ['other', 'expenses', 'director_fees', 'total_payments']
data.loc['BHATNAGAR SANJAY',correct] = np.array([0, 137864, 0, 137864])
```

Durante a validação, dois registros apresentaram valores discrepantes: "BELFER ROBERT" e "BHATNAGAR SANJAY". O pdf foi utilizado para consultar os dados corretos e logo após foi realizado o recadastramento manual dos mesmos.


```python
stock = ['restricted_stock_deferred',
         'restricted_stock',
         'exercised_stock_options',
         'total_stock_value']
```


```python
data[stock] = data[stock].fillna(0)
```


```python
data[data[stock[:-1]].sum(axis=1) != data.total_stock_value][stock]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>restricted_stock_deferred</th>
      <th>restricted_stock</th>
      <th>exercised_stock_options</th>
      <th>total_stock_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BELFER ROBERT</th>
      <td>44093.0</td>
      <td>0.0</td>
      <td>3285.0</td>
      <td>-44093.0</td>
    </tr>
    <tr>
      <th>BHATNAGAR SANJAY</th>
      <td>15456290.0</td>
      <td>-2604490.0</td>
      <td>2604490.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
correct = ['restricted_stock_deferred','restricted_stock', 'exercised_stock_options', 'total_stock_value']
data.loc['BELFER ROBERT',correct] = np.array([-44093, 44093, 0, 0])
```


```python
correct = ['restricted_stock_deferred','restricted_stock', 'exercised_stock_options', 'total_stock_value']
data.loc['BHATNAGAR SANJAY',correct] = np.array([-2604490, 2604490, 15456290, 15456290])
```

O processo anterior foi realizado de maneira semelhante para os dados de ações.


```python
email = ['to_messages',
         'from_poi_to_this_person',
         'from_messages',
         'from_this_person_to_poi',
         'shared_receipt_with_poi']
```


```python
data[email].info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 146 entries, ALLEN PHILLIP K to YEAP SOON
    Data columns (total 5 columns):
    to_messages                86 non-null float64
    from_poi_to_this_person    86 non-null float64
    from_messages              86 non-null float64
    from_this_person_to_poi    86 non-null float64
    shared_receipt_with_poi    86 non-null float64
    dtypes: float64(5)
    memory usage: 11.8+ KB



```python
data[email+[label]].groupby(label).mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>to_messages</th>
      <th>from_poi_to_this_person</th>
      <th>from_messages</th>
      <th>from_this_person_to_poi</th>
      <th>shared_receipt_with_poi</th>
    </tr>
    <tr>
      <th>poi</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>False</th>
      <td>2007.111111</td>
      <td>58.500000</td>
      <td>668.763889</td>
      <td>36.277778</td>
      <td>1058.527778</td>
    </tr>
    <tr>
      <th>True</th>
      <td>2417.142857</td>
      <td>97.785714</td>
      <td>300.357143</td>
      <td>66.714286</td>
      <td>1783.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
imp = Imputer(np.nan)
data.loc[data[label] == 1, email] = imp.fit_transform(data[email][data[label]==1])
data.loc[data[label] == 0, email] = imp.fit_transform(data[email][data[label]==0])
```

Os atributos de e-mail apresentavam menos de 60% de completude. Para eles, optou-se por preencher os intervalos com a média dos valores de cada classe. A função "Imputer" da biblioteca _sklearn_ foi escolhida para o desenvolvimento da tarefa.


```python
payments = list(set(payments)-set(few_poi_values))
stock    = list(set(stock)-set(few_poi_values))
email    = list(set(email)-set(few_poi_values))
```


```python
data = data[[label]+payments+stock+email]
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>poi</th>
      <th>salary</th>
      <th>other</th>
      <th>expenses</th>
      <th>long_term_incentive</th>
      <th>bonus</th>
      <th>total_payments</th>
      <th>total_stock_value</th>
      <th>restricted_stock</th>
      <th>exercised_stock_options</th>
      <th>shared_receipt_with_poi</th>
      <th>from_this_person_to_poi</th>
      <th>to_messages</th>
      <th>from_poi_to_this_person</th>
      <th>from_messages</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ALLEN PHILLIP K</th>
      <td>False</td>
      <td>201955.0</td>
      <td>152.0</td>
      <td>13868.0</td>
      <td>304805.0</td>
      <td>4175000.0</td>
      <td>4484442.0</td>
      <td>1729541.0</td>
      <td>126027.0</td>
      <td>1729541.0</td>
      <td>1407.000000</td>
      <td>65.000000</td>
      <td>2902.000000</td>
      <td>47.0</td>
      <td>2195.000000</td>
    </tr>
    <tr>
      <th>BADUM JAMES P</th>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3486.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>182466.0</td>
      <td>257817.0</td>
      <td>0.0</td>
      <td>257817.0</td>
      <td>1058.527778</td>
      <td>36.277778</td>
      <td>2007.111111</td>
      <td>58.5</td>
      <td>668.763889</td>
    </tr>
    <tr>
      <th>BANNANTINE JAMES M</th>
      <td>False</td>
      <td>477.0</td>
      <td>864523.0</td>
      <td>56301.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>916197.0</td>
      <td>5243487.0</td>
      <td>1757552.0</td>
      <td>4046157.0</td>
      <td>465.000000</td>
      <td>0.000000</td>
      <td>566.000000</td>
      <td>39.0</td>
      <td>29.000000</td>
    </tr>
    <tr>
      <th>BAXTER JOHN C</th>
      <td>False</td>
      <td>267102.0</td>
      <td>2660303.0</td>
      <td>11200.0</td>
      <td>1586055.0</td>
      <td>1200000.0</td>
      <td>5634343.0</td>
      <td>10623258.0</td>
      <td>3942714.0</td>
      <td>6680544.0</td>
      <td>1058.527778</td>
      <td>36.277778</td>
      <td>2007.111111</td>
      <td>58.5</td>
      <td>668.763889</td>
    </tr>
    <tr>
      <th>BAY FRANKLIN R</th>
      <td>False</td>
      <td>239671.0</td>
      <td>69.0</td>
      <td>129142.0</td>
      <td>0.0</td>
      <td>400000.0</td>
      <td>827696.0</td>
      <td>63014.0</td>
      <td>145796.0</td>
      <td>0.0</td>
      <td>1058.527778</td>
      <td>36.277778</td>
      <td>2007.111111</td>
      <td>58.5</td>
      <td>668.763889</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.shape
```




    (146, 15)



Por fim, foram removidos os atributos listados por possuírem poucos dados de POIs, e as colunas do _dataset_ foram reordenadas.

---

## Outliers Investigation


```python
data.loc['LOCKHART EUGENE E']
```




    poi                          False
    salary                           0
    other                            0
    expenses                         0
    long_term_incentive              0
    bonus                            0
    total_payments                   0
    total_stock_value                0
    restricted_stock                 0
    exercised_stock_options          0
    shared_receipt_with_poi    1058.53
    from_this_person_to_poi    36.2778
    to_messages                2007.11
    from_poi_to_this_person       58.5
    from_messages              668.764
    Name: LOCKHART EUGENE E, dtype: object




```python
data.drop('LOCKHART EUGENE E', inplace=True)

```


```python
data.drop(['TOTAL', 'THE TRAVEL AGENCY IN THE PARK'], inplace=True)
```


```python
data.shape
```




    (143, 15)



Houve a preocupação de procurar e remover valores atípicos, como por exemplo o indivíduo "LOCKHART EUGENE E" que , excetuando o rótulo "poi", não possuia informação alguma. Além disso, durante o estudo do pdf disponibilizado, foram encontrados dois registros que não representavam pessoas, são eles: "TOTAL", "THE TRAVEL AGENCY IN THE PARK".

---

## Feature Engineer


```python
def KBestTable(sel, df, features):
    names = df[features].columns.values[sel.get_support()]
    scores = pd.Series(sel.scores_, names).sort_values(ascending=False)
    return scores
```


```python
features = payments+stock+email
sel = SelectKBest(f_classif, k = 'all').fit(data[features], data[label])
KBestTable(sel, data, features)
```




    total_stock_value          22.510549
    exercised_stock_options    22.348975
    bonus                      20.792252
    salary                     18.289684
    shared_receipt_with_poi    10.409148
    long_term_incentive         9.922186
    total_payments              9.283874
    restricted_stock            8.825442
    from_poi_to_this_person     5.478692
    expenses                    5.418900
    other                       4.202436
    from_this_person_to_poi     2.445551
    from_messages               1.050952
    to_messages                 0.660154
    dtype: float64



Os atributos de pagamento, ações e email foram avaliados com o auxílio da função "SelectKBest" do pacote "sklearn". Como parâmetros foram escolhidos o pontuador "f_classif", baseado em análise de variância, e "k" igual a "all" para pontuar todos os campos.


```python
data['ratio_from_poi'] = data.from_this_person_to_poi/data.from_messages
data['ratio_to_poi']   = data.from_poi_to_this_person/data.to_messages
data['ratio_with_poi'] = data.shared_receipt_with_poi/data.to_messages
new = ['ratio_with_poi', 'ratio_to_poi', 'ratio_from_poi']
```

Devido a baixa pontuação das colunas de e-mail novos campos foram criados com base nos existente, visando obter uma maior pontuação. Levantou-se a hipótese de que numeros brutos de e-mails recebidos e enviados podem não ser tão relevantes para identificação de pessoas de interesse quanto a proporção de e-mails enviados e recebidos entre os mesmos, ou seja, quanto mais um indivíduo se comunica com um poi, maior seria a chance desse mesmo também ser um, já que atividades ilicitas podem requerer cooperação de outros, o que os tornariam cúmplices e consequentemente pessoas de interesse.


```python
features = email+new
sel = SelectKBest(f_classif, k = 'all').fit(data[features], data[label])
KBestTable(sel, data, features)
```




    ratio_from_poi             25.878195
    ratio_with_poi             15.693633
    shared_receipt_with_poi    10.409148
    from_poi_to_this_person     5.478692
    ratio_to_poi                2.592766
    from_this_person_to_poi     2.445551
    from_messages               1.050952
    to_messages                 0.660154
    dtype: float64




```python
email = new
```

Os novos atributos foram avaliados em conjunto com os de e-mail. Nota-se uma melhora na pontuação dos campos criados em relação ao seus insumos, devido a isso deu-se preferência à utilização de tais campos em relação aos atributos de e-mail originais.

---

## Feature Selection


```python
features = payments+stock+email
sel = SelectKBest(f_classif, k = 'all').fit(data[features], data[label])
KBestTable(sel, data, features)
```




    ratio_from_poi             25.878195
    total_stock_value          22.510549
    exercised_stock_options    22.348975
    bonus                      20.792252
    salary                     18.289684
    ratio_with_poi             15.693633
    long_term_incentive         9.922186
    total_payments              9.283874
    restricted_stock            8.825442
    expenses                    5.418900
    other                       4.202436
    ratio_to_poi                2.592766
    dtype: float64




```python
features_list = [label]+features
features_list
```




    ['poi',
     'salary',
     'other',
     'expenses',
     'long_term_incentive',
     'bonus',
     'total_payments',
     'total_stock_value',
     'restricted_stock',
     'exercised_stock_options',
     'ratio_with_poi',
     'ratio_to_poi',
     'ratio_from_poi']



Foram selecionados os atributos de pagamentos, ações e as colunas de e-mails criadas no lugar das originais, além do rótulo, que será necessário para etapa de aprendizagem de máquina. 


---

## Algorithms


```python
my_dataset = data.to_dict(orient='index')
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
```

Após as etapas de análise, tratamento e seleção de dados, o _dataset_ foi transformado novamente em dicionário contando novamente com a ajuda do pacote _pandas_. Foram utilizadas as funções "featureFormat" e "targetFeatureSplit" disponibilizadas pela Udacity no pacote "feature_format". Como resultado, o dataset foi divido em rótulos e atributos.


```python
from sklearn.naive_bayes  import GaussianNB
from sklearn.tree         import DecisionTreeClassifier
from sklearn.ensemble     import RandomForestClassifier
```

Três algoritmos de aprendizagem de máquina foram selecionados: Naive Bayes, Árvore de decisão e Florestas aleatórias.


```python
cv = StratifiedShuffleSplit(n_splits=1000, random_state=42)
scoring = 'f1_macro'
```

Nesta seção, também foram definidos o algoritmo de validação cruzada e a métrica de avaliação para escolha dos melhores parâmetros de cada modelo.

---

## Parameters Tunning


```python
skb = {'SKB__k': ['all'] + list(range(2,len(features[0]),2))}

params = {}
params.update(skb)
params
```




    {'SKB__k': ['all', 2, 4, 6, 8, 10]}



Para iniciar o processo de escolha dos melhores parâmetros, foi adotado a função "SelectKBest" com os valores possíveis: 2, 4, 6, 8, 10 ou todos os atributos.


```python
t0 = time()

nb = {}
nb.update(params)
params2 = params.copy().update(nb)
pipe = Pipeline(steps=[('SKB', SelectKBest()), ('clf', GaussianNB())])
clf = GridSearchCV(pipe, param_grid = nb, cv=cv, scoring = scoring).fit(features, labels)
nb = clf.best_estimator_

print('Params Tunning:', round(time() - t0, 3), 'segundos')
print('Best Params: ', clf.best_params_)
```

    Params Tunning: 17.459 segundos
    Best Params:  {'SKB__k': 6}



```python
t0 = time()
test_classifier(nb, my_dataset, features_list)
print('Validation Time:', round(time() - t0, 3), 'segundos')
```

    Pipeline(memory=None,
         steps=[('SKB', SelectKBest(k=6, score_func=<function f_classif at 0x7ff21d33b2f0>)), ('clf', GaussianNB(priors=None))])
    	Accuracy: 0.84353	Precision: 0.39401	Recall: 0.32250	F1: 0.35469	F2: 0.33465
    	Total predictions: 15000	True positives:  645	False positives:  992	False negatives: 1355	True negatives: 12008
    
    Validation Time: 1.538 segundos


O classificador "GaussianNB" não necessitou de otimização de parâmetros. Ainda assim, foi afinado a quantidade de atributos na busca do melhor valor para a métrica f1.


```python
t0 = time()

dt = {'clf__criterion'        : ['gini', 'entropy'],
      'clf__max_depth'        : [2, 4, 6],
      'clf__min_samples_leaf' : [2, 4, 6],
      'clf__random_state'     : [42]}

dt.update(params)

pipe = Pipeline(steps=[('SKB', SelectKBest()), ('clf', DecisionTreeClassifier())])
clf = GridSearchCV(pipe, param_grid = dt, cv=cv, scoring=scoring).fit(features, labels)
dt = clf.best_estimator_

print('Params Tunning:', round(time() - t0, 3), 'segundos')
print('Best Params: ', clf.best_params_)
```

    Params Tunning: 362.644 segundos
    Best Params:  {'SKB__k': 'all', 'clf__criterion': 'entropy', 'clf__max_depth': 2, 'clf__min_samples_leaf': 2, 'clf__random_state': 42}



```python
t0 = time()
test_classifier(dt, my_dataset, features_list)
print ('Validation Time:', round(time() - t0, 3), 'segundos')
```

    Pipeline(memory=None,
         steps=[('SKB', SelectKBest(k='all', score_func=<function f_classif at 0x7ff21d33b2f0>)), ('clf', DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=2,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=2, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=42,
                splitter='best'))])
    	Accuracy: 0.89907	Precision: 0.57482	Recall: 0.93350	F1: 0.71151	F2: 0.82993
    	Total predictions: 15000	True positives: 1867	False positives: 1381	False negatives:  133	True negatives: 11619
    
    Validation Time: 1.444 segundos


Para a árvore de decisão, além da quantidade de atributos utilizados, também foram afinados os parâmetros "criterion", "max_depth" e "min_samples_leaf". Os melhores parâmetros foram selecionados pela função "GridSearchCV" e utilizados pela função de avaliação disponibilizada pela Udacity, "test_classifier".


```python
t0 = time()

rf = {'clf__n_estimators' : [10, 25, 50],
      'clf__criterion': ['entropy'],
      'clf__max_depth': [2],
      'clf__min_samples_leaf': [2],
      'clf__min_samples_split': [2],
      'clf__random_state': [42]}

rf.update(params)

pipe = Pipeline(steps=[('SKB', SelectKBest()), ('clf', RandomForestClassifier())])
clf = GridSearchCV(pipe, param_grid = rf, cv=cv, scoring=scoring).fit(features, labels)
rf = clf.best_estimator_

print('Params Tunning:', round(time() - t0, 3), 'segundos')
print('Best Params: ', clf.best_params_)
```

    Params Tunning: 604.336 segundos
    Best Params:  {'SKB__k': 10, 'clf__criterion': 'entropy', 'clf__max_depth': 2, 'clf__min_samples_leaf': 2, 'clf__min_samples_split': 2, 'clf__n_estimators': 50, 'clf__random_state': 42}



```python
t0 = time()
test_classifier(rf, my_dataset, features_list)
print ('Validation Time:', round(time() - t0, 3), 'segundos')
```

    Pipeline(memory=None,
         steps=[('SKB', SelectKBest(k=10, score_func=<function f_classif at 0x7ff21d33b2f0>)), ('clf', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                max_depth=2, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=No...stimators=50, n_jobs=1,
                oob_score=False, random_state=42, verbose=0, warm_start=False))])
    	Accuracy: 0.87673	Precision: 0.67933	Recall: 0.14300	F1: 0.23627	F2: 0.16981
    	Total predictions: 15000	True positives:  286	False positives:  135	False negatives: 1714	True negatives: 12865
    
    Validation Time: 51.513 segundos


Um procedimento semelhante ao usado no afinamento de parâmetros da árvore de decisão também foi utilizado para a floresta aleatória.

## Validation


```python
val = pd.DataFrame({'GaussianNB'   : [0.427, 0.324, 0.368],
                    'DecisionTree' : [0.577, 0.923, 0.710],
                    'RandomForest' : [0.623, 0.146, 0.237]},
                    index=['Precision', 'Recall', 'F1'])

val.T.sort_values(by='F1', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>DecisionTree</th>
      <td>0.577</td>
      <td>0.923</td>
      <td>0.710</td>
    </tr>
    <tr>
      <th>GaussianNB</th>
      <td>0.427</td>
      <td>0.324</td>
      <td>0.368</td>
    </tr>
    <tr>
      <th>RandomForest</th>
      <td>0.623</td>
      <td>0.146</td>
      <td>0.237</td>
    </tr>
  </tbody>
</table>
</div>




```python
dump_classifier_and_data(dt, my_dataset, features_list)
```

Como mostra a tabela acima, em relação à métrica F1, a árvore de decisão foi o algoritmo que se saiu melhor dentre os três estudados. O algoritmo tal como o _dataset_ e a lista de atributos foram serializados e salvos em disco com o auxílio da função "dump_classifier_and_data" disponibilizada pela Udacity por meio do pacote "teste.py".

## Conclusion

> 1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

R: O objetivo do projeto é identificar possíveis pessoas de interesses (POIs) da empresa Enron, ou seja, funcionários que possam ter participado da fraude que levou uma das maiores empresas dos EUA à falência. Para isso, um modelo preditivo pode ser construído a fim de auxiliar na identificação, modelado a partir de dados de indivíduos que já foram considerados POI, onde possíveis padrões possam encontrados e utilizados para detecção de outros.
Como insumo foram disponibilizados dados públicos com 14 atributos financeiros, 6 relacionados a e-mails e 1 rótulo pré-processados, referentes à 146 indivíduos. Infelizmente, haviam algumas informações faltando e pouca quantidade de pessoas de interesse (apenas 18).
Além disso, alguns registros foram removidos por serem considerados _outliers_, como por exemplo a linha "LOCKHART EUGENE E" que possuia apenas valores faltantes, e as linhas "TOTAL" e "THE TRAVEL AGENCY IN THE PARK" que não representavam pessoas.


> 1. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]

R: Todas as 21 colunas do _dataset_ foram avalidas utilizando uma métrica baseada em Análise de Variância (f_classif) em conjunto com uma função de seleção dos melhores parâmetros (SelectKBest), de acordo com tal métrica.
Devido a baixa relevância dos atributos de e-mail nesta avaliação, foram criados novos atributos utilizando aqueles como insumo, levando em consideração a hipótese de que a proporção de e-mails enviadas, recebidas e compartilhadas com pessoas de interesse possa ser mais relevante que a quantidade bruta desses valores. Os novos campos obtiveram pontuações melhores na avaliação, e portanto, foram usados em vez dos antigos.
Não foi necessário realizar dimensionamento dos dados, já que os algoritmos selecionados não demandavam tal processamento. A escolha de características para cada modelo foi feita automaticamente utilizando função para aprimoramento sistemático de parâmetros (GridSearchCV).


> 1. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

R: Foram escolhidos três algoritmos para geração dos modelos: Naive Bayes Gaussiano, Árvore de Decisão e Floresta Aleatória. Os algoritmos passaram por um processo de otimização de parâmetros, a fim de obter a melhor combinação deles que maximizasse o valor da métrica F1. Para isso o GridSearchCV foi utilizado para selecioanar o melhor número de caracteriscas e os melhores parâmetros de cada algoritmo (dentre os disponibilizados). O algoritmo de melhor desempenho foi a Árvore de Decisão, obtendo o valor 0.710 para a métrica F1, 0.923 para a métrica abrangência e 0.577 para a métrica precisão. O resultado foi satisfatório já que as duas últimas métricas apresentaram valores maiores que 0.3, atendendo assim aos requisitos do projeto.


> 1. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

R: Afinar os parâmetros de um algoritmo significa escolher os melhores valores para maximizar a pontuação de acordo com uma métrica definida, ou seja, escolher os melhores valores para obter o melhor modelo para aquele cenário. O processo foi feito sistemáticamente utilizando o GridSearchCV e listando possíveis valores para parâmetros selecionados, e repetido para os três algoritmos escolhidos. Além de selecionar a melhor quantidade de características a ser utilizada por cada modelo com ajuda de _pipeline_ utilizando função de mesmo nome presente no pacote _sklearn_. Algum dos parâmetros ajustados, não citados acima, foram: numero de árvores da floresta aleatória, critério de seleção, profundidade máxima e quantidade mínima de folhas da árvore de decisão.



> 1. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]

R: A validação pode ser utilizada para avaliar o nível de aprendizado de um modelo.É possível ensinar e treinar o modelo com a totalidade dos dados, o que pode acabar apresentando grande variância, fazendo com que o modelo apresente bons resultados para os dados de treinamento, mas que apresentem uma performance muito inferior para dados novos por não conseguir generalizar o aprendizado. Deste modo, a fim de minimizar o erro de generalização, foi utilizada validação cruzada com a função "StratifiedShuffleSplit" da biblioteca _sklearn_, por meio dessa foi possível misturar e separar os dados em conjuntos de treinamento e teste várias vezes.

> 1. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

R: Foram levadas em consideração três métricas para avaliação do modelo: precisão (precision), abrangência (recall) e F1 (F1). Precisão refere-se à quantidade de pessoas identificadas como POI pelo modelo, que eram realmente pessoas de interesses. Quanto à abrangência, entende-se que de todos os POIs existentes no conjunto de dados, quantos foram identificados. Já a métrica F1 representa uma média harmônica entre as duas outras métricas citadas.

# References

1. <https://github.com/MwillianM/FlchainExploration>
1. <https://github.com/MwillianM/OpenStreetMapDataWrangling>
1. <https://github.com/udacity/ud120-projects>
1. <https://br.udacity.com/>
1. <https://stackoverflow.com/>
1. <https://pandas.pydata.org/>
1. <http://www.numpy.org/>
1. <http://scikit-learn.org/>
1. <https://stats.stackexchange.com/>
1. <https://olegleyz.github.io/enron_classifier.html>
1. <https://medium.com/@williamkoehrsen/machine-learning-with-python-on-the-enron-dataset-8d71015be26d/>
1. <https://www.kaggle.com/tsilveira/machine-learning-tutorial-enron-e-mails>
