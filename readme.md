Eron Fraud Identification
===

Python
---

---

**Author :** Matheus Willian Machado  
**Date :** Jul 30, 2018

---

Project Overview
---

>Banque o detetive e coloque suas habilidades de aprendizado de máquina em uso através da construção de um algoritmo para identificar funcionários da Enron que possam ter cometido fraude. Sua base será um conjunto de dados financeiros e de e-mail público da Enron.
> 
> (Udacity).

---

# Introduction

Em 2000, Enron era uma das maiores empresas dos Estados Unidos. Já em 2002, ela colapsou e quebrou devido a uma fraude que envolveu grande parte da corporação. Resultando em uma investigação federal, muitos dados que são normalmente confidenciais, se tornaram públicos, incluindo dezenas de milhares de e-mails e detalhes financeiros para os executivos dos mais altos níveis da empresa. Neste projeto, você irá bancar o detetive, e colocar suas habilidades na construção de um modelo preditivo que visará determinar se um funcionário é ou não um funcionário de interesse (POI). Um funcionário de interesse é um funcionário que participou do escândalo da empresa Enron. Para te auxiliar neste trabalho de detetive, nós combinamos os dados financeiros e sobre e-mails dos funcionários investigados neste caso de fraude, o que significa que eles foram indiciados, fecharam acordos com o governo, ou testemunharam em troca de imunidade no processo.

# Data Exploration

A resposta do estudante trata as características mais importantes do conjunto de dados e usa estas características para fazer as suas análises.
Características importantes incluem:

+ número total de data points
+ alocação entre classes (POI/non-POI)
+ número de características usadas
+ existem características com muitos valores faltando? etc.

---

# Outliers Investigation

O aluno identifica o(s) outlier(s) nos dados financeiros e explica como eles foram removidos ou tratados

# Feature Engineer

Pelo menos uma característica foi implementada. A justificativa para ela foi dada nas respostas escritas e o efeito desta característica na performance final foi testada. O aluno não precisa incluir a nova característica no conjunto de dados final.

# Feature Selection

Seleção de características univariadas ou recursivas foi feita ou as características foram escolhidas manualmente (diferente combinações de características foram feitas e o desempenho foi documentado para cada uma delas). Características selecionadas são documentadas e o número selecionado foi justificado. Para um algoritmo que suporta a verificação da importância das variáveis (ex. decision tree) ou pontuação das características (ex. SelectKBest), estas estão documentadas também.

+ SelectKbest

# Feature Scaling

Se o algoritmo requerir características com ajuste de escala, esta foi feita nos dados.

# Algorithm Choice

Pelo menos 2 algoritmos diferentes são usados e seus desempenhos são comparados, com o de melhor desempenho sendo usado no modelo final.

# Parameters Tunning

A resposta endereça o que significa fazer o afinamento dos parâmetros e porque é importante fazê-lo.

Pelo menos um parâmetro importante com 3 valores é investigado sistematicamente, ou qualquer dos seguintes são verdadeiros:

+ GridSearchCV usado para a busca do melhor parâmetro
+ Vários parâmetros são afinados
+ Busca de parâmetros incorporado na seleção do algoritmo (ex. parâmetros afinados para mais de um algoritmo e a melhor combinação algoritmo-parâmetros selecionada para a análise final).

# Validation

Pelo menos duas métricas apropriadas são usadas para avaliar a performance do algoritmo (ex: precisão e abrangência - precision and recall), e o aluno explica o que estas métricas medem no contexto desta tarefa.

A resposta explica o que é validação e porque ela é importante.

# Train/Test Split

O desempenho do modelo final é medido dividindo a base de dados entre base de treinamento e teste ou através do uso de validação cruzada (cross validation), especificando o tipo de validação usado.

# Cross Validation

Quando tester.py é usado para avaliar a performance, precision e recall são, os dois, ao menos 0.3.

# Conclusion

> 1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]
> 1. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]
> 1. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]
> 1. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]
> 1. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]
> 1. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]


# References

1. <https://stackoverflow.com/>
1. <https://pandas.pydata.org/>
1. <https://stats.stackexchange.com/>
1. <https://olegleyz.github.io/enron_classifier.html>
1. <https://medium.com/@williamkoehrsen/machine-learning-with-python-on-the-enron-dataset-8d71015be26d/>
1. <https://www.kaggle.com/tsilveira/machine-learning-tutorial-enron-e-mails>

---
Em 2000, Enron era uma das maiores empresas dos Estados Unidos. Já em 2002, ela colapsou e quebrou devido a uma fraude que envolveu grande parte da corporação. Resultando em uma investigação federal, muitos dados que são normalmente confidenciais, se tornaram públicos, incluindo dezenas de milhares de e-mails e detalhes financeiros para os executivos dos mais altos níveis da empresa. Neste projeto, você irá bancar o detetive, e colocar suas habilidades na construção de um modelo preditivo que visará determinar se um funcionário é ou não um funcionário de interesse (POI). Um funcionário de interesse é um funcionário que participou do escândalo da empresa Enron. Para te auxiliar neste trabalho de detetive, nós combinamos os dados financeiros e sobre e-mails dos funcionários investigados neste caso de fraude, o que significa que eles foram indiciados, fecharam acordos com o governo, ou testemunharam em troca de imunidade no processo.