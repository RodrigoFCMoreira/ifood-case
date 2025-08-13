# ifood-case
Opção 1 - Recomendada

1 - Acessar o Databricks Community Edition: https://community.cloud.databricks.com/, fazer login
2 - No centro da tela clique na opçäo Main do Git
3 - clonar Todos os arquivos desse repositório 
    https://github.com/RodrigoFCMoreira/ifood-case.git
ou
    git@github.com:RodrigoFCMoreira/ifood-case.git

4 - Siga para /notebooks/1_data_processing.ipynb

5 - Descomente e rode a primeira linha !pip install -r ../requirements.txt
 
6 - Se desejar criar novamente as bases de variáveis explicativas (books) rode o notebook 1_data_processing.ipynb

7 - Para modelagem rode o 2_modeling.ipynb 

---------------------------------------------------------####---------------------------------------------------------------------------------
Opção 2
Caso queira baixar em seu ambiente local(näo recomendado) 
lembre que vc vai precisar ter o spark instalado localmente para rodar o notebook 1_data_processing.ipynb
Neste caso recomendo utilizar as bases já rodadas disponíveis ao clonar o repositório e rodar apenas o notebook 2_modeling.ipynb

No vscode:

0 - baixar e instalar o Python 3.11.9 https://www.python.org/downloads/release/python-3119/ (Windows installer (64-bit))

1 - cmd criar uma venv com a versão Python 3.11.9
    C:\caminho\para\python3.11.9\python -m venv meu_ambiente_venv
    meu_ambiente\Scripts\activate

2 - após ativar a venv no cmd rode o comando python -m pip install --upgrade pip

3 - Lembre-se de instalar o plugin Jupyter notebook no vscode

4 - Siga para /notebooks/1_data_processing.ipynb

5 - Descomente e rode a primeira linha #!pip install -r ../requirements.txt


### Todas as funções estão documentadas em funcoes_.py, funcoes.py e funcoes_pyspark##


