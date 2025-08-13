# ifood-case
## Opção 1 - Recomendada

1 - Acessar o Databricks Community Edition: https://community.cloud.databricks.com/, fazer login<br>

2 - No centro da tela clique na opçäo Main do Git <br>

3 - clonar Todos os arquivos desse repositório https://github.com/RodrigoFCMoreira/ifood-case.git ou git@github.com:RodrigoFCMoreira/ifood-case.git<br>

4 - Siga para /notebooks/1_data_processing.ipynb<br>

5 - Descomente e rode a primeira linha !pip install -r ../requirements.txt<br>
 
6 - Se desejar criar novamente as bases de variáveis explicativas (books) rode o notebook 1_data_processing.ipynb<br>

7 - Para modelagem rode o 2_modeling.ipynb <br>

## Opção 2
Caso queira baixar em seu ambiente local(näo recomendado) <br>
lembre que vc vai precisar ter o spark instalado localmente para rodar o notebook 1_data_processing.ipynb<br>
Neste caso recomendo utilizar as bases já rodadas disponíveis ao clonar o repositório e rodar apenas o notebook 2_modeling.ipynb<br>

No vscode:<br>

0 - baixar e instalar o Python 3.11.9 https://www.python.org/downloads/release/python-3119/ (Windows installer (64-bit))<br>

1 - cmd criar uma venv com a versão Python 3.11.9 <br>

C:\caminho\para\python3.11.9\python -m venv meu_ambiente_venv<br>
meu_ambiente\Scripts\activate<br>

2 - após ativar a venv no cmd rode o comando python -m pip install --upgrade pip<br>

3 - Lembre-se de instalar o plugin Jupyter notebook no vscode<br>

4 - Siga para /notebooks/1_data_processing.ipynb<br>

5 - Descomente e rode a primeira linha #!pip install -r ../requirements.txt<br>


#### Todas as funções estão documentadas em funcoes_.py, funcoes.py e funcoes_pyspark##


