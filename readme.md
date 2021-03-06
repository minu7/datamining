# INSTALLATION
The following commands assume the working directory is the root of the project.
## Docker installation
0. needed docker installed on machine
1. ```docker-compose build```
2. run each command as ./python (to not run the python installed on machine but the python inside docker)
## On machine installation
0. needed python 3 installed on machine
1. ```pip install -r requirements.txt```
2. ```python download.py```

# Preprocessing
For now we have two scripts which prepare the bow (bag of words) from the excel provided.
The excel is in app/data and the scripts work with an excel formatted like this one.
1. Preparing the bow using the most N common words  
``` python app/cli/output_bow.py [N] [output file name] ```  
Example: to use 200 most common words and the output file name is data-200-most-common.csv  
``` python app/cli/output_bow.py 200 data-200-most-common.csv ```  

2. Preparing the bow using keywords provided in excel  
``` python app/cli/output_bow_keywords.py [output file name] ```  
Example: the output file name is data-keyword.csv  
``` python app/cli/output_bow_keywords.py data-keyword.csv ```  

# Import to MongoDB

Using docker:  
``` docker-compose run flask python app/excel_to_mongo/import_all.py ```  

Without docker:  
It's important check if MongoDB it's up and set the env variable DB to the link to the MongoDB service (probabily: localhost).  
The MongoDB service, in this development phase, must have the following credential:  
1. user: root
2. password: password
So, run:
``` python app/excel_to_mongo/import_all.py ```  
