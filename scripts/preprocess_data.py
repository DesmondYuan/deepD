"""
This is the main script of data preprocessing.
For quick start, preprocessed data files can be found in the ./data/ directory and can be directly used for training.

Note:
    - Following is the script used for fetching gene expression data from the Gene Expression database of Normal
    and Tumor tissues 2 (GENT2).
    - The GENT2 database is from the publication Park, S., Yoon, B., Kim, S. et al. GENT2: an updated gene expression
    database for normal and tumor tissues. BMC Med Genomics 12, 101 (2019). https://doi.org/10.1186/s12920-019-0514-7
    - The database SQL we used here can be accessed at http://www.appex.kr/web_download/GENT2/GENT2_dump.sql.gz
    - Preprocessed data files can be found in the ./data/ directory
"""

import mysql.connector
import pandas as pd
import numpy as np
from collections import Counter


def query(db, cmd, csor=None, fetch=False):
    if csor is None:
        csor = db.cursor()
    print(csor.execute(cmd))
    if fetch:
        outs = csor.fetchall()
        print("Data query: ", str(outs)[:200])
    else:
        outs = None
    return outs


#### Step 1: Load data ####
# from SQL http://www.appex.kr/web_download/GENT2/GENT2_dump.sql.gz
mysql_server = "34.94.210.202"# Here the demo is using a private MySQL server hosting GENT2_dump.sql
sql = mysql.connector.connect(host=mysql_server, user="root")
cursor = sql.cursor()
query(sql, "USE gent2", cursor, fetch=False)

l1000_genes = pd.read_csv("data/L1000_reference.csv")['alias'].values.flatten()  # note the aliases of genes
gene_list_for_sql = "("
for gene in l1000_genes:
    gene_list_for_sql += "'{}', ".format(gene)
    gene_list_for_sql += "'{} ', ".format(gene)  # for a formatting issue of GENT2
gene_list_for_sql = gene_list_for_sql[:-2] + ")"   # "(GENE0, GENE1, ...)"

query_with_l1000 = "SELECT * FROM GeneProfile_U133Plus2_GeneSymbol WHERE probeID IN " + gene_list_for_sql
data = query(sql, query_with_l1000, cursor, fetch=True)
df = pd.DataFrame(data, columns=['Gene', 'Cell', 'Expression'])
df['Gene'] = [i.strip() for i in df['Gene']]  # GENT gene name might have random spaces before the actual name
df = pd.pivot_table(df, index='Cell', columns='Gene').astype(int)
df.to_csv('data/GENT2_U133Plus2_L1000_genes.csv')
# supposed to be 44,857 x (978, )

query(sql, "SHOW tables", cursor, fetch=True)
query(sql, "SHOW columns from Subtype_Meta", cursor, fetch=True)
labels = query(sql, "SELECT GSE, GSM, Tissue, Disease, Subtype from Subtype_Meta", cursor, fetch=True)
labels = pd.DataFrame(labels, columns=('GSE', 'GSM', 'Tissue', 'Disease', 'Subtype'))

diseases = set(labels['Disease'])  # all cancers
merged_df = df.join(labels)
merged_df.to_csv("")


#### Step 2: Remove outliers by the expression level of GAPDH ####
assert 'GAPDH' in l1000_genes
ct_gene_idx = [i for i, x in enumerate(l1000_genes) if x == 'GAPDH']
ct_gene_values = data.iloc[:, ct_gene_idx].values
ct_threhold = np.quantile(ct_gene_values, (0.05, 0.95))
ct_gene_filtered_pos = np.where([ct_threhold[0] < v < ct_threhold[1] for v in ct_gene_values])[0]
data1 = data.iloc[ct_gene_filtered_pos]


#### Step 3: Merge with labels and filter ####
data2 = data1.merge(labels, how='left', left_on=data1.index, right_on=labels.index, copy=False)
data2.dropna(inplace=True)
data2.drop_duplicates(subset='key_0', inplace=True)
data2.set_index('key_0', inplace=True)
data2.to_csv("data/Dataset1_GENT_L1000_U133Plus2.csv")
print(Counter(data2['feature_0']))


#### Step 4: random partition ####
np.random.seed(1234)
withheld_pos = np.random.choice(range(data2.shape[0]), int(0.1 * data2.shape[0]), replace=False)
experiment_pos = list(set(range(data2.shape[0])) - set(withheld_pos))
withheld_pos.sort()
data3_withheld, data3_experiment = data2.iloc[withheld_pos], data2.iloc[experiment_pos]
data3_withheld.to_csv("data/Dataset1_GENT_L1000_U133Plus2.withheld.csv")
data3_experiment.to_csv("data/Dataset1_GENT_L1000_U133Plus2.experiment.csv")
