
import pandas as pd
df=pd.read_csv('/Users/koushikswarna/Downloads/Dermatology Project/dataverse_files/ISIC2018_Task3_Test_GroundTruth.csv')
print(df['dx'].unique().tolist())
