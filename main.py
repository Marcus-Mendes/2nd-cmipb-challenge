# main.py

import IgG_Prediction
import Monocytes_Prediction
import Gene_Expression_Prediction
import pandas as pd
import sys


##########################################################################################
#                                                                                        #
# main.py:                                                                               #
#                                                                                        #
# The main.py script is designed to execute three separate prediction analyses (IgG      #
# Prediction, Monocytes Prediction, and Gene Expression Prediction) and then merge       #
# their outputs into a single file. It requires command-line arguments to specify each   #
# analysis's input and output file paths.                                                #
#                                                                                        #
# How to Run:                                                                            #
#                                                                                        #
# To run this script, you must provide paths for the training and testing CSV files for  #
# each of the three analyses and an output directory. The script expects a total of      #
# seven command-line arguments in the following order: IgG training CSV, IgG testing     #
# CSV, Monocytes training CSV, Monocytes testing CSV, Gene expression training CSV,      #
# Gene expression testing CSV, output path for the temp and final results. The input     #
# data frame needs to include the following columns: subject_id,                         #
# planned_day_relative_to_boost, infancy_vac, biological_sex, race, age (need to         #
# calculate from the DOB), and all the columns related to the gene expression/cell       #
# quantification/Ig titers.                                                              #
#                                                                                        #
##########################################################################################


# Define file paths for each script
train_csv_1 = sys.argv[1]
test_csv_1 = sys.argv[2]
output_csv_1 = sys.argv[7]

train_csv_2 = sys.argv[3]
test_csv_2 = sys.argv[4]
output_csv_2 = sys.argv[7]

train_csv_3 = sys.argv[5]
test_csv_3 = sys.argv[6]
output_csv_3 = sys.argv[7]

# Run functions from each script
IgG_Prediction.run_analysis(train_csv_1, test_csv_1, output_csv_1)
Monocytes_Prediction.run_analysis(train_csv_2, test_csv_2, output_csv_2)
Gene_Expression_Prediction.run_analysis(train_csv_3, test_csv_3, output_csv_3)

# Read the outputs
df1 = pd.read_csv(f'{output_csv_1}/igg_fc_testing.csv')
df2 = pd.read_csv(f'{output_csv_2}/Monocytes_testing.csv')
df3 = pd.read_csv(f'{output_csv_3}/Gene-Expression_testing.csv')

# Merge the dataframes on 'subject_id'
merged_df = df1.merge(df2, on='subject_id').merge(df3, on='subject_id')

# Save the merged dataframe
merged_df.to_csv(f'{output_csv_1}/Final_result.csv', index=False)

