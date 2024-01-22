# 2nd-cmipb-challenge

How to Run main.py
The main.py script executes three distinct predictive analyses: IgG Prediction, Monocytes Prediction, and Gene Expression Prediction. This script mandates the command-line arguments to get the paths for input datasets and the output directory.

Usage Guidelines:
To run this script, the user needs to specify the file paths for the training and testing datasets across each of the trio of analyses. Additionally, a singular output directory needs to be specified. The script anticipates seven command-line arguments, arranged as follows:

1. IgG Training CSV: Path to the training dataset for IgG Prediction.
2. IgG Testing CSV: Path to the testing dataset for IgG Prediction.
3. Monocytes Training CSV: Path to the training dataset for Monocytes Prediction.
4. Monocytes Testing CSV: Path to the testing dataset for Monocytes Prediction.
5. Gene Expression Training CSV: Path to the training dataset for Gene Expression Prediction.
6. Gene Expression Testing CSV: Path to the testing dataset for Gene Expression Prediction.
7. Output Directory: The path to the directory where temporary and final results will be stored.

Data Requirements:
The input data frames for each analysis should encompass the following columns:
- subject_id
- planned_day_relative_to_boost
- infancy_vac
- biological_sex
- Race
- Age (The subject's age, calculated from the Date of Birth (DOB))
- Columns pertinent to gene expression, cell quantification, and immunoglobulin (Ig) titers.
