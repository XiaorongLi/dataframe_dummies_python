import unittest
from df_dummies import dfCategorical


df = pd.read_csv('./survey_results_public.csv')
dfdum = dfCategorical(df, 'Salary')

#test names of numerical variables

assertEqual(dfdum.num_vars, ['Respondent', 'CareerSatisfaction', 'JobSatisfaction', 'HoursPerWeek', \
	'StackOverflowSatisfaction', 'Salary', 'ExpectedSalary'], 'incorrect num_vars')

#test removing response column: no_res_df
assertEqual(dfdum.no_res_df.shape[0], 5009, 'incorrect no_res_df: removing response column failed')
assertEqual(dfdum.no_res_df.isnull().sum().sum(), 228037, 'incorrect number of missing values in no_res_df\
	likely due to unexpected imputation.')

#test imputed dataframe
assertEqual(dfdum.imputed_df[num_vars].isnull().sum().sum(), 5009, 'incorrect number of missing values in imputed_df')

#test final dataframe with dummy variables
##with imputation
assertEqual(dfdum.encode_dummies().shape[0], 5009, 'incorrect number of rows in final dataframe with dummy variables')
assertEqual(dfdum.encode_dummies().shape[1], 12079, 'incorrect number of columns in final dataframe with dummy variables')

assertEqual(dfdum.encode_dummies()[num_vars].isnull().sum().sum(), 5009, 'incorrect number of missing numerical\
 values in final dataframe with dummy variables when impute=True')

##without imputation
assertEqual(dfdum.encode_dummies(impute=False)[num_vars].isnull().sum().sum(), 8093, 'incorrect number of missing numerical\
 values in final dataframe with dummy variables when impute=False')

##dummy_na = True
assertEqual(dfdum.encode_dummies(dummy_na=False).shape[1], 11938, 'incorrect number of dummy variables in final dataframe,\
	likely due to wrong value of dummy_na'
